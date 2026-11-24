"""Direct PyTorch / Transformers CUDA pure-S2S backend for `LiquidAI/LFM3-Omni-2B`.

Uses HuggingFace remote-code streaming APIs natively on a single GPU:
- Causal streaming log-mel extraction (`feature_extractor.stream`)
- Stateful LiveASR causal decoding (`model.transcribe_live`)
- Incremental audio prefill into persistent hybrid `DynamicCache` (`model.prefill_audio`)
- Native interleaved generation (`model.generate(..., generation_mode="interleaved")`)
- Frame-1 24 kHz neural audio decoding (`LiquidCodecModel.decode(..., streaming_cache=...)`)
- Exact hybrid Conv/KV snapshot rollback on mid-generation and post-generation barge-in

Zero `mlx` imports anywhere in this module.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .config import EngineSettings, PersonaConfig, ToolSpec
from .languages import get_language_profile, normalize_language_code
from .protocol import (
    BatchPlanOutcome,
    TurnOutcome,
    TurnRecord,
    append_transcript_piece,
    clean_display_text,
    float32_to_pcm16,
    has_real_speech,
    is_likely_agent_echo,
    is_semantically_complete,
    normalize_tool_calls_for_template,
    parse_router_query,
    parse_tool_calls,
    pcm16_to_float32,
    render_tool_calls,
    required_empty_chunks,
    truncate_assistant_turn_for_barge_in,
)

# ---------------------------------------------------------------------------
# Lazy Torch / Transformers & Snapshot Helpers (pure Python importable)
# ---------------------------------------------------------------------------


class _GenerationCancelled(Exception):
    """Raised cleanly inside HF streamers when cooperative barge-in triggers."""


def _try_torch() -> Any | None:
    try:
        import torch  # type: ignore[import-not-found]

        return torch
    except ImportError:
        return None


def _no_grad_ctx() -> Any:
    torch = _try_torch()
    if torch is not None:
        return torch.no_grad()
    from contextlib import nullcontext

    return nullcontext()


def _item_int(val: Any) -> int:
    if hasattr(val, "item"):
        return int(val.item())
    return int(val)


def _to_numpy_2d_float32(val: Any) -> np.ndarray:
    if val is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(val, "detach"):
        arr = val.detach().cpu().float().numpy()
    else:
        arr = np.asarray(val, dtype=np.float32)
    return arr.reshape(-1)


def _to_long_tensor(values: Any, device: Any = None) -> Any:
    torch = _try_torch()
    if torch is not None:
        if isinstance(values, torch.Tensor):
            return values.to(device=device, dtype=torch.long)
        return torch.as_tensor(values, dtype=torch.long, device=device)
    return np.asarray(values, dtype=np.int64)


def _to_device(tensor: Any, device: Any) -> Any:
    if tensor is None:
        return None
    if hasattr(tensor, "to") and device is not None:
        return tensor.to(device=device)
    return tensor


def _ones_mask(length: int, device: Any = None) -> Any:
    torch = _try_torch()
    if torch is not None:
        return torch.ones((1, int(length)), dtype=torch.long, device=device)
    return np.ones((1, int(length)), dtype=np.int64)


def _cat_dim1(a: Any, b: Any) -> Any:
    torch = _try_torch()
    if torch is not None and isinstance(a, torch.Tensor):
        return torch.cat((a, b.to(a.device, a.dtype)), dim=1)
    return np.concatenate((np.asarray(a), np.asarray(b)), axis=1)


def _clone_obj(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "clone") and callable(obj.clone):
        return obj.clone()
    return copy.deepcopy(obj)


def _snapshot_cache(cache: Any, attention_mask: Any) -> tuple[Any, Any]:
    return copy.deepcopy(cache), _clone_obj(attention_mask)


def _restore_cache(snapshot: tuple[Any, Any]) -> tuple[Any, Any]:
    cache, mask = snapshot
    return copy.deepcopy(cache), _clone_obj(mask)


def _cache_seq_len(cache: Any) -> int:
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    return int(getattr(cache, "seq_len", 0))


def _resolve_token_id(tokenizer: Any, token: str, default: int = -1) -> int:
    try:
        tid = tokenizer.convert_tokens_to_ids(token)
        return int(tid) if tid is not None and int(tid) >= 0 else default
    except Exception:
        return default


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    return [int(x) for x in ids]


def _make_streaming_codec_cache(codec: Any) -> Any:
    make_fn = getattr(codec, "make_streaming_cache", None)
    if callable(make_fn):
        return make_fn()
    mod = sys.modules.get(type(codec).__module__)
    cache_cls = getattr(mod, "LiquidCodecStreamingCache", None) if mod else None
    if cache_cls is not None:
        return cache_cls(codec.config)
    return None


def _make_dynamic_cache(model: Any) -> Any:
    make_fn = getattr(model, "make_conv_cache", None)
    if callable(make_fn):
        return make_fn()
    from transformers.cache_utils import DynamicCache

    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
    return DynamicCache(config=text_cfg) if text_cfg is not None else DynamicCache()


def _select_torch_device(requested: str) -> str:
    req = (requested or "auto").strip().lower()
    if req != "auto":
        return req
    torch = _try_torch()
    if torch is not None:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Streaming Codec & Cooperative Slot Helpers
# ---------------------------------------------------------------------------


class _SlotCancelController:
    """Shared slot counter triggering cooperative mic polling & immediate abort."""

    def __init__(
        self,
        cancel_event: Any,
        poll_barge_in: Optional[Callable[[], None]],
        poll_interval: int = 32,
    ) -> None:
        self.cancel_event = cancel_event
        self.poll_barge_in = poll_barge_in
        self.poll_interval = max(1, int(poll_interval))
        self.slots = 0

    def check_and_step(self) -> None:
        if self.cancel_event.is_set():
            raise _GenerationCancelled()
        self.slots += 1
        if (
            self.poll_barge_in is not None
            and self.slots % self.poll_interval == 0
        ):
            self.poll_barge_in()
            if self.cancel_event.is_set():
                raise _GenerationCancelled()


class LiquidCodecStreamingFlusher:
    """Frame-1 streaming 24 kHz LiquidCodec wrapper for PyTorch code frames."""

    def __init__(
        self,
        codec: Any,
        on_audio_chunk: Callable[[bytes], None],
        *,
        first_chunk_frames: int = 1,
        stream_chunk_frames: int = 2,
        num_codebooks: int = 12,
        codebook_size: int = 2017,
        audio_eos_code: int = 2016,
        device: Any = None,
    ) -> None:
        self.codec = codec
        self.on_audio_chunk = on_audio_chunk
        self.first_chunk_frames = max(1, int(first_chunk_frames))
        self.stream_chunk_frames = max(1, int(stream_chunk_frames))
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.audio_eos_code = int(audio_eos_code)
        self.device = device

        self._cache = _make_streaming_codec_cache(codec)
        self._buf: list[Any] = []
        self._eos_seen = False
        self._finished = False
        self.audio_chunks_sent = 0
        self.first_chunk_samples = 0
        self.total_frames = 0
        self.total_pcm_bytes = 0
        self.chunk_decode_ms: list[float] = []

    def add_frame(self, codes: Any) -> bool:
        """Buffer one `(num_codebooks,)` frame; return False if EOS or invalid."""
        if self._eos_seen or self._finished:
            return False
        arr = (
            codes.detach().cpu().numpy()
            if hasattr(codes, "detach")
            else np.asarray(codes)
        ).reshape(-1)
        if arr.size != self.num_codebooks:
            return False
        if bool(
            np.any(arr == self.audio_eos_code)
            or np.any(arr < 0)
            or np.any(arr >= self.codebook_size)
        ):
            self._eos_seen = True
            return False
        self._buf.append(arr.astype(np.int64, copy=True))
        self.total_frames += 1
        threshold = (
            self.first_chunk_frames
            if self.audio_chunks_sent == 0
            else self.stream_chunk_frames
        )
        if len(self._buf) >= threshold:
            self._flush(finalize=False)
        return True

    def _decode_and_emit(self, codes_np: np.ndarray, *, finalize: bool) -> None:
        t0 = time.perf_counter()
        codes_tensor = _to_long_tensor(codes_np, device=self.device)
        with _no_grad_ctx():
            try:
                out = self.codec.decode(
                    codes_tensor,
                    streaming_cache=self._cache,
                    finalize=finalize,
                    return_dict=True,
                )
            except TypeError:
                out = self.codec.decode(
                    codes_tensor,
                    streaming_cache=self._cache,
                    finalize=finalize,
                )
        wav_raw = getattr(out, "audio_values", out)
        wav_f32 = _to_numpy_2d_float32(wav_raw)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        pcm = float32_to_pcm16(wav_f32)
        if pcm:
            self.chunk_decode_ms.append(elapsed_ms)
            if self.audio_chunks_sent == 0:
                self.first_chunk_samples = len(pcm) // 2
            self.audio_chunks_sent += 1
            self.total_pcm_bytes += len(pcm)
            self.on_audio_chunk(pcm)

    def _flush(self, *, finalize: bool) -> None:
        if not self._buf:
            return
        # Shape: (batch=1, num_codebooks, num_frames)
        stacked = np.stack(self._buf, axis=-1)[None, :, :]
        self._buf.clear()
        self._decode_and_emit(stacked, finalize=finalize)

    def finish(self) -> None:
        """Flush buffered frames and any trailing overlap-add ISTFT samples."""
        if self._finished:
            return
        self._finished = True
        if self._buf:
            self._flush(finalize=True)
            return
        started = bool(getattr(self._cache, "started", False))
        cache_done = bool(getattr(self._cache, "finished", False))
        if started and not cache_done:
            empty = np.zeros((1, self.num_codebooks, 0), dtype=np.int64)
            self._decode_and_emit(empty, finalize=True)


class _SlotTextStreamer:
    """HF `BaseStreamer`-compatible text streamer skipping initial prompt echo."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        slot_ctrl: _SlotCancelController,
        shared_token_ids: list[int],
        on_text_delta: Callable[[str], None],
    ) -> None:
        self.tokenizer = tokenizer
        self.slot_ctrl = slot_ctrl
        self.shared_token_ids = shared_token_ids
        self.on_text_delta = on_text_delta
        self._prompt_seen = False
        self._gen_ids: list[int] = []
        self._emitted_len = 0

    def put(self, value: Any) -> None:
        self.slot_ctrl.check_and_step()
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        arr = (
            value.detach().cpu().reshape(-1).tolist()
            if hasattr(value, "detach")
            else np.asarray(value).reshape(-1).tolist()
        )
        for tid in arr:
            tok = int(tid)
            self.shared_token_ids.append(tok)
            self._gen_ids.append(tok)
        decoded = self.tokenizer.decode(self._gen_ids, skip_special_tokens=False)
        if decoded.endswith("\ufffd"):
            return
        if len(decoded) > self._emitted_len:
            delta = decoded[self._emitted_len :]
            self._emitted_len = len(decoded)
            if delta:
                self.on_text_delta(delta)

    def end(self) -> None:
        if not self._gen_ids:
            return
        decoded = self.tokenizer.decode(self._gen_ids, skip_special_tokens=False)
        if len(decoded) > self._emitted_len:
            delta = decoded[self._emitted_len :]
            self._emitted_len = len(decoded)
            if delta:
                self.on_text_delta(delta)


class _SlotAudioStreamer:
    """HF `BaseStreamer`-compatible audio frame streamer feeding `LiquidCodecStreamingFlusher`."""

    def __init__(
        self,
        flusher: LiquidCodecStreamingFlusher,
        *,
        slot_ctrl: _SlotCancelController,
        audio_out_token_id: int,
        shared_token_ids: list[int],
        shared_codes: list[np.ndarray],
    ) -> None:
        self.flusher = flusher
        self.slot_ctrl = slot_ctrl
        self.audio_out_token_id = int(audio_out_token_id)
        self.shared_token_ids = shared_token_ids
        self.shared_codes = shared_codes

    def put(self, value: Any) -> None:
        self.slot_ctrl.check_and_step()
        arr = (
            value.detach().cpu().numpy()
            if hasattr(value, "detach")
            else np.asarray(value)
        )
        if arr.ndim == 2:
            frames = [arr[i] for i in range(arr.shape[0])]
        else:
            frames = [arr.reshape(-1)]
        for frame in frames:
            if self.flusher.add_frame(frame):
                self.shared_token_ids.append(self.audio_out_token_id)
                self.shared_codes.append(frame.astype(np.int32, copy=True).reshape(-1))

    def end(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Stateful Causal LiveASR Listener
# ---------------------------------------------------------------------------


class TransformersLiveS2SListener:
    """Always-streaming causal LiveASR listener backed by HF `transcribe_live`."""

    LIVE_ASR_PROMPT = (
        "<|startoftext|><|im_start|>system\n"
        "Perform live ASR.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        feature_extractor: Any,
        *,
        device: Any = None,
        live_chunk_tokens: int = 12,
        idle_reset_sec: float = 15.0,
        hard_reset_sec: float = 30.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.device = device
        self.live_chunk_tokens = max(4, int(live_chunk_tokens))
        self.idle_reset_sec = float(idle_reset_sec)
        self.hard_reset_sec = float(hard_reset_sec)

        # 12 tokens * 8 frames/tok * 160 samples/frame * 2 bytes = 30720 B (960 ms @ 16 kHz)
        self.block_bytes: int = self.live_chunk_tokens * 8 * 160 * 2
        self._live_prompt_ids = _to_long_tensor(
            [_encode_ids(tokenizer, self.LIVE_ASR_PROMPT)],
            device=device,
        )

        suppress: set[int] = set()
        for tok in ("<|mixed_end|>", "<|endoftext|>"):
            tid = _resolve_token_id(tokenizer, tok)
            if tid >= 0:
                suppress.add(tid)
        self._suppress_ids = sorted(suppress)

        stop: set[int] = set()
        for tok in ("<|endoftext|>", "<|im_start|>"):
            tid = _resolve_token_id(tokenizer, tok)
            if tid >= 0:
                stop.add(tid)
        self._stop_ids = sorted(stop)

        self._feat_cache: Any = None
        self._live_state: Any = None
        self._needs_context_restart: bool = False
        self._audio_tokens_in_window: int = 0
        self._active_sec: float = 0.0
        self._idle_sec: float = 0.0
        self.reset_count: int = 0

    @property
    def window_seconds(self) -> float:
        if self._audio_tokens_in_window > 0:
            return round(self._audio_tokens_in_window / 12.5, 2)
        return round(self._active_sec, 2)

    def reset_window(self) -> None:
        self._feat_cache = None
        self._live_state = None
        self._needs_context_restart = False
        self._audio_tokens_in_window = 0
        self._active_sec = 0.0
        self._idle_sec = 0.0
        self.reset_count += 1

    def _run_transcribe_step(
        self,
        pcm16_block: bytes,
        feat_cache: Any,
        live_state: Any,
        *,
        restart_context: bool,
    ) -> tuple[str, Any, Any, int]:
        wav_f32 = pcm16_to_float32(pcm16_block)
        feat_out, new_feat_cache = self.feature_extractor.stream(
            wav_f32,
            cache=feat_cache,
            is_final=False,
            sampling_rate=16000,
        )
        input_features = _to_device(feat_out["input_features"], self.device)
        if input_features.shape[-1] == 0:
            return "", new_feat_cache, live_state, 0

        prompt_ids = (
            self._live_prompt_ids
            if live_state is None or restart_context
            else None
        )
        with _no_grad_ctx():
            out = self.model.transcribe_live(
                input_features=input_features,
                state=live_state,
                input_ids=prompt_ids,
                is_final=False,
                max_new_tokens=48,
                suppress_token_ids=self._suppress_ids,
                stop_token_ids=self._stop_ids,
            )
        seq = getattr(out, "sequences", None)
        toks = (
            seq[0].detach().cpu().tolist()
            if seq is not None and hasattr(seq, "detach")
            else (np.asarray(seq[0]).tolist() if seq is not None else [])
        )
        text = self.tokenizer.decode(toks, skip_special_tokens=True) if toks else ""
        num_tokens = int(getattr(out, "num_audio_tokens", 0) or 0)
        return text, new_feat_cache, out.state, num_tokens

    def step_block(self, pcm16_block: bytes) -> str:
        """Transcribe one causal streaming audio block via persistent LiveASR."""
        if not pcm16_block:
            return ""
        text, self._feat_cache, self._live_state, num_tokens = (
            self._run_transcribe_step(
                pcm16_block,
                self._feat_cache,
                self._live_state,
                restart_context=self._needs_context_restart,
            )
        )
        self._needs_context_restart = False
        dur_sec = len(pcm16_block) / 32000.0
        self._audio_tokens_in_window += max(0, num_tokens)
        self._active_sec += dur_sec
        if has_real_speech(text):
            self._idle_sec = 0.0
            return text
        self._idle_sec += dur_sec
        return ""

    def probe_silence(
        self, pcm16_half_block: bytes
    ) -> tuple[bool, Any, Any, int, float]:
        """Non-mutating check on a short trailing block for early endpointing."""
        if not pcm16_half_block or self._live_state is None:
            return False, None, None, 0, 0.0
        feat_copy = copy.deepcopy(self._feat_cache)
        state_copy = copy.deepcopy(self._live_state)
        text, feat_after, state_after, num_toks = self._run_transcribe_step(
            pcm16_half_block,
            feat_copy,
            state_copy,
            restart_context=self._needs_context_restart,
        )
        dur_sec = len(pcm16_half_block) / 32000.0
        return (not has_real_speech(text)), feat_after, state_after, num_toks, dur_sec

    def adopt_probe_state(
        self, feat_after: Any, state_after: Any, num_toks: int, dur_sec: float
    ) -> None:
        """Commit an accepted silence probe so zero encoder work is duplicated."""
        self._feat_cache = feat_after
        self._live_state = state_after
        self._needs_context_restart = False
        self._audio_tokens_in_window += max(0, num_toks)
        self._active_sec += dur_sec
        self._idle_sec += dur_sec

    def maybe_reset(self, is_idle: bool) -> bool:
        """Roll over LiveASR LLM context cleanly at idle (15 s) or hard (30 s) limits."""
        if self._live_state is None and self._feat_cache is None:
            return False
        if self.window_seconds >= self.hard_reset_sec:
            if is_idle:
                self.reset_window()
            else:
                # Roll over the LLM KV window while preserving causal encoder state.
                self._needs_context_restart = True
                self._audio_tokens_in_window = 0
                self._active_sec = 0.0
                self.reset_count += 1
            return True
        if is_idle and self._idle_sec >= self.idle_reset_sec:
            self.reset_window()
            return True
        return False


# ---------------------------------------------------------------------------
# Streaming Conversation Session (Incremental Prefill + Interleaved S2S)
# ---------------------------------------------------------------------------


class TransformersStreamingConversationSession:
    """Single-interface pure-S2S session using persistent PyTorch KV/Conv caches."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        tokenizer: Any,
        feature_extractor: Any,
        codec: Any,
        settings: EngineSettings,
        *,
        device: Any = None,
        model_lock: Optional[threading.RLock] = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.codec = codec
        self.settings = settings
        self.device = device
        self._model_lock = model_lock or threading.RLock()

        self.listener = TransformersLiveS2SListener(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=device,
            live_chunk_tokens=settings.live_chunk_tokens,
            idle_reset_sec=settings.listener_idle_reset_sec,
            hard_reset_sec=settings.listener_hard_reset_sec,
        )

        cfg = getattr(model, "config", None)
        self._audio_out_token_id = int(getattr(cfg, "audio_out_token_id", 125032))
        self._audio_eos_code = int(getattr(cfg, "audio_eos_code", 2016))
        self._num_codebooks = int(getattr(cfg, "num_codebooks", 12))
        self._codebook_size = int(getattr(cfg, "codebook_size", 2017))
        self._mixed_start_id = int(getattr(cfg, "mixed_start_token_id", 125015))
        self._im_end_id = _resolve_token_id(tokenizer, "<|im_end|>", 124900)
        self._newline_ids = _encode_ids(tokenizer, "\n")

        self._vision_start_tok = self._pick_valid_token(
            getattr(processor, "vision_start_token", "<|vision_start|>"),
            fallback="<|vision_start|>",
        )
        self._vision_end_tok = self._pick_valid_token(
            getattr(processor, "vision_end_token", "<|vision_end|>"),
            fallback="<|vision_end|>",
        )

        self.system_prompt: str = ""
        self.tools: list[ToolSpec] | None = None
        self.language: str = "en"
        self.history: list[TurnRecord] = []
        self.compaction_count: int = 0
        self._rolling_summary: str = ""
        self._pending_filler_text: str = ""

        self._conv_cache: Any = None
        self._conv_attention_mask: Any = None
        self._pre_user_snapshot: Optional[tuple[Any, Any]] = None
        self._pre_turn_snapshot: Optional[tuple[Any, Any]] = None

        self._pcm_buf = bytearray()
        self._listen_lock = threading.Lock()
        self._pre_speech_blocks: deque[bytes] = deque(
            maxlen=max(1, int(settings.pre_speech_blocks))
        )
        self.user_speaking: bool = False
        self.user_turn_open: bool = False
        self.active_user_turn_id: Optional[str] = None
        self.active_user_transcript: str = ""
        self.active_user_pcm_blocks: list[bytes] = []
        self._utt_feat_cache: Any = None
        self._utt_enc_cache: Any = None
        self.pending_barge_in_pcm: Optional[list[bytes]] = None
        self.empty_chunk_streak: int = 0
        self.user_blocks_in_turn: int = 0
        self._last_speech_block_ts: float = 0.0
        self._half_stride_probed_blocks: int = -1
        self.max_user_blocks: int = max(4, int(round(settings.max_user_turn_sec / 0.96)))

    def _pick_valid_token(self, preferred: str, *, fallback: str) -> str:
        if _resolve_token_id(self.tokenizer, preferred) >= 0:
            return preferred
        if _resolve_token_id(self.tokenizer, fallback) >= 0:
            return fallback
        return preferred

    @property
    def conv_tokens(self) -> int:
        if self._conv_cache is not None:
            cached = _cache_seq_len(self._conv_cache)
            if cached > 0:
                return cached
        total_chars = len(self.system_prompt) + len(self._rolling_summary)
        for t in self.history:
            total_chars += len(t.text or "")
            if t.user_pcm16:
                total_chars += int((len(t.user_pcm16) / 32000.0) * 50)
        return max(1, total_chars // 4)

    def _tokenize(self, text: str) -> Any:
        return _to_long_tensor([_encode_ids(self.tokenizer, text)], device=self.device)

    def _render_system_block(self) -> str:
        parts = [self.system_prompt]
        if self._rolling_summary:
            parts.append(f"EARLIER CALL SUMMARY:\n{self._rolling_summary}")
        content = "\n\n".join(p for p in parts if p)
        if self.tools:
            schemas = [t.to_openai_schema()["function"] for t in self.tools]
            tool_lines = [json.dumps(s, ensure_ascii=False) for s in schemas]
            content = (
                f"{content}\n\n<tools>\n" + "\n".join(tool_lines) + "\n</tools>"
                if content
                else "<tools>\n" + "\n".join(tool_lines) + "\n</tools>"
            )
        return f"<|startoftext|><|im_start|>system\n{content}<|im_end|>\n"

    def _prefill_token_ids(self, ids: Any) -> None:
        if ids is None or ids.shape[1] == 0:
            return
        with self._model_lock, _no_grad_ctx():
            past_len = _cache_seq_len(self._conv_cache)
            new_ones = _ones_mask(ids.shape[1], device=self.device)
            if self._conv_attention_mask is None or self._conv_attention_mask.shape[1] != past_len:
                base_mask = _ones_mask(past_len, device=self.device)
            else:
                base_mask = self._conv_attention_mask
            extended_mask = _cat_dim1(base_mask, new_ones)
            pos_ids = _to_long_tensor(
                [list(range(past_len, past_len + int(ids.shape[1])))],
                device=self.device,
            )
            out = self.model.model(
                input_ids=ids,
                attention_mask=extended_mask,
                position_ids=pos_ids,
                past_key_values=self._conv_cache,
                use_cache=True,
            )
            self._conv_cache = out.past_key_values
            self._conv_attention_mask = _ones_mask(
                _cache_seq_len(self._conv_cache), device=self.device
            )

    def _prefill_text(self, text: str) -> None:
        if not text:
            return
        self._prefill_token_ids(self._tokenize(text))

    def _prefill_embeds(self, embeds: Any) -> None:
        if embeds is None or embeds.shape[1] == 0:
            return
        with self._model_lock, _no_grad_ctx():
            past_len = _cache_seq_len(self._conv_cache)
            new_len = int(embeds.shape[1])
            base_mask = _ones_mask(past_len, device=self.device)
            extended_mask = _cat_dim1(base_mask, _ones_mask(new_len, device=self.device))
            pos_ids = _to_long_tensor(
                [list(range(past_len, past_len + new_len))],
                device=self.device,
            )
            out = self.model.model(
                inputs_embeds=embeds,
                attention_mask=extended_mask,
                position_ids=pos_ids,
                past_key_values=self._conv_cache,
                use_cache=True,
            )
            self._conv_cache = out.past_key_values
            self._conv_attention_mask = _ones_mask(
                _cache_seq_len(self._conv_cache), device=self.device
            )

    def _assistant_turn_embeds(self, turn: TurnRecord) -> Any:
        torch = _try_torch()
        has_embed_fn = hasattr(self.model, "get_input_embeddings") and hasattr(
            getattr(self.model, "model", None), "get_audio_out_embeds"
        )
        if (
            torch is not None
            and has_embed_fn
            and turn.raw_token_ids
            and turn.raw_audio_codes is not None
        ):
            raw_ids = list(turn.raw_token_ids)
            raw_codes = np.asarray(turn.raw_audio_codes, dtype=np.int64)
            placeholders = sum(1 for t in raw_ids if t == self._audio_out_token_id)
            if placeholders == raw_codes.shape[0] and placeholders > 0:
                if raw_ids[-1] != self._im_end_id:
                    raw_ids.append(self._im_end_id)
                prefix_ids = _encode_ids(
                    self.tokenizer, "<|im_start|>assistant\n<think></think>"
                )
                full_ids = prefix_ids + raw_ids + self._newline_ids
                ids_t = torch.tensor([full_ids], dtype=torch.long, device=self.device)
                embeds = self.model.get_input_embeddings()(ids_t)
                mask = ids_t == self._audio_out_token_id
                codes_t = torch.as_tensor(raw_codes, dtype=torch.long, device=self.device)
                audio_embs = self.model.model.get_audio_out_embeds(codes_t).to(embeds.dtype)
                embeds[mask] = audio_embs
                return embeds

        rendered = f"<|im_start|>assistant\n<think></think>{turn.text.strip()}<|im_end|>\n"
        ids = self._tokenize(rendered)
        if torch is not None and hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()(ids)
        return ids

    def _stream_pcm_into_conv_cache(self, pcm16_bytes: bytes, *, is_final: bool) -> None:
        if not pcm16_bytes and not is_final:
            return
        with self._model_lock, _no_grad_ctx():
            wav_f32 = pcm16_to_float32(pcm16_bytes)
            feat_out, self._utt_feat_cache = self.feature_extractor.stream(
                wav_f32,
                cache=self._utt_feat_cache,
                is_final=is_final,
                sampling_rate=16000,
            )
            mel = _to_device(feat_out["input_features"], self.device)
            if mel.shape[-1] == 0 and not is_final:
                return
            out = self.model.prefill_audio(
                input_features=mel,
                past_key_values=self._conv_cache,
                encoder_cache=self._utt_enc_cache,
                attention_mask=self._conv_attention_mask,
                is_final=is_final,
            )
            self._conv_cache = out.past_key_values
            self._utt_enc_cache = out.encoder_cache
            self._conv_attention_mask = (
                out.attention_mask
                if getattr(out, "attention_mask", None) is not None
                else _ones_mask(_cache_seq_len(self._conv_cache), device=self.device)
            )

    def _rebuild_cache_from_history(self) -> None:
        with self._model_lock:
            self._conv_cache = _make_dynamic_cache(self.model)
            self._conv_attention_mask = _ones_mask(0, device=self.device)
            self._pre_turn_snapshot = None
            self._pre_user_snapshot = None
            self._prefill_text(self._render_system_block())

            recent_audio_indices: set[int] = set()
            audio_quota = max(1, int(self.settings.keep_recent_user_audio_turns))
            for idx in range(len(self.history) - 1, -1, -1):
                if self.history[idx].role == "user" and self.history[idx].user_pcm16:
                    recent_audio_indices.add(idx)
                    audio_quota -= 1
                    if audio_quota <= 0:
                        break

            for idx, turn in enumerate(self.history):
                if turn.role == "user":
                    if idx in recent_audio_indices and turn.user_pcm16:
                        self._utt_feat_cache = None
                        self._utt_enc_cache = None
                        self._prefill_text(f"<|im_start|>user\n{self._vision_start_tok}")
                        self._stream_pcm_into_conv_cache(turn.user_pcm16, is_final=True)
                        self._prefill_text(f"{self._vision_end_tok}<|im_end|>\n")
                        self._utt_feat_cache = None
                        self._utt_enc_cache = None
                    else:
                        self._prefill_text(f"<|im_start|>user\n{turn.text}<|im_end|>\n")
                elif turn.role == "assistant":
                    if turn.tool_calls:
                        norm = normalize_tool_calls_for_template(turn.tool_calls)
                        self._prefill_text(
                            f"<|im_start|>assistant\n<think></think>{render_tool_calls(norm)}<|im_end|>\n"
                        )
                    else:
                        embeds = self._assistant_turn_embeds(turn)
                        if hasattr(self.model, "get_input_embeddings") and _try_torch() is not None:
                            self._prefill_embeds(embeds)
                        else:
                            self._prefill_token_ids(embeds)
                elif turn.role == "tool":
                    self._prefill_text(
                        f"<|im_start|>tool\n<tool_response>\n{turn.text}\n</tool_response><|im_end|>\n"
                    )

    def initialize_conversation(
        self,
        system_prompt: str,
        tools: list[ToolSpec] | None,
        language: str = "en",
    ) -> None:
        self.system_prompt = system_prompt
        self.tools = list(tools) if tools else None
        self.language = normalize_language_code(language)
        self.history.clear()
        self.compaction_count = 0
        self._rolling_summary = ""
        self._pending_filler_text = ""
        with self._listen_lock:
            self._pcm_buf.clear()
            self._pre_speech_blocks.clear()
            self.user_speaking = False
            self.user_turn_open = False
            self.active_user_turn_id = None
            self.active_user_transcript = ""
            self.active_user_pcm_blocks.clear()
            self._utt_feat_cache = None
            self._utt_enc_cache = None
            self.pending_barge_in_pcm = None
            self.empty_chunk_streak = 0
            self.user_blocks_in_turn = 0
            self._last_speech_block_ts = 0.0
            self._half_stride_probed_blocks = -1
            self.listener.reset_window()
            self.listener.reset_count = 0
        self._rebuild_cache_from_history()

    def update_system_config(
        self,
        system_prompt: str,
        tools: list[ToolSpec] | None,
        language: str = "en",
    ) -> None:
        self.system_prompt = system_prompt
        self.tools = list(tools) if tools else None
        self.language = normalize_language_code(language)
        if not self.user_speaking and not self.user_turn_open:
            self._rebuild_cache_from_history()

    def append_text_user_turn(self, text: str) -> None:
        clean = (text or "").strip()
        if not clean:
            return
        self._prefill_text(f"<|im_start|>user\n{clean}<|im_end|>\n")
        self.history.append(TurnRecord(role="user", text=clean))

    def append_tool_exchange(self, calls: list[dict[str, Any]], results_json: str) -> None:
        norm_calls = normalize_tool_calls_for_template(calls)
        self.history.append(
            TurnRecord(
                role="assistant",
                text=render_tool_calls(norm_calls),
                tool_calls=norm_calls,
            )
        )
        self.history.append(TurnRecord(role="tool", text=results_json))
        self._prefill_text(
            f"<|im_start|>assistant\n<think></think>{render_tool_calls(norm_calls)}<|im_end|>\n"
            f"<|im_start|>tool\n<tool_response>\n{results_json}\n</tool_response><|im_end|>\n"
        )

    def append_delegated_knowledge_context(
        self,
        query: str,
        answer: str,
        source: str,
    ) -> None:
        filler_ctx = (
            f'You already spoke the holding phrase "{self._pending_filler_text}". '
            if self._pending_filler_text
            else ""
        )
        note = (
            f'[Knowledge Expert ({source}) answer for "{query}": {answer}]\n'
            f"{filler_ctx}"
            "Now speak the clear, accurate answer concisely to the caller in the active language "
            "without repeating the earlier holding phrase."
        )
        self.history.append(TurnRecord(role="user", text=note))
        self._prefill_text(f"<|im_start|>user\n{note}<|im_end|>\n")

    def finalize_greeting_turn(self) -> None:
        if self.history and self.history[0].role == "user":
            pickup = get_language_profile(self.language).call_pickup_phrase
            self.history[0] = TurnRecord(role="user", text=pickup)
        if not self.user_speaking and not self.user_turn_open:
            self._rebuild_cache_from_history()

    def prepare_for_batched_generation(self) -> str:
        return self.latest_user_text()

    def latest_user_text(self) -> str:
        for t in reversed(self.history):
            if t.role == "user" and t.text.strip():
                return t.text.strip()
        return ""

    def latest_assistant_text(self) -> str:
        for t in reversed(self.history):
            if t.role == "assistant" and t.text.strip():
                return t.text.strip()
        return ""

    def _open_user_turn_in_cache(self, onset_blocks_pcm: list[bytes]) -> None:
        with self._model_lock:
            self._pre_user_snapshot = _snapshot_cache(
                self._conv_cache, self._conv_attention_mask
            )
            self._utt_feat_cache = None
            self._utt_enc_cache = None
            self._prefill_text(f"<|im_start|>user\n{self._vision_start_tok}")
            for block in onset_blocks_pcm:
                self._stream_pcm_into_conv_cache(block, is_final=False)
            self.user_turn_open = True

    def _commit_current_user_turn(self, events: list[dict[str, Any]]) -> None:
        tid = self.active_user_turn_id or f"usr-{uuid.uuid4().hex[:8]}"
        final_text = self.active_user_transcript.strip() or "(inaudible)"
        endpoint_ms = (
            max(0.0, (time.perf_counter() - self._last_speech_block_ts) * 1000.0)
            if self._last_speech_block_ts > 0
            else None
        )
        pcm_joined = b"".join(self.active_user_pcm_blocks)

        if self.pending_barge_in_pcm is not None:
            self._open_user_turn_in_cache(self.pending_barge_in_pcm)
            self.pending_barge_in_pcm = None
        if self.user_turn_open:
            self._stream_pcm_into_conv_cache(b"", is_final=True)
            self._prefill_text(f"{self._vision_end_tok}<|im_end|>\n")

        self.history.append(
            TurnRecord(
                role="user",
                text=final_text,
                user_pcm16=pcm_joined if pcm_joined else None,
            )
        )
        self.user_speaking = False
        self.user_turn_open = False
        self.active_user_turn_id = None
        self.active_user_transcript = ""
        self.active_user_pcm_blocks.clear()
        self._utt_feat_cache = None
        self._utt_enc_cache = None
        self._pre_user_snapshot = None
        self.empty_chunk_streak = 0
        self.user_blocks_in_turn = 0
        self._half_stride_probed_blocks = -1
        self.listener.reset_window()

        last_speech_ts = (
            self._last_speech_block_ts if self._last_speech_block_ts > 0.0 else None
        )
        events.append(
            {
                "type": "user_turn_committed",
                "turn_id": tid,
                "text": final_text,
                "endpoint_ms": round(endpoint_ms, 1) if endpoint_ms is not None else None,
                "last_speech_ts": last_speech_ts,
            }
        )

    def ingest_pcm_and_listen(
        self,
        pcm16_bytes: bytes,
        agent_speaking: bool = False,
        recent_agent_text: str = "",
        patience: str = "balanced",
    ) -> list[dict[str, Any]]:
        if not pcm16_bytes:
            return []
        with self._listen_lock:
            self._pcm_buf.extend(pcm16_bytes)
            events: list[dict[str, Any]] = []
            bsize = self.listener.block_bytes

            while len(self._pcm_buf) >= bsize:
                block = bytes(self._pcm_buf[:bsize])
                del self._pcm_buf[:bsize]

                delta_text = self.listener.step_block(block)
                speech = has_real_speech(delta_text)

                if not self.user_speaking:
                    if speech:
                        if agent_speaking and is_likely_agent_echo(
                            delta_text, recent_agent_text, lang_code=self.language
                        ):
                            self._pre_speech_blocks.append(block)
                            self.listener.maybe_reset(is_idle=True)
                            continue

                        turn_id = f"usr-{uuid.uuid4().hex[:8]}"
                        self.user_speaking = True
                        self.active_user_turn_id = turn_id
                        self.active_user_transcript = delta_text.strip()
                        self.empty_chunk_streak = 0
                        self.user_blocks_in_turn = 1
                        self._last_speech_block_ts = time.perf_counter()
                        self._half_stride_probed_blocks = -1

                        onset_blocks = list(self._pre_speech_blocks)[-1:] + [block]
                        self._pre_speech_blocks.clear()
                        self.active_user_pcm_blocks = list(onset_blocks)

                        if agent_speaking:
                            self.pending_barge_in_pcm = list(onset_blocks)
                            events.append(
                                {
                                    "type": "barge_in_detected",
                                    "heard_text": self.active_user_transcript,
                                }
                            )
                        else:
                            self._open_user_turn_in_cache(onset_blocks)
                        events.append(
                            {
                                "type": "user_speech_start",
                                "turn_id": turn_id,
                                "text": self.active_user_transcript,
                            }
                        )
                    else:
                        self._pre_speech_blocks.append(block)
                        self.listener.maybe_reset(is_idle=True)
                else:
                    self.user_blocks_in_turn += 1
                    if speech:
                        self.empty_chunk_streak = 0
                        self._last_speech_block_ts = time.perf_counter()
                        self.active_user_transcript = append_transcript_piece(
                            self.active_user_transcript, delta_text
                        )
                        self.active_user_pcm_blocks.append(block)
                        if self.pending_barge_in_pcm is not None:
                            self.pending_barge_in_pcm.append(block)
                        elif self.user_turn_open:
                            self._stream_pcm_into_conv_cache(block, is_final=False)
                        events.append(
                            {
                                "type": "user_speech_delta",
                                "turn_id": self.active_user_turn_id,
                                "text": self.active_user_transcript,
                            }
                        )
                    else:
                        self.empty_chunk_streak += 1
                        if self.empty_chunk_streak == 1:
                            self.active_user_pcm_blocks.append(block)
                            if self.pending_barge_in_pcm is not None:
                                self.pending_barge_in_pcm.append(block)
                            elif self.user_turn_open:
                                self._stream_pcm_into_conv_cache(block, is_final=False)
                        needed = required_empty_chunks(
                            self.active_user_transcript,
                            patience,
                            lang_code=self.language,
                        )
                        if (
                            self.empty_chunk_streak >= needed
                            or self.user_blocks_in_turn >= self.max_user_blocks
                        ):
                            self._commit_current_user_turn(events)
                            self.listener.maybe_reset(is_idle=True)
                    self.listener.maybe_reset(is_idle=False)

            half_bytes = bsize // 2
            if (
                self.user_speaking
                and not agent_speaking
                and self.empty_chunk_streak == 0
                and (patience or "balanced").lower() in ("snappy", "balanced")
                and len(self._pcm_buf) >= half_bytes
                and half_bytes > 0
                and self._half_stride_probed_blocks != self.user_blocks_in_turn
                and is_semantically_complete(
                    self.active_user_transcript, lang_code=self.language
                )
            ):
                self._half_stride_probed_blocks = self.user_blocks_in_turn
                half_block = bytes(self._pcm_buf[:half_bytes])
                is_silent, feat_after, state_after, num_toks, dur_sec = (
                    self.listener.probe_silence(half_block)
                )
                if is_silent:
                    del self._pcm_buf[:half_bytes]
                    self.listener.adopt_probe_state(
                        feat_after, state_after, num_toks, dur_sec
                    )
                    self.active_user_pcm_blocks.append(half_block)
                    if self.pending_barge_in_pcm is not None:
                        self.pending_barge_in_pcm.append(half_block)
                    elif self.user_turn_open:
                        self._stream_pcm_into_conv_cache(half_block, is_final=False)
                    self._commit_current_user_turn(events)
                    self.listener.maybe_reset(is_idle=True)

            return events

    def flush_ptt_audio(self, pcm16_bytes: bytes) -> list[dict[str, Any]]:
        if not pcm16_bytes:
            return []
        with self._listen_lock:
            self.listener.reset_window()
            # 1) Encode complete PTT clip via causal streaming LiveASR in 960ms blocks.
            bsize = self.listener.block_bytes
            parts: list[str] = []
            for idx in range(0, len(pcm16_bytes), bsize):
                piece = self.listener.step_block(pcm16_bytes[idx : idx + bsize])
                if has_real_speech(piece):
                    parts.append(piece)
            self.listener.reset_window()

            transcript = " ".join(p.strip() for p in parts if p.strip()).strip() or "(spoken message)"
            turn_id = f"usr-{uuid.uuid4().hex[:8]}"

            # 2) Incrementally prefill the PTT audio into persistent _conv_cache.
            self._open_user_turn_in_cache([pcm16_bytes])
            self._stream_pcm_into_conv_cache(b"", is_final=True)
            self._prefill_text(f"{self._vision_end_tok}<|im_end|>\n")
            self.user_turn_open = False
            self._utt_feat_cache = None
            self._utt_enc_cache = None
            self._pre_user_snapshot = None

            self.history.append(
                TurnRecord(role="user", text=transcript, user_pcm16=pcm16_bytes)
            )
            now_ts = time.perf_counter()
            return [
                {"type": "user_speech_start", "turn_id": turn_id, "text": transcript},
                {
                    "type": "user_turn_committed",
                    "turn_id": turn_id,
                    "text": transcript,
                    "last_speech_ts": now_ts,
                },
            ]

    def rollback_interrupted_assistant_turn(
        self,
        played_frames: int,
        interrupted_turn: Optional[TurnRecord],
        replace_last_history: bool,
    ) -> Optional[TurnRecord]:
        with self._model_lock:
            if self._pre_turn_snapshot is None:
                if self.pending_barge_in_pcm is not None:
                    if not self.user_turn_open:
                        self._open_user_turn_in_cache(self.pending_barge_in_pcm)
                    else:
                        for blk in self.pending_barge_in_pcm:
                            self._stream_pcm_into_conv_cache(blk, is_final=False)
                    self.pending_barge_in_pcm = None
                return None

            # Keep intact if caller spoke only in the final trailing ~320 ms.
            if (
                replace_last_history
                and interrupted_turn is not None
                and interrupted_turn.raw_audio_codes is not None
            ):
                codes = np.asarray(interrupted_turn.raw_audio_codes, dtype=np.int32)
                total_non_eos = int(np.sum(~np.any(codes == self._audio_eos_code, axis=-1)))
                if total_non_eos > 0 and int(played_frames) >= max(1, total_non_eos - 4):
                    self._pre_turn_snapshot = None
                    if self.pending_barge_in_pcm is not None:
                        if not self.user_turn_open:
                            self._open_user_turn_in_cache(self.pending_barge_in_pcm)
                        else:
                            for blk in self.pending_barge_in_pcm:
                                self._stream_pcm_into_conv_cache(blk, is_final=False)
                        self.pending_barge_in_pcm = None
                    return None

            self._conv_cache, self._conv_attention_mask = _restore_cache(
                self._pre_turn_snapshot
            )
            self._pre_turn_snapshot = None

            asst_idx: Optional[int] = None
            if replace_last_history and self.history:
                if self.history[-1].role == "assistant":
                    asst_idx = len(self.history) - 1
                elif (
                    len(self.history) >= 2
                    and self.history[-1].role == "user"
                    and self.history[-2].role == "assistant"
                ):
                    asst_idx = len(self.history) - 2

            truncated: Optional[TurnRecord] = None
            if interrupted_turn is not None and int(played_frames) > 0:
                truncated = truncate_assistant_turn_for_barge_in(
                    interrupted_turn,
                    played_frames=int(played_frames),
                    audio_out_token_id=self._audio_out_token_id,
                    audio_eos_code=self._audio_eos_code,
                    im_end_id=self._im_end_id,
                    tokenizer=self.tokenizer,
                )
                if replace_last_history and asst_idx is not None:
                    self.history[asst_idx] = truncated
                elif self.history and self.history[-1].role == "user":
                    self.history.insert(len(self.history) - 1, truncated)
                else:
                    self.history.append(truncated)

                embeds = self._assistant_turn_embeds(truncated)
                if hasattr(self.model, "get_input_embeddings") and _try_torch() is not None:
                    self._prefill_embeds(embeds)
                else:
                    self._prefill_token_ids(embeds)
            elif replace_last_history and asst_idx is not None:
                if asst_idx > 1:
                    self.history.pop(asst_idx)
                elif interrupted_turn is not None:
                    preserved = TurnRecord(role="assistant", text=interrupted_turn.text)
                    self.history[asst_idx] = preserved
                    self._prefill_text(
                        f"<|im_start|>assistant\n<think></think>{preserved.text}<|im_end|>\n"
                    )

            if self.pending_barge_in_pcm is not None:
                self._open_user_turn_in_cache(self.pending_barge_in_pcm)
                self.pending_barge_in_pcm = None

            return truncated

    def maybe_compact_context(self) -> bool:
        if (
            self.conv_tokens < self.settings.compaction_trigger_tokens
            or len(self.history) <= 6
            or self.user_speaking
            or self.user_turn_open
        ):
            return False
        kept_tail = self.history[-6:]
        older = self.history[:-6]
        digest_lines = [f"{t.role}: {t.text}" for t in older if t.text.strip()]
        if not digest_lines:
            return False
        try:
            prompt_str = (
                "<|startoftext|><|im_start|>system\n"
                "Produce a compact factual call memory summary.<|im_end|>\n"
                "<|im_start|>user\n"
                "Summarize the following Karoo Crest Bank telesales call state concisely "
                "(customer name, income, card tier preferences, questions answered, tool results):\n\n"
                + "\n".join(digest_lines[-16:])
                + "<|im_end|>\n<|im_start|>assistant\n<think></think>"
            )
            summary = self._complete_text(prompt_str, max_new_tokens=220)
            if summary:
                self._rolling_summary = summary
        except Exception:
            self._rolling_summary = " | ".join(digest_lines[-6:])
        self.history = list(kept_tail)
        self.compaction_count += 1
        self._rebuild_cache_from_history()
        return True

    def _make_generation_config(
        self,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> Any:
        try:
            from transformers import GenerationConfig

            return GenerationConfig(
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                temperature=float(temperature) if do_sample else None,
                top_p=float(top_p) if do_sample else None,
                repetition_penalty=float(repetition_penalty),
                eos_token_id=[self._im_end_id],
                pad_token_id=_resolve_token_id(self.tokenizer, "<|pad|>", 124893),
            )
        except ImportError:
            from types import SimpleNamespace

            return SimpleNamespace(
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
            )

    def _complete_text(self, prompt_str: str, *, max_new_tokens: int = 64) -> str:
        ids = self._tokenize(prompt_str)
        prompt_len = int(ids.shape[1])
        mask = _ones_mask(prompt_len, device=self.device)
        gen_cfg = self._make_generation_config(
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        with self._model_lock, _no_grad_ctx():
            out = self.model.generate(
                input_ids=ids,
                attention_mask=mask,
                generation_config=gen_cfg,
                generation_mode="text",
                past_key_values=None,
                return_dict_in_generate=True,
            )
        seq = getattr(out, "sequences", out)
        row = seq[0].detach().cpu().tolist() if hasattr(seq[0], "detach") else list(seq[0])
        gen_ids = row[prompt_len:]
        return clean_display_text(self.tokenizer.decode(gen_ids, skip_special_tokens=True))

    def _stream_tts_fallback(
        self,
        text: str,
        persona: PersonaConfig,
        cancel_event: Any,
        on_event: Callable[[dict[str, Any]], None],
        t_start: float,
        poll_barge_in: Optional[Callable[[], None]] = None,
    ) -> tuple[int, float, int, int, list[float]]:
        profile = get_language_profile(self.language)
        system_msg = f"Perform TTS in {profile.tts_accent} using a {persona.voice_gender} voice."
        prompt_str = (
            f"<|startoftext|><|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n<think></think><|audio_start|>"
        )
        ids = self._tokenize(prompt_str)
        mask = _ones_mask(ids.shape[1], device=self.device)
        ttfa_ms = 0.0
        audio_chunks_sent = 0
        first_chunk_samples = 0

        def on_pcm(pcm: bytes) -> None:
            nonlocal ttfa_ms, audio_chunks_sent, first_chunk_samples
            if ttfa_ms == 0.0:
                ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                first_chunk_samples = len(pcm) // 2
            audio_chunks_sent += 1
            on_event({"type": "audio_chunk", "pcm16": pcm, "samples": len(pcm) // 2})

        flusher = LiquidCodecStreamingFlusher(
            self.codec,
            on_pcm,
            first_chunk_frames=self.settings.first_chunk_frames,
            stream_chunk_frames=self.settings.stream_chunk_frames,
            num_codebooks=self._num_codebooks,
            codebook_size=self._codebook_size,
            audio_eos_code=self._audio_eos_code,
            device=self.device,
        )
        slot_ctrl = _SlotCancelController(
            cancel_event,
            poll_barge_in,
            poll_interval=self.settings.coop_listen_slot_interval,
        )
        shared_ids: list[int] = []
        shared_codes: list[np.ndarray] = []
        audio_stream = _SlotAudioStreamer(
            flusher,
            slot_ctrl=slot_ctrl,
            audio_out_token_id=self._audio_out_token_id,
            shared_token_ids=shared_ids,
            shared_codes=shared_codes,
        )
        gen_cfg = self._make_generation_config(
            max_new_tokens=int(self.settings.max_slots),
            do_sample=False,
        )
        try:
            with self._model_lock, _no_grad_ctx():
                self.model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    generation_config=gen_cfg,
                    generation_mode="audio",
                    audio_temperature=float(self.settings.audio_temperature),
                    audio_top_k=int(self.settings.audio_top_k),
                    audio_repetition_penalty=float(self.settings.audio_repetition_penalty),
                    audio_repetition_window=int(self.settings.audio_repetition_window),
                    audio_streamer=audio_stream,
                    past_key_values=None,
                    return_dict_in_generate=True,
                )
        except _GenerationCancelled:
            pass
        if not cancel_event.is_set():
            flusher.finish()
        return (
            flusher.total_frames,
            ttfa_ms,
            audio_chunks_sent,
            first_chunk_samples,
            flusher.chunk_decode_ms,
        )

    def _stream_omni_chat(
        self,
        *,
        persona: PersonaConfig,
        tools: list[ToolSpec] | None,
        force_interleaved: bool,
        cancel_event: Any,
        on_event: Callable[[dict[str, Any]], None],
        poll_mic_fn: Optional[Callable[[], list[dict[str, Any]]]] = None,
        segment: str = "answer",
        max_tokens: Optional[int] = None,
        trailing_user_instruction: str = "",
        mutate_cache: bool = True,
    ) -> TurnOutcome:
        t_start = time.perf_counter()
        use_tools = bool(tools) and not force_interleaved

        with self._model_lock:
            if mutate_cache:
                self._pre_turn_snapshot = _snapshot_cache(
                    self._conv_cache, self._conv_attention_mask
                )
                work_cache = self._conv_cache
                work_mask = self._conv_attention_mask
            else:
                work_cache, work_mask = _snapshot_cache(
                    self._conv_cache, self._conv_attention_mask
                )

            suffix_parts: list[str] = []
            if trailing_user_instruction:
                suffix_parts.append(
                    f"<|im_start|>user\n{trailing_user_instruction}<|im_end|>\n"
                )

            if use_tools:
                gen_mode: Optional[str] = None
                asst_prefix = "<|im_start|>assistant\n<think></think>"
                prefix_marker_id: Optional[int] = None
            else:
                gen_mode = "interleaved"
                asst_prefix = "<|im_start|>assistant\n<think></think><|mixed_start|>"
                prefix_marker_id = self._mixed_start_id

            prompt_suffix = "".join(suffix_parts) + asst_prefix
            input_ids = self._tokenize(prompt_suffix)
            past_len = _cache_seq_len(work_cache)
            base_mask = (
                work_mask
                if work_mask is not None and work_mask.shape[1] == past_len
                else _ones_mask(past_len, device=self.device)
            )
            extended_mask = _cat_dim1(
                base_mask, _ones_mask(input_ids.shape[1], device=self.device)
            )

            raw_text_acc = ""
            clean_text_acc = ""
            ttft_ms = 0.0
            ttfa_ms = 0.0
            chunks_sent = 0
            first_chunk_samples = 0
            text_streaming_allowed = not use_tools
            tool_mode_detected = False

            shared_token_ids: list[int] = (
                [prefix_marker_id] if prefix_marker_id is not None else []
            )
            shared_codes: list[np.ndarray] = []

            def emit_clean_diff(new_clean: str) -> None:
                nonlocal clean_text_acc
                if new_clean == clean_text_acc:
                    return
                diff = (
                    new_clean[len(clean_text_acc) :]
                    if new_clean.startswith(clean_text_acc)
                    else new_clean
                )
                clean_text_acc = new_clean
                if diff:
                    on_event(
                        {
                            "type": "text_delta",
                            "delta": diff,
                            "text": clean_text_acc,
                            "segment": segment,
                        }
                    )

            def on_text_delta(piece: str) -> None:
                nonlocal raw_text_acc, ttft_ms, text_streaming_allowed, tool_mode_detected
                if not piece:
                    return
                raw_text_acc += piece
                if ttft_ms == 0.0:
                    cleaned_so_far = clean_display_text(raw_text_acc)
                    if cleaned_so_far:
                        ttft_ms = (time.perf_counter() - t_start) * 1000.0
                if "<think>" in raw_text_acc and "</think>" not in raw_text_acc:
                    return

                new_clean = clean_display_text(raw_text_acc)
                stripped = new_clean.lstrip()
                if (
                    stripped.startswith("<tool_call")
                    or stripped.startswith("<function")
                    or "<tool_call>" in new_clean
                ):
                    tool_mode_detected = True
                    text_streaming_allowed = False
                    return

                if not text_streaming_allowed and not tool_mode_detected:
                    if (
                        "<|mixed_start|>" in raw_text_acc
                        or "<|audio_start|>" in raw_text_acc
                        or chunks_sent > 0
                        or (len(stripped) >= 16 and "<" not in stripped)
                    ):
                        text_streaming_allowed = True

                if text_streaming_allowed and new_clean:
                    emit_clean_diff(new_clean)

            def on_audio_pcm(pcm: bytes) -> None:
                nonlocal chunks_sent, first_chunk_samples, ttfa_ms, text_streaming_allowed
                if not pcm:
                    return
                if ttfa_ms == 0.0:
                    ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                    first_chunk_samples = len(pcm) // 2
                if not tool_mode_detected and not text_streaming_allowed:
                    new_clean = clean_display_text(raw_text_acc)
                    if new_clean and not new_clean.lstrip().startswith("<"):
                        text_streaming_allowed = True
                        emit_clean_diff(new_clean)
                chunks_sent += 1
                on_event(
                    {
                        "type": "audio_chunk",
                        "pcm16": pcm,
                        "samples": len(pcm) // 2,
                    }
                )

            def poll_barge_in() -> None:
                if poll_mic_fn is None:
                    return
                for ev in poll_mic_fn():
                    if ev.get("type") == "barge_in_detected":
                        cancel_event.set()
                        return

            flusher = LiquidCodecStreamingFlusher(
                self.codec,
                on_audio_pcm,
                first_chunk_frames=self.settings.first_chunk_frames,
                stream_chunk_frames=self.settings.stream_chunk_frames,
                num_codebooks=self._num_codebooks,
                codebook_size=self._codebook_size,
                audio_eos_code=self._audio_eos_code,
                device=self.device,
            )
            slot_ctrl = _SlotCancelController(
                cancel_event,
                poll_barge_in if poll_mic_fn is not None else None,
                poll_interval=self.settings.coop_listen_slot_interval,
            )
            text_streamer = _SlotTextStreamer(
                self.tokenizer,
                slot_ctrl=slot_ctrl,
                shared_token_ids=shared_token_ids,
                on_text_delta=on_text_delta,
            )
            audio_streamer = _SlotAudioStreamer(
                flusher,
                slot_ctrl=slot_ctrl,
                audio_out_token_id=self._audio_out_token_id,
                shared_token_ids=shared_token_ids,
                shared_codes=shared_codes,
            )

            temp = float(self.settings.text_temperature)
            gen_cfg = self._make_generation_config(
                max_new_tokens=int(max_tokens or self.settings.max_slots),
                do_sample=temp > 0.0,
                temperature=temp,
                top_p=float(self.settings.text_top_p),
                repetition_penalty=float(self.settings.text_repetition_penalty),
            )

            cancelled = False
            try:
                with _no_grad_ctx():
                    gen_out = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=extended_mask,
                        generation_config=gen_cfg,
                        generation_mode=gen_mode,
                        audio_temperature=float(self.settings.audio_temperature),
                        audio_top_k=int(self.settings.audio_top_k),
                        audio_repetition_penalty=float(self.settings.audio_repetition_penalty),
                        audio_repetition_window=int(self.settings.audio_repetition_window),
                        streamer=text_streamer,
                        audio_streamer=audio_streamer,
                        past_key_values=work_cache,
                        return_dict_in_generate=True,
                    )
            except _GenerationCancelled:
                cancelled = True
                gen_out = None

            if cancelled or cancel_event.is_set():
                if mutate_cache and self._pre_turn_snapshot is not None:
                    self._conv_cache, self._conv_attention_mask = _restore_cache(
                        self._pre_turn_snapshot
                    )
                partial_codes = (
                    np.stack(shared_codes, axis=0).astype(np.int32)
                    if shared_codes
                    else None
                )
                partial_rec = TurnRecord(
                    role="assistant",
                    text=clean_text_acc,
                    raw_token_ids=shared_token_ids or None,
                    raw_audio_codes=partial_codes,
                )
                return TurnOutcome(
                    kind="cancelled",
                    turn_record=partial_rec,
                    text=clean_text_acc,
                    audio_frames=flusher.total_frames,
                    ttfa_ms=ttfa_ms,
                    ttft_ms=ttft_ms,
                    audio_chunks_sent=chunks_sent,
                    first_chunk_samples=first_chunk_samples,
                    gen_wall_ms=(time.perf_counter() - t_start) * 1000.0,
                    cancelled=True,
                )

            flusher.finish()
            parsed_calls = parse_tool_calls(raw_text_acc) if use_tools else []
            if parsed_calls:
                if mutate_cache and self._pre_turn_snapshot is not None:
                    # Roll back to pre-turn; caller calls `append_tool_exchange` next.
                    self._conv_cache, self._conv_attention_mask = _restore_cache(
                        self._pre_turn_snapshot
                    )
                    self._pre_turn_snapshot = None
                return TurnOutcome(
                    kind="tool_calls",
                    text="",
                    tool_calls=parsed_calls,
                    ttft_ms=ttft_ms,
                    gen_wall_ms=(time.perf_counter() - t_start) * 1000.0,
                )

            final_text = clean_display_text(raw_text_acc) or clean_text_acc
            if final_text and final_text != clean_text_acc:
                emit_clean_diff(final_text)

            used_tts_fallback = False
            if final_text and chunks_sent == 0 and not cancel_event.is_set():
                used_tts_fallback = True
                fb_frames, fb_ttfa, fb_chunks, fb_first, fb_decode_ms = (
                    self._stream_tts_fallback(
                        final_text,
                        persona,
                        cancel_event,
                        on_event,
                        t_start,
                        poll_barge_in=poll_barge_in if poll_mic_fn is not None else None,
                    )
                )
                chunks_sent += fb_chunks
                flusher.chunk_decode_ms.extend(fb_decode_ms)
                total_frames = fb_frames
                if ttfa_ms == 0.0:
                    ttfa_ms = fb_ttfa
                    first_chunk_samples = fb_first
                if cancel_event.is_set():
                    if mutate_cache and self._pre_turn_snapshot is not None:
                        self._conv_cache, self._conv_attention_mask = _restore_cache(
                            self._pre_turn_snapshot
                        )
                    partial_rec = TurnRecord(
                        role="assistant",
                        text=final_text,
                        raw_audio_codes=[None] * total_frames if total_frames else None,
                    )
                    return TurnOutcome(
                        kind="cancelled",
                        turn_record=partial_rec,
                        text=final_text,
                        audio_frames=total_frames,
                        ttfa_ms=ttfa_ms,
                        ttft_ms=ttft_ms,
                        audio_chunks_sent=chunks_sent,
                        first_chunk_samples=first_chunk_samples,
                        gen_wall_ms=(time.perf_counter() - t_start) * 1000.0,
                        cancelled=True,
                        used_tts_fallback=True,
                    )
            else:
                total_frames = flusher.total_frames

            if mutate_cache:
                self._conv_cache = work_cache
                self._conv_attention_mask = _ones_mask(
                    _cache_seq_len(self._conv_cache), device=self.device
                )
                ended_with_im_end = bool(
                    shared_token_ids and shared_token_ids[-1] == self._im_end_id
                )
                if gen_out is not None and getattr(gen_out, "sequences", None) is not None:
                    seq_tail = _item_int(gen_out.sequences[0, -1])
                    ended_with_im_end = seq_tail == self._im_end_id
                self._prefill_text("\n" if ended_with_im_end else "<|im_end|>\n")

            wall_ms = (time.perf_counter() - t_start) * 1000.0
            codes_arr = (
                np.stack(shared_codes, axis=0).astype(np.int32)
                if shared_codes
                else ([None] * total_frames if total_frames else None)
            )
            turn_rec = TurnRecord(
                role="assistant",
                text=final_text,
                raw_token_ids=shared_token_ids or None,
                raw_audio_codes=codes_arr,
            )
            codec_avg = (
                round(sum(flusher.chunk_decode_ms) / len(flusher.chunk_decode_ms), 2)
                if flusher.chunk_decode_ms
                else 0.0
            )
            return TurnOutcome(
                kind="response",
                turn_record=turn_rec,
                text=final_text,
                audio_frames=total_frames,
                ttfa_ms=ttfa_ms,
                ttft_ms=ttft_ms or ttfa_ms,
                slots_per_sec=round(
                    max(len(shared_token_ids), total_frames) / max(wall_ms / 1000.0, 1e-3),
                    1,
                ),
                audio_chunks_sent=chunks_sent,
                first_chunk_samples=first_chunk_samples,
                codec_chunk_avg_ms=codec_avg,
                gen_wall_ms=wall_ms,
                used_tts_fallback=used_tts_fallback,
            )

    def generate_assistant_turn(
        self,
        persona: PersonaConfig,
        tools: list[ToolSpec] | None,
        force_interleaved: bool,
        cancel_event: Any,
        on_event: Callable[[dict[str, Any]], None],
        poll_mic_fn: Optional[Callable[[], list[dict[str, Any]]]] = None,
        segment: str = "answer",
    ) -> TurnOutcome:
        outcome = self._stream_omni_chat(
            persona=persona,
            tools=tools,
            force_interleaved=force_interleaved,
            cancel_event=cancel_event,
            on_event=on_event,
            poll_mic_fn=poll_mic_fn,
            segment=segment,
            mutate_cache=True,
        )
        self._pending_filler_text = ""
        if (
            outcome.kind == "response"
            and not outcome.cancelled
            and outcome.turn_record is not None
        ):
            self.history.append(outcome.turn_record)
        return outcome

    def _run_router_call(self, user_text: str, *, has_tools: bool = True) -> tuple[str, str, float]:
        t0 = time.perf_counter()
        profile = get_language_profile(self.language)
        prev_asst = self.latest_assistant_text()
        ctx_line = f"Previous advisor turn: {prev_asst}\n" if prev_asst else ""
        tool_rule = (
            "- Output `TOOL` ONLY when the customer explicitly gives concrete income, loan amount, term, "
            "or application/callback details requiring CRM pre-qualification, repayment calculation, application submission, or callback scheduling.\n"
            "- Output `DIRECT` for greetings, yes/no, short clarifications, objections, complaints/meta-questions "
            "(\"Why are you so slow?\", \"Why are you repeating?\"), or general conversation.\n"
            if has_tools
            else "- Output `DIRECT` for greetings, yes/no, short clarifications, objections, complaints/meta-questions, and general conversation.\n"
        )
        examples = (
            "Examples:\n"
            "- \"No, I can't tell you.\" -> DIRECT\n"
            "- \"Why are you repeating yourself?\" -> DIRECT\n"
            + ("- \"I earn R35,000 full-time with SA ID.\" -> TOOL\n" if has_tools else "")
            + "- \"What are the Protea Platinum lounge fees?\" -> DELEGATE: What are the Ubuntu Protea Platinum airport lounge fees?"
        )
        sys_prompt = (
            "Output strictly `DIRECT`, `TOOL`, or `DELEGATE: <question>`."
            if has_tools
            else "Output strictly `DIRECT` or `DELEGATE: <question>`."
        )
        prompt_str = (
            f"<|startoftext|><|im_start|>system\n{sys_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{ctx_line}"
            f"Customer just said: {user_text or '(audio turn)'}\n\n"
            "Classify into exactly ONE line:\n"
            f"{tool_rule}"
            f"- Output `DELEGATE: <concise standalone question in {profile.router_language_name}>` "
            "ONLY when the customer asks a specific product policy, tariff, fee, FX, grace period, or NCA/FICA/POPIA question.\n\n"
            f"{examples}"
            "<|im_end|>\n<|im_start|>assistant\n<think></think>"
        )
        raw = self._complete_text(
            prompt_str, max_new_tokens=int(self.settings.max_router_tokens)
        )
        dec, query = parse_router_query(raw, user_text)
        if dec == "tool" and not has_tools:
            dec = "direct"
        return dec, query, (time.perf_counter() - t0) * 1000.0

    def _run_query_extract_call(self, user_text: str) -> str:
        profile = get_language_profile(self.language)
        prev_asst = self.latest_assistant_text()
        user_body = (
            (f"Previous advisor turn: {prev_asst}\n" if prev_asst else "")
            + f"Customer message: {user_text or '(audio turn)'}\n"
            + f"Rewrite this as one crisp standalone banking knowledge question in {profile.router_language_name}."
        )
        prompt_str = (
            "<|startoftext|><|im_start|>system\n"
            "Output ONLY one concise standalone question.<|im_end|>\n"
            f"<|im_start|>user\n{user_body}<|im_end|>\n"
            "<|im_start|>assistant\n<think></think>"
        )
        raw = self._complete_text(
            prompt_str, max_new_tokens=int(self.settings.max_router_tokens)
        )
        return raw.strip() or user_text.strip() or "(banking question)"

    def run_filler_and_query_batch(
        self,
        cancel_event: Any,
        on_event: Callable[[dict[str, Any]], None],
        poll_mic_fn: Optional[Callable[[], list[dict[str, Any]]]] = None,
    ) -> BatchPlanOutcome:
        profile = get_language_profile(self.language)
        user_text = self.latest_user_text()
        persona_stub = PersonaConfig(
            id="filler",
            name="Agent",
            role="Advisor",
            voice_gender="female",
            voice_accent=profile.tts_accent,
            greeting_trigger="",
            system_prompt="",
        )
        query = self._run_query_extract_call(user_text)
        if cancel_event.is_set():
            return BatchPlanOutcome(decision="cancelled", delegation_query=query)
        if query:
            on_event({"type": "delegation_query_ready", "query": query})

        filler_out = self._stream_omni_chat(
            persona=persona_stub,
            tools=None,
            force_interleaved=True,
            cancel_event=cancel_event,
            on_event=on_event,
            poll_mic_fn=poll_mic_fn,
            segment="filler",
            max_tokens=min(int(self.settings.max_filler_slots), 120),
            trailing_user_instruction=profile.filler_instruction,
            mutate_cache=False,
        )
        if filler_out.cancelled or cancel_event.is_set():
            return BatchPlanOutcome(
                decision="cancelled",
                delegation_query=query,
                filler_text=filler_out.text,
                filler_turn=filler_out.turn_record,
                filler_audio_frames=filler_out.audio_frames,
            )
        self._pending_filler_text = filler_out.text
        return BatchPlanOutcome(
            decision="delegate",
            delegation_query=query,
            filler_text=filler_out.text,
            filler_turn=filler_out.turn_record,
            filler_audio_frames=filler_out.audio_frames,
            ttft_ms=filler_out.ttft_ms,
            ttfa_filler_ms=filler_out.ttfa_ms,
            audio_chunks_sent=filler_out.audio_chunks_sent,
            first_chunk_samples=filler_out.first_chunk_samples,
            slots_per_sec=filler_out.slots_per_sec,
        )

    def run_speculative_selective_batch(
        self,
        persona: PersonaConfig,
        tools: list[ToolSpec] | None,
        dynamic_filler_enabled: bool,
        cancel_event: Any,
        on_event: Callable[[dict[str, Any]], None],
        poll_mic_fn: Optional[Callable[[], list[dict[str, Any]]]] = None,
    ) -> BatchPlanOutcome:
        user_text = self.latest_user_text()
        profile = get_language_profile(self.language)
        dec, query, route_ms = self._run_router_call(user_text, has_tools=bool(tools))
        if cancel_event.is_set():
            return BatchPlanOutcome(decision="cancelled", route_decision_ms=route_ms)

        if dec == "direct":
            out = self._stream_omni_chat(
                persona=persona,
                tools=None,
                force_interleaved=True,
                cancel_event=cancel_event,
                on_event=on_event,
                poll_mic_fn=poll_mic_fn,
                segment="answer",
                mutate_cache=True,
            )
            if out.cancelled or cancel_event.is_set():
                return BatchPlanOutcome(
                    decision="cancelled",
                    filler_turn=out.turn_record,
                    route_decision_ms=route_ms,
                )
            self.history.append(
                out.turn_record or TurnRecord(role="assistant", text=out.text)
            )
            return BatchPlanOutcome(
                decision="direct",
                direct_outcome=out,
                route_decision_ms=route_ms,
                ttft_ms=out.ttft_ms,
                ttfa_direct_ms=out.ttfa_ms,
                audio_chunks_sent=out.audio_chunks_sent,
                first_chunk_samples=out.first_chunk_samples,
                slots_per_sec=out.slots_per_sec,
            )

        if dec == "tool" and tools:
            tool_out = self._stream_omni_chat(
                persona=persona,
                tools=tools,
                force_interleaved=False,
                cancel_event=cancel_event,
                on_event=on_event,
                poll_mic_fn=poll_mic_fn,
                segment="answer",
                mutate_cache=True,
            )
            if tool_out.cancelled or cancel_event.is_set():
                return BatchPlanOutcome(
                    decision="cancelled",
                    filler_turn=tool_out.turn_record,
                    route_decision_ms=route_ms,
                )
            if tool_out.kind == "tool_calls":
                return BatchPlanOutcome(
                    decision="tool_calls",
                    direct_outcome=tool_out,
                    route_decision_ms=route_ms,
                )
            self.history.append(
                tool_out.turn_record or TurnRecord(role="assistant", text=tool_out.text)
            )
            return BatchPlanOutcome(
                decision="direct",
                direct_outcome=tool_out,
                route_decision_ms=route_ms,
                ttft_ms=tool_out.ttft_ms,
                ttfa_direct_ms=tool_out.ttfa_ms,
                audio_chunks_sent=tool_out.audio_chunks_sent,
                first_chunk_samples=tool_out.first_chunk_samples,
                slots_per_sec=tool_out.slots_per_sec,
            )

        if query:
            on_event({"type": "delegation_query_ready", "query": query})

        filler_out: Optional[TurnOutcome] = None
        if dynamic_filler_enabled and not cancel_event.is_set():
            filler_out = self._stream_omni_chat(
                persona=persona,
                tools=None,
                force_interleaved=True,
                cancel_event=cancel_event,
                on_event=on_event,
                poll_mic_fn=poll_mic_fn,
                segment="filler",
                max_tokens=min(int(self.settings.max_filler_slots), 120),
                trailing_user_instruction=profile.filler_instruction,
                mutate_cache=False,
            )
        if cancel_event.is_set() or (filler_out is not None and filler_out.cancelled):
            return BatchPlanOutcome(
                decision="cancelled",
                delegation_query=query,
                filler_text=filler_out.text if filler_out else "",
                filler_turn=filler_out.turn_record if filler_out else None,
                filler_audio_frames=filler_out.audio_frames if filler_out else 0,
                route_decision_ms=route_ms,
            )
        if filler_out is not None:
            self._pending_filler_text = filler_out.text
        return BatchPlanOutcome(
            decision="delegate",
            delegation_query=query,
            filler_text=filler_out.text if filler_out else "",
            filler_turn=filler_out.turn_record if filler_out else None,
            filler_audio_frames=filler_out.audio_frames if filler_out else 0,
            route_decision_ms=route_ms,
            ttft_ms=filler_out.ttft_ms if filler_out else 0.0,
            ttfa_filler_ms=filler_out.ttfa_ms if filler_out else 0.0,
            audio_chunks_sent=filler_out.audio_chunks_sent if filler_out else 0,
            first_chunk_samples=filler_out.first_chunk_samples if filler_out else 0,
            slots_per_sec=filler_out.slots_per_sec if filler_out else 0.0,
        )


# ---------------------------------------------------------------------------
# Engine Lifecycle Wrapper
# ---------------------------------------------------------------------------


class TransformersOmniEngine:
    """In-process PyTorch/Transformers CUDA engine for `LiquidAI/LFM3-Omni-2B`."""

    def __init__(
        self,
        settings: EngineSettings,
        *,
        model: Any = None,
        processor: Any = None,
        codec: Any = None,
        device: Any = None,
    ) -> None:
        self.settings = settings
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", None) if processor else None
        self.feature_extractor = (
            getattr(processor, "feature_extractor", None) if processor else None
        )
        self.codec = codec
        self.device = device
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tf-lfm3"
        )
        self._model_lock = threading.RLock()
        self._load_lock = threading.Lock()
        self._loaded = bool(model is not None and processor is not None)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _resolve_model_dir(self) -> str:
        env_dir = os.environ.get("VLLM_LOCAL_MODEL_DIR", "").strip()
        if env_dir and Path(env_dir).exists():
            return env_dir
        if Path(self.settings.model_id).exists():
            return self.settings.model_id
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=self.settings.model_id,
            revision=self.settings.model_revision or None,
        )

    def load(self) -> None:
        """Load LFM3-Omni-2B, processor, and LiquidCodec onto GPU cleanly."""
        with self._load_lock:
            if self._loaded:
                return
            t0 = time.perf_counter()
            if self.model is not None and self.processor is not None:
                self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
                self.feature_extractor = getattr(
                    self.processor, "feature_extractor", None
                )
                self.codec = self.codec or getattr(self.processor, "audio_tokenizer", None)
                self._loaded = True
                return

            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            model_dir = self._resolve_model_dir()
            device_str = _select_torch_device(self.settings.transformers_device)
            dtype = torch.bfloat16 if device_str.startswith("cuda") else torch.float32
            attn_impl = (
                self.settings.transformers_attn_implementation or "sdpa"
            ).strip()

            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
            model = AutoModelForMultimodalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
            )
            model = model.to(device_str).eval()

            codec = getattr(processor, "audio_tokenizer", None)
            if codec is None:
                mod = sys.modules.get(".".join(type(model).__module__.split(".")[:-1]))
                codec_mod_name = f"{type(model).__module__.rsplit('.', 1)[0]}.modeling_liquid_codec"
                codec_mod = sys.modules.get(codec_mod_name)
                if codec_mod is not None and hasattr(codec_mod, "LiquidCodecModel"):
                    codec = codec_mod.LiquidCodecModel.from_pretrained(
                        model_dir, subfolder="audio_tokenizer"
                    )
            if codec is not None:
                codec = codec.to(device_str).eval()

            self.device = device_str
            self.processor = processor
            self.tokenizer = getattr(processor, "tokenizer", processor)
            self.feature_extractor = getattr(processor, "feature_extractor", None)
            self.codec = codec
            self.model = model
            self._warmup_unlocked()
            self._loaded = True
            elapsed = (time.perf_counter() - t0) * 1000.0
            print(
                f"  [transformers] Loaded {self.settings.model_id} on {device_str} "
                f"({dtype}, attn={attn_impl}) in {elapsed:.0f} ms",
                flush=True,
            )

    def _warmup_unlocked(self) -> None:
        torch = _try_torch()
        if torch is None or self.model is None or self.tokenizer is None:
            return
        with torch.no_grad():
            dummy_ids = torch.tensor(
                [_encode_ids(self.tokenizer, "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>")],
                dtype=torch.long,
                device=self.device,
            )
            self.model.generate(
                input_ids=dummy_ids,
                max_new_tokens=2,
                do_sample=False,
                generation_mode="text",
                return_dict_in_generate=True,
            )
            if self.feature_extractor is not None:
                feat_out, _ = self.feature_extractor.stream(
                    np.zeros((15360,), dtype=np.float32),
                    cache=None,
                    is_final=False,
                    sampling_rate=16000,
                )
                mel = feat_out["input_features"].to(self.device)
                live_ids = torch.tensor(
                    [_encode_ids(self.tokenizer, TransformersLiveS2SListener.LIVE_ASR_PROMPT)],
                    dtype=torch.long,
                    device=self.device,
                )
                self.model.transcribe_live(
                    mel, state=None, input_ids=live_ids, is_final=False, max_new_tokens=4
                )
            if self.codec is not None:
                cfg = getattr(self.model, "config", None)
                nq = int(getattr(cfg, "num_codebooks", 12))
                c_cache = _make_streaming_codec_cache(self.codec)
                self.codec.decode(
                    torch.zeros((1, nq, 1), dtype=torch.long, device=self.device),
                    streaming_cache=c_cache,
                    finalize=True,
                    return_dict=True,
                )

    def run_on_worker(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Future[Any]:
        return self._executor.submit(fn, *args, **kwargs)

    def create_pipeline(self) -> TransformersStreamingConversationSession:
        if not self._loaded:
            self.load()
        return TransformersStreamingConversationSession(
            model=self.model,
            processor=self.processor,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            codec=self.codec,
            settings=self.settings,
            device=self.device,
            model_lock=self._model_lock,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
        with self._load_lock:
            self.model = None
            self.codec = None
            self.processor = None
            self._loaded = False
        torch = _try_torch()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
