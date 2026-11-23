"""vLLM-Omni (Linux/CUDA) pure-S2S inference backend via native in-process `AsyncOmni`."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
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
    has_real_speech,
    is_likely_agent_echo,
    parse_tool_calls,
    render_tool_calls,
    required_empty_chunks,
)

# ---------------------------------------------------------------------------
# Pure-NumPy audio & prompt helpers (zero MLX / vLLM top-level imports)
# ---------------------------------------------------------------------------


@dataclass
class _SimpleSamplingParams:
    """Fallback stand-in for `vllm.SamplingParams` when `vllm` is not installed."""

    temperature: float = 0.0
    max_tokens: int = 900
    top_p: float = 1.0
    output_kind: Any = "DELTA"
    extra_args: dict[str, Any] = field(default_factory=dict)


def _pcm16_to_float32(pcm16_bytes: bytes) -> np.ndarray:
    """Decode little-endian 16-bit mono PCM bytes to float32 in [-1.0, 1.0]."""
    if not pcm16_bytes:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(pcm16_bytes, dtype="<i2").astype(np.float32) / 32768.0


def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Encode float32 waveform in [-1.0, 1.0] to little-endian 16-bit PCM bytes."""
    if audio.size == 0:
        return b""
    clipped = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _to_float32_1d(arr_or_tensor: Any) -> np.ndarray:
    """Convert a torch Tensor or array-like chunk into a 1D float32 numpy array."""
    if arr_or_tensor is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(arr_or_tensor, "detach"):
        arr = arr_or_tensor.detach().cpu().float().numpy()
    else:
        arr = np.asarray(arr_or_tensor, dtype=np.float32)
    return arr.reshape(-1)


def _audio_output_to_pcm16(audio_obj: Any) -> bytes:
    """Normalize Stage 1 `multimodal_output['audio']` payload into 24 kHz PCM16 bytes."""
    if audio_obj is None:
        return b""
    if isinstance(audio_obj, (list, tuple)):
        parts = [_to_float32_1d(item) for item in audio_obj if item is not None]
        parts = [p for p in parts if p.size > 0]
        if not parts:
            return b""
        combined = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
    else:
        combined = _to_float32_1d(audio_obj)
    if combined.size == 0:
        return b""
    return _float32_to_pcm16(combined)


def _extract_mm_audio_pcm16(output: Any) -> bytes:
    """Extract 24 kHz PCM16 bytes from Stage 1 `OmniRequestOutput` (supports `Mapping` payloads)."""
    mm = getattr(output, "multimodal_output", None)
    if not mm:
        for completion_output in getattr(output, "outputs", None) or ():
            completion_mm = getattr(completion_output, "multimodal_output", None)
            if completion_mm:
                mm = completion_mm
                break
    if not isinstance(mm, Mapping):
        return b""
    if "audio" in mm:
        audio_obj = mm["audio"]
    elif "model_outputs" in mm:
        audio_obj = mm["model_outputs"]
    else:
        return b""
    return _audio_output_to_pcm16(audio_obj)


def _normalize_tool_calls_for_template(
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure tool call `arguments` are Python dicts required by LFM3 chat template."""
    normalized: list[dict[str, Any]] = []
    for call in calls:
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                parsed = {}
            args_dict = parsed if isinstance(parsed, dict) else {}
        elif isinstance(raw_args, dict):
            args_dict = dict(raw_args)
        else:
            args_dict = {}
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": str(fn.get("name") or ""),
                    "arguments": args_dict,
                },
            }
        )
    return normalized


def _parse_router_query(raw_router_text: str, fallback_text: str) -> tuple[str, str]:
    """Parse `DIRECT` or `DELEGATE: <question>` into `(kind, standalone_question)`."""
    cleaned = clean_display_text(raw_router_text).strip()
    upper = cleaned.upper()
    if "DELEGATE" in upper:
        idx = upper.find("DELEGATE")
        tail = cleaned[idx + len("DELEGATE") :].lstrip(" :-\n\t")
        return "delegate", (tail.strip() or fallback_text.strip() or "(banking question)")
    return "direct", ""


def _norm_word(tok: str) -> str:
    return tok.strip(".,!?;:\"'`()[]{}¿¡").lower()


def _dedup_overlap_delta(prev_text: str, new_text: str) -> str:
    """Strip overlapping prefix words from consecutive 1.28 s sliding ASR windows."""
    prev_clean = (prev_text or "").strip()
    new_clean = (new_text or "").strip()
    if not new_clean:
        return ""
    if not prev_clean:
        return new_clean

    prev_words = [_norm_word(w) for w in prev_clean.split() if _norm_word(w)]
    orig_new_words = new_clean.split()
    norm_new_words = [_norm_word(w) for w in orig_new_words]
    if not prev_words or not norm_new_words:
        return new_clean

    max_k = min(len(prev_words), len(norm_new_words), 10)
    overlap_k = 0
    for k in range(max_k, 0, -1):
        if prev_words[-k:] == norm_new_words[:k]:
            overlap_k = k
            break

    if overlap_k >= len(orig_new_words):
        return ""
    return " ".join(orig_new_words[overlap_k:]).strip()


# ---------------------------------------------------------------------------
# Dedicated asyncio bridge for native in-process vllm_omni.AsyncOmni
# ---------------------------------------------------------------------------


class _NativeOmniRunner:
    """Owns the background asyncio event loop driving `vllm_omni.AsyncOmni`."""

    def __init__(
        self,
        omni: Any,
        tokenizer: Any,
        *,
        timeout_sec: float = 90.0,
    ) -> None:
        self.omni = omni
        self.tokenizer = tokenizer
        self.num_stages: int = max(1, int(getattr(omni, "num_stages", 2)))
        self.timeout_sec: float = float(timeout_sec)

        self._sampling_params_cls: Any = None
        self._delta_kind: Any = "DELTA"
        self._final_kind: Any = "FINAL_ONLY"
        self._resolve_sampling_types()

        self._loop_ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="vllm-omni-loop",
            daemon=True,
        )
        self._thread.start()
        self._loop_ready.wait(timeout=5.0)

    def _resolve_sampling_types(self) -> None:
        try:
            from vllm import SamplingParams
            from vllm.sampling_params import RequestOutputKind

            self._sampling_params_cls = SamplingParams
            self._delta_kind = RequestOutputKind.DELTA
            self._final_kind = RequestOutputKind.FINAL_ONLY
        except ImportError:
            self._sampling_params_cls = _SimpleSamplingParams
            self._delta_kind = "DELTA"
            self._final_kind = "FINAL_ONLY"

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                try:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                except Exception:
                    pass
            self._loop.close()

    def build_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        audio_clips_16k: Optional[list[np.ndarray]] = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        append_suffix: str = "",
    ) -> dict[str, Any]:
        """Render LFM3 chat messages + optional tools/audio into a native `TokensPrompt` dict."""
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": bool(add_generation_prompt),
            "continue_final_message": bool(continue_final_message),
        }
        if tools:
            template_kwargs["tools"] = tools

        rendered = str(self.tokenizer.apply_chat_template(messages, **template_kwargs))
        if append_suffix:
            rendered += append_suffix

        # Critical: LFM3 chat_template already emits leading bos_token.
        prompt_ids = list(self.tokenizer.encode(rendered, add_special_tokens=False))
        prompt: dict[str, Any] = {"prompt_token_ids": prompt_ids}
        if audio_clips_16k:
            clips = [
                (np.asarray(clip, dtype=np.float32).reshape(-1), 16000)
                for clip in audio_clips_16k
            ]
            prompt["multi_modal_data"] = {
                "audio": clips[0] if len(clips) == 1 else clips
            }
        return prompt

    def build_live_asr_prompt(self, wav_f32: np.ndarray) -> dict[str, Any]:
        """Build native LFM3 assistant-turn LiveASR prompt (`<|audio|>` without `<think>` or vision spans)."""
        rendered = (
            "<|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n"
            "<|im_start|>assistant\n<|audio|>"
        )
        prompt_ids = list(self.tokenizer.encode(rendered, add_special_tokens=False))
        return {
            "prompt_token_ids": prompt_ids,
            "multi_modal_data": {
                "audio": (np.asarray(wav_f32, dtype=np.float32).reshape(-1), 16000)
            },
            "mm_processor_kwargs": {"audio_markers": False},
        }

    def build_sampling_params_list(
        self,
        *,
        temperature: float,
        max_tokens: int,
        extra_args: Optional[dict[str, Any]] = None,
        top_p: Optional[float] = None,
        streaming: bool = False,
    ) -> list[Any]:
        """Construct per-stage `SamplingParams` matching `omni.num_stages`."""
        out_kind = self._delta_kind if streaming else self._final_kind
        stage0_kwargs: dict[str, Any] = {
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "output_kind": out_kind,
            "extra_args": dict(extra_args or {}),
        }
        if top_p is not None and float(temperature) > 0.0:
            stage0_kwargs["top_p"] = float(top_p)

        params_list = [self._sampling_params_cls(**stage0_kwargs)]
        for _ in range(1, self.num_stages):
            params_list.append(
                self._sampling_params_cls(
                    temperature=0.0,
                    max_tokens=1,
                    output_kind=out_kind,
                )
            )
        return params_list

    def complete_text(
        self,
        prompt: dict[str, Any],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        extra_args: Optional[dict[str, Any]] = None,
    ) -> str:
        """Run a non-streaming text-only request on `AsyncOmni`."""
        req_id = f"txt-{uuid.uuid4().hex[:12]}"
        params = self.build_sampling_params_list(
            temperature=temperature,
            max_tokens=max_tokens,
            extra_args=extra_args if extra_args is not None else {"generation_mode": "text"},
            streaming=False,
        )

        async def _run() -> str:
            latest_text = ""
            async for output in self.omni.generate(
                prompt=prompt,
                request_id=req_id,
                sampling_params_list=params,
                output_modalities=["text"],
            ):
                if getattr(output, "error", None):
                    raise RuntimeError(str(output.error))
                out_type = getattr(output, "final_output_type", None)
                stage_id = getattr(output, "stage_id", None)
                if out_type == "text" or stage_id in (None, 0):
                    outputs = getattr(output, "outputs", None) or []
                    if outputs and getattr(outputs[0], "text", None) is not None:
                        latest_text = str(outputs[0].text)
            return clean_display_text(latest_text)

        fut = asyncio.run_coroutine_threadsafe(_run(), self._loop)
        try:
            return fut.result(timeout=self.timeout_sec)
        except BaseException:
            if not fut.done():
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.omni.abort(req_id), self._loop
                    )
                except Exception:
                    pass
            raise

    def stream_generate(
        self,
        prompt: dict[str, Any],
        *,
        sampling_params_list: list[Any],
        output_modalities: list[str],
        cancel_event: threading.Event,
        on_text_delta: Callable[[str], None],
        on_audio_pcm: Callable[[bytes], None],
        poll_barge_in: Optional[Callable[[], None]] = None,
    ) -> None:
        """Stream Stage 0 text + Stage 1 audio via a thread-safe queue (never blocks `_loop`)."""
        item_q: queue.Queue[tuple[str, Any]] = queue.Queue()
        req_id = f"omni-{uuid.uuid4().hex[:12]}"

        async def _pump() -> None:
            finished_normally = False
            try:
                async for output in self.omni.generate(
                    prompt=prompt,
                    request_id=req_id,
                    sampling_params_list=sampling_params_list,
                    output_modalities=output_modalities,
                ):
                    if cancel_event.is_set():
                        break
                    if getattr(output, "error", None):
                        raise RuntimeError(str(output.error))

                    out_type = getattr(output, "final_output_type", None)
                    if out_type is None:
                        out_type = (
                            "audio" if getattr(output, "stage_id", None) == 1 else "text"
                        )

                    outputs = getattr(output, "outputs", None) or []
                    out0 = outputs[0] if outputs else None
                    if out_type == "text":
                        piece = getattr(out0, "text", "") if out0 is not None else ""
                        if piece:
                            item_q.put(("text", str(piece)))
                    elif out_type == "audio" or getattr(output, "stage_id", None) == 1:
                        pcm = _extract_mm_audio_pcm16(output)
                        if pcm:
                            item_q.put(("audio", pcm))
                finished_normally = not cancel_event.is_set()
            except Exception as exc:
                if not cancel_event.is_set():
                    item_q.put(("error", exc))
            finally:
                if not finished_normally:
                    try:
                        await self.omni.abort(req_id)
                    except Exception:
                        pass
                item_q.put(("done", None))

        pump_fut = asyncio.run_coroutine_threadsafe(_pump(), self._loop)
        last_poll = time.perf_counter()
        abort_requested = False

        def _trigger_abort() -> None:
            nonlocal abort_requested
            if abort_requested:
                return
            abort_requested = True
            if not pump_fut.done():
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.omni.abort(req_id), self._loop
                    )
                except Exception:
                    pass

        try:
            while True:
                if cancel_event.is_set():
                    _trigger_abort()
                now = time.perf_counter()
                if (
                    poll_barge_in is not None
                    and not cancel_event.is_set()
                    and (now - last_poll >= 0.32)
                ):
                    last_poll = now
                    poll_barge_in()
                    if cancel_event.is_set():
                        _trigger_abort()

                try:
                    kind, payload = item_q.get(timeout=0.05)
                except queue.Empty:
                    if cancel_event.is_set() and pump_fut.done():
                        break
                    continue

                if kind == "done":
                    break
                if kind == "error":
                    raise payload
                if cancel_event.is_set():
                    _trigger_abort()
                    continue
                if kind == "text":
                    on_text_delta(payload)
                elif kind == "audio":
                    on_audio_pcm(payload)

            join_timeout = 1.5 if cancel_event.is_set() else 5.0
            try:
                pump_fut.result(timeout=join_timeout)
            except Exception:
                if not cancel_event.is_set():
                    raise
        except BaseException:
            _trigger_abort()
            raise

    def shutdown(self) -> None:
        omni_shutdown = getattr(self.omni, "shutdown", None)
        if callable(omni_shutdown):
            try:
                omni_shutdown()
            except Exception:
                pass
        if self._loop.is_running():
            async def _cancel_all() -> None:
                current = asyncio.current_task()
                for task in asyncio.all_tasks():
                    if task is not current and not task.done():
                        task.cancel()

            try:
                asyncio.run_coroutine_threadsafe(_cancel_all(), self._loop).result(timeout=2.0)
            except Exception:
                pass
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# Pure-S2S Live Listener & Conversation Session
# ---------------------------------------------------------------------------


class VllmLiveS2SListener:
    """Pure-S2S always-streaming audio listener backed by native `AsyncOmni`."""

    def __init__(
        self,
        runner: _NativeOmniRunner,
        idle_reset_sec: float = 15.0,
        hard_reset_sec: float = 30.0,
        block_bytes: int = 20480,  # 640 ms @ 16 kHz mono Int16
        overlap_bytes: int = 30720,  # 960 ms left overlap (1.60 s O(1) window)
    ) -> None:
        self._runner = runner
        self.idle_reset_sec = float(idle_reset_sec)
        self.hard_reset_sec = float(hard_reset_sec)
        self.block_bytes = int(block_bytes)
        self.overlap_bytes = int(overlap_bytes)

        self._prev_tail_pcm: bytes = b""
        self._prev_chunk_text: str = ""
        self._active_sec: float = 0.0
        self._idle_sec: float = 0.0
        self.reset_count: int = 0

    @property
    def window_seconds(self) -> float:
        return round(self._active_sec, 2)

    def reset_window(self) -> None:
        self._prev_tail_pcm = b""
        self._prev_chunk_text = ""
        self._active_sec = 0.0
        self._idle_sec = 0.0
        self.reset_count += 1

    def step_block(self, pcm16_block: bytes) -> str:
        """Transcribe a bounded O(1) sliding window via native assistant-turn LiveASR."""
        if not pcm16_block:
            return ""
        window_pcm = self._prev_tail_pcm + pcm16_block
        self._prev_tail_pcm = (
            window_pcm[-self.overlap_bytes :] if self.overlap_bytes > 0 else b""
        )
        wav_f32 = _pcm16_to_float32(window_pcm)
        prompt = self._runner.build_live_asr_prompt(wav_f32)
        raw_text = self._runner.complete_text(
            prompt,
            max_tokens=32,
            temperature=0.0,
            extra_args={"generation_mode": "text"},
        )
        block_dur_sec = len(pcm16_block) / 32000.0
        if not has_real_speech(raw_text):
            self._prev_chunk_text = ""
            self._idle_sec += block_dur_sec
            return ""

        delta = _dedup_overlap_delta(self._prev_chunk_text, raw_text)
        self._prev_chunk_text = raw_text
        self._active_sec = min(self.hard_reset_sec, self._active_sec + block_dur_sec)
        if has_real_speech(delta):
            self._idle_sec = 0.0
        else:
            self._idle_sec += block_dur_sec
        return delta

    def maybe_reset(self, is_idle: bool) -> bool:
        """Reset overlap/window tracking when idle or active duration hits bounds."""
        if self._active_sec >= self.hard_reset_sec or (
            is_idle
            and self._idle_sec >= self.idle_reset_sec
            and (len(self._prev_tail_pcm) > 0 or bool(self._prev_chunk_text))
        ):
            self.reset_window()
            return True
        return False


class VllmStreamingConversationSession:
    """Single-interface S2S conversation orchestrator running against native `AsyncOmni`."""

    def __init__(
        self,
        runner: _NativeOmniRunner,
        settings: EngineSettings,
    ) -> None:
        self._runner = runner
        self.settings = settings
        self.listener = VllmLiveS2SListener(
            runner=runner,
            idle_reset_sec=settings.listener_idle_reset_sec,
            hard_reset_sec=settings.listener_hard_reset_sec,
        )

        self.system_prompt: str = ""
        self.tools: list[ToolSpec] | None = None
        self.language: str = "en"
        self.history: list[TurnRecord] = []
        self.compaction_count: int = 0
        self._rolling_summary: str = ""
        self._pending_filler_text: str = ""

        self._pcm_buf = bytearray()
        self._listen_lock = threading.Lock()
        self._pre_speech_blocks: deque[bytes] = deque(maxlen=settings.pre_speech_blocks)
        self.user_speaking: bool = False
        self.active_user_turn_id: Optional[str] = None
        self.active_user_transcript: str = ""
        self.active_user_pcm_blocks: list[bytes] = []
        self.empty_chunk_streak: int = 0
        self.user_blocks_in_turn: int = 0
        self._last_speech_block_ts: float = 0.0
        self.max_user_blocks: int = max(4, int(round(settings.max_user_turn_sec / 0.64)))

    @property
    def conv_tokens(self) -> int:
        total_chars = len(self.system_prompt) + len(self._rolling_summary)
        for t in self.history:
            total_chars += len(t.text or "")
            if t.user_pcm16:
                total_chars += int((len(t.user_pcm16) / 32000.0) * 50)
        return max(1, total_chars // 4)

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
            self.active_user_turn_id = None
            self.active_user_transcript = ""
            self.active_user_pcm_blocks.clear()
            self.empty_chunk_streak = 0
            self.user_blocks_in_turn = 0
            self.listener.reset_window()
            self.listener.reset_count = 0

    def update_system_config(
        self,
        system_prompt: str,
        tools: list[ToolSpec] | None,
        language: str = "en",
    ) -> None:
        self.system_prompt = system_prompt
        self.tools = list(tools) if tools else None
        self.language = normalize_language_code(language)

    def append_text_user_turn(self, text: str) -> None:
        clean = (text or "").strip()
        if clean:
            self.history.append(TurnRecord(role="user", text=clean))

    def append_tool_exchange(self, calls: list[dict[str, Any]], results_json: str) -> None:
        normalized_calls = _normalize_tool_calls_for_template(calls)
        self.history.append(
            TurnRecord(
                role="assistant",
                text=render_tool_calls(normalized_calls),
                tool_calls=normalized_calls,
            )
        )
        self.history.append(TurnRecord(role="tool", text=results_json))

    def append_delegated_knowledge_context(
        self,
        query: str,
        answer: str,
        source: str,
    ) -> None:
        note = (
            f"[Knowledge Expert ({source}) answer for \"{query}\": {answer}]\n"
            "Now speak the clear, accurate answer concisely to the caller in the active language "
            "without repeating the earlier holding phrase."
        )
        self.history.append(TurnRecord(role="user", text=note))

    def finalize_greeting_turn(self) -> None:
        if len(self.history) >= 2 and self.history[0].role == "user":
            pickup = get_language_profile(self.language).call_pickup_phrase
            self.history[0] = TurnRecord(role="user", text=pickup)

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

    def _commit_current_user_turn(self, events: list[dict[str, Any]]) -> None:
        tid = self.active_user_turn_id or f"usr-{uuid.uuid4().hex[:8]}"
        final_text = self.active_user_transcript.strip() or "(inaudible)"
        endpoint_ms = (
            max(0.0, (time.perf_counter() - self._last_speech_block_ts) * 1000.0)
            if self._last_speech_block_ts > 0
            else None
        )
        pcm_joined = b"".join(self.active_user_pcm_blocks)
        self.history.append(
            TurnRecord(
                role="user",
                text=final_text,
                user_pcm16=pcm_joined if pcm_joined else None,
            )
        )
        self.user_speaking = False
        self.active_user_turn_id = None
        self.active_user_transcript = ""
        self.active_user_pcm_blocks.clear()
        self.empty_chunk_streak = 0
        self.user_blocks_in_turn = 0
        self.listener.reset_window()
        events.append(
            {
                "type": "user_turn_committed",
                "turn_id": tid,
                "text": final_text,
                "endpoint_ms": round(endpoint_ms, 1) if endpoint_ms is not None else None,
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

                        onset_blocks = list(self._pre_speech_blocks)[-1:] + [block]
                        self._pre_speech_blocks.clear()
                        self.active_user_pcm_blocks = list(onset_blocks)

                        if agent_speaking:
                            events.append(
                                {
                                    "type": "barge_in_detected",
                                    "heard_text": self.active_user_transcript,
                                }
                            )
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

            return events

    def flush_ptt_audio(self, pcm16_bytes: bytes) -> list[dict[str, Any]]:
        if not pcm16_bytes:
            return []
        with self._listen_lock:
            self.listener.reset_window()
            profile = get_language_profile(self.language)
            wav_f32 = _pcm16_to_float32(pcm16_bytes)
            prompt = self._runner.build_prompt(
                [
                    {
                        "role": "system",
                        "content": f"Perform ASR in {profile.router_language_name}.",
                    },
                    {"role": "user", "content": [{"type": "audio"}]},
                ],
                audio_clips_16k=[wav_f32],
                add_generation_prompt=True,
            )
            transcript = self._runner.complete_text(
                prompt,
                max_tokens=256,
                temperature=0.0,
                extra_args={"generation_mode": "text"},
            )
            self.listener.reset_window()
            transcript = transcript.strip() or "(spoken message)"
            turn_id = f"usr-{uuid.uuid4().hex[:8]}"
            self.history.append(
                TurnRecord(role="user", text=transcript, user_pcm16=pcm16_bytes)
            )
            return [
                {"type": "user_speech_start", "turn_id": turn_id, "text": transcript},
                {"type": "user_turn_committed", "turn_id": turn_id, "text": transcript},
            ]

    def rollback_interrupted_assistant_turn(
        self,
        played_frames: int,
        interrupted_turn: Optional[TurnRecord],
        replace_last_history: bool,
    ) -> Optional[TurnRecord]:
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

        if interrupted_turn is None or int(played_frames) <= 0:
            if replace_last_history and asst_idx is not None and asst_idx > 1:
                self.history.pop(asst_idx)
            return None

        total_frames = 0
        if interrupted_turn.raw_audio_codes is not None and len(interrupted_turn.raw_audio_codes) > 0:
            total_frames = int(len(interrupted_turn.raw_audio_codes))
        words = (interrupted_turn.text or "").split()
        if not words:
            return None
        if total_frames > 0 and played_frames < total_frames:
            ratio = max(0.1, min(1.0, played_frames / float(total_frames)))
            keep_words = max(1, int(round(len(words) * ratio)))
        else:
            keep_words = max(1, min(len(words), int(played_frames) // 3))
        trunc_text = " ".join(words[:keep_words])
        if keep_words < len(words):
            trunc_text += "..."
        truncated = TurnRecord(role="assistant", text=trunc_text)
        if replace_last_history and asst_idx is not None:
            self.history[asst_idx] = truncated
        elif self.history and self.history[-1].role == "user":
            self.history.insert(len(self.history) - 1, truncated)
        else:
            self.history.append(truncated)
        return truncated

    def maybe_compact_context(self) -> bool:
        if self.conv_tokens < self.settings.compaction_trigger_tokens or len(self.history) <= 6:
            return False
        kept_tail = self.history[-6:]
        older = self.history[:-6]
        digest_lines = [f"{t.role}: {t.text}" for t in older if t.text.strip()]
        if not digest_lines:
            return False
        try:
            prompt_text = (
                "Summarize the following Karoo Crest Bank telesales call state concisely "
                "(customer name, income, card tier preferences, questions answered, tool results):\n\n"
                + "\n".join(digest_lines[-16:])
            )
            prompt = self._runner.build_prompt(
                [
                    {"role": "system", "content": "Produce a compact factual call memory summary."},
                    {"role": "user", "content": prompt_text},
                ],
                add_generation_prompt=True,
            )
            summary = self._runner.complete_text(
                prompt,
                max_tokens=220,
                temperature=0.0,
                extra_args={"generation_mode": "text"},
            )
            if summary:
                self._rolling_summary = summary
        except Exception:
            self._rolling_summary = " | ".join(digest_lines[-6:])
        self.history = list(kept_tail)
        self.compaction_count += 1
        return True

    def _build_prompt(
        self,
        *,
        include_tools: bool = True,
        extra_system_note: str = "",
        force_mixed_prefix: bool = False,
    ) -> dict[str, Any]:
        sys_parts = [self.system_prompt]
        if self._rolling_summary:
            sys_parts.append(f"EARLIER CALL SUMMARY:\n{self._rolling_summary}")
        if self._pending_filler_text:
            sys_parts.append(
                f"You already spoke the holding phrase: \"{self._pending_filler_text}\". "
                "Continue seamlessly with the factual answer."
            )
        if extra_system_note:
            sys_parts.append(extra_system_note)

        tool_schemas: Optional[list[dict[str, Any]]] = None
        if include_tools and self.tools:
            tool_schemas = [t.to_openai_schema() for t in self.tools]

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "\n\n".join(p for p in sys_parts if p)}
        ]

        # Keep raw audio waveforms for the most recent user turns; older turns fall back to text.
        recent_audio_indices: set[int] = set()
        audio_quota = max(1, int(self.settings.keep_recent_user_audio_turns))
        for idx in range(len(self.history) - 1, -1, -1):
            if self.history[idx].role == "user" and self.history[idx].user_pcm16:
                recent_audio_indices.add(idx)
                audio_quota -= 1
                if audio_quota <= 0:
                    break

        audio_clips_16k: list[np.ndarray] = []
        for idx, turn in enumerate(self.history):
            if turn.role == "user":
                if idx in recent_audio_indices and turn.user_pcm16:
                    audio_clips_16k.append(_pcm16_to_float32(turn.user_pcm16))
                    messages.append({"role": "user", "content": [{"type": "audio"}]})
                else:
                    messages.append({"role": "user", "content": turn.text})
            elif turn.role == "assistant":
                if turn.tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": _normalize_tool_calls_for_template(turn.tool_calls),
                        }
                    )
                else:
                    messages.append({"role": "assistant", "content": turn.text})
            elif turn.role == "tool":
                messages.append({"role": "tool", "content": turn.text})

        use_mixed_prefix = bool(force_mixed_prefix and not tool_schemas)
        return self._runner.build_prompt(
            messages,
            tools=tool_schemas,
            audio_clips_16k=audio_clips_16k or None,
            add_generation_prompt=True,
            continue_final_message=False,
            append_suffix="<|mixed_start|>" if use_mixed_prefix else "",
        )

    def _stream_tts_fallback(
        self,
        text: str,
        persona: PersonaConfig,
        cancel_event: threading.Event,
        emit: Callable[[dict[str, Any]], None],
        t_start: float,
        poll_barge_in: Optional[Callable[[], None]] = None,
    ) -> tuple[int, float, int, int]:
        profile = get_language_profile(self.language)
        system_msg = f"Perform TTS in {profile.tts_accent} using a {persona.voice_gender} voice."
        prompt = self._runner.build_prompt(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text},
            ],
            add_generation_prompt=True,
            append_suffix="<|audio_start|>",
        )
        params = self._runner.build_sampling_params_list(
            temperature=0.0,
            max_tokens=int(self.settings.max_slots),
            extra_args={
                "audio_temperature": float(self.settings.audio_temperature),
                "audio_top_k": int(self.settings.audio_top_k),
                "audio_repetition_penalty": float(self.settings.audio_repetition_penalty),
            },
            streaming=True,
        )

        chunks_sent = 0
        first_chunk_samples = 0
        ttfa_ms = 0.0
        total_pcm_bytes = 0

        def on_audio_pcm(pcm: bytes) -> None:
            nonlocal chunks_sent, first_chunk_samples, ttfa_ms, total_pcm_bytes
            if not pcm:
                return
            if ttfa_ms == 0.0:
                ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                first_chunk_samples = len(pcm) // 2
            chunks_sent += 1
            total_pcm_bytes += len(pcm)
            emit({"type": "audio_chunk", "pcm16": pcm, "samples": len(pcm) // 2})

        self._runner.stream_generate(
            prompt,
            sampling_params_list=params,
            output_modalities=["audio"],
            cancel_event=cancel_event,
            on_text_delta=lambda _piece: None,
            on_audio_pcm=on_audio_pcm,
            poll_barge_in=poll_barge_in,
        )

        audio_frames = max(0, total_pcm_bytes // (2 * 1920))
        return audio_frames, ttfa_ms, chunks_sent, first_chunk_samples

    def _stream_omni_chat(
        self,
        *,
        persona: PersonaConfig,
        tools: list[ToolSpec] | None,
        force_interleaved: bool,
        cancel_event: threading.Event,
        on_event: Callable[[dict[str, Any]], None],
        poll_mic_fn: Optional[Callable[[], list[dict[str, Any]]]] = None,
        segment: str = "answer",
        max_tokens: Optional[int] = None,
        extra_system_note: str = "",
    ) -> TurnOutcome:
        t_start = time.perf_counter()
        use_tools = bool(tools) and not force_interleaved
        prompt = self._build_prompt(
            include_tools=use_tools,
            extra_system_note=extra_system_note,
            force_mixed_prefix=not use_tools,
        )
        sampling_params_list = self._runner.build_sampling_params_list(
            temperature=float(self.settings.text_temperature),
            top_p=float(self.settings.text_top_p),
            max_tokens=int(max_tokens or self.settings.max_slots),
            extra_args={
                "audio_temperature": float(self.settings.audio_temperature),
                "audio_top_k": int(self.settings.audio_top_k),
                "audio_repetition_penalty": float(self.settings.audio_repetition_penalty),
            },
            streaming=True,
        )

        raw_text_acc = ""
        clean_text_acc = ""
        ttft_ms = 0.0
        ttfa_ms = 0.0
        chunks_sent = 0
        first_chunk_samples = 0
        total_audio_bytes = 0
        text_streaming_allowed = not use_tools
        tool_mode_detected = False

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
            nonlocal chunks_sent, first_chunk_samples, ttfa_ms, total_audio_bytes, text_streaming_allowed
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
            total_audio_bytes += len(pcm)
            on_event(
                {
                    "type": "audio_chunk",
                    "pcm16": pcm,
                    "samples": len(pcm) // 2,
                }
            )

        def check_barge_in() -> None:
            if poll_mic_fn is None:
                return
            for ev in poll_mic_fn():
                if ev.get("type") == "barge_in_detected":
                    cancel_event.set()
                    return

        self._runner.stream_generate(
            prompt,
            sampling_params_list=sampling_params_list,
            output_modalities=["text", "audio"],
            cancel_event=cancel_event,
            on_text_delta=on_text_delta,
            on_audio_pcm=on_audio_pcm,
            poll_barge_in=check_barge_in if poll_mic_fn is not None else None,
        )

        if cancel_event.is_set():
            partial_frames = total_audio_bytes // (2 * 1920)
            partial_codes = [None] * partial_frames if partial_frames else None
            partial_rec = TurnRecord(
                role="assistant",
                text=clean_text_acc,
                raw_audio_codes=partial_codes,
            )
            return TurnOutcome(
                kind="cancelled",
                turn_record=partial_rec,
                text=clean_text_acc,
                audio_frames=partial_frames,
                ttfa_ms=ttfa_ms,
                ttft_ms=ttft_ms,
                audio_chunks_sent=chunks_sent,
                first_chunk_samples=first_chunk_samples,
                gen_wall_ms=(time.perf_counter() - t_start) * 1000.0,
                cancelled=True,
            )

        parsed_calls = parse_tool_calls(raw_text_acc) if use_tools else []
        if parsed_calls:
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
            fb_frames, fb_ttfa, fb_chunks, fb_first = self._stream_tts_fallback(
                final_text,
                persona,
                cancel_event,
                on_event,
                t_start,
                poll_barge_in=check_barge_in if poll_mic_fn is not None else None,
            )
            chunks_sent += fb_chunks
            total_audio_bytes += fb_frames * 2 * 1920
            if ttfa_ms == 0.0:
                ttfa_ms = fb_ttfa
                first_chunk_samples = fb_first
            if cancel_event.is_set():
                partial_frames = max(0, total_audio_bytes // (2 * 1920))
                partial_rec = TurnRecord(
                    role="assistant",
                    text=final_text,
                    raw_audio_codes=[None] * partial_frames if partial_frames else None,
                )
                return TurnOutcome(
                    kind="cancelled",
                    turn_record=partial_rec,
                    text=final_text,
                    audio_frames=partial_frames,
                    ttfa_ms=ttfa_ms,
                    ttft_ms=ttft_ms,
                    audio_chunks_sent=chunks_sent,
                    first_chunk_samples=first_chunk_samples,
                    gen_wall_ms=(time.perf_counter() - t_start) * 1000.0,
                    cancelled=True,
                    used_tts_fallback=True,
                )

        audio_frames = max(0, total_audio_bytes // (2 * 1920))
        wall_ms = (time.perf_counter() - t_start) * 1000.0
        turn_rec = TurnRecord(
            role="assistant",
            text=final_text,
            raw_audio_codes=[None] * audio_frames if audio_frames else None,
        )
        return TurnOutcome(
            kind="response",
            turn_record=turn_rec,
            text=final_text,
            audio_frames=audio_frames,
            ttfa_ms=ttfa_ms,
            ttft_ms=ttft_ms or ttfa_ms,
            slots_per_sec=round(
                (len(final_text.split()) + audio_frames) / max(wall_ms / 1000.0, 1e-3), 1
            ),
            audio_chunks_sent=chunks_sent,
            first_chunk_samples=first_chunk_samples,
            gen_wall_ms=wall_ms,
            used_tts_fallback=used_tts_fallback,
        )

    def generate_assistant_turn(
        self,
        persona: PersonaConfig,
        tools: list[ToolSpec] | None,
        force_interleaved: bool,
        cancel_event: threading.Event,
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
        )
        self._pending_filler_text = ""
        if outcome.kind == "response" and not outcome.cancelled and outcome.turn_record is not None:
            self.history.append(outcome.turn_record)
        return outcome

    def _run_router_call(self, user_text: str) -> tuple[str, str, float]:
        t0 = time.perf_counter()
        profile = get_language_profile(self.language)
        prev_asst = self.latest_assistant_text()
        ctx_line = f"Previous advisor turn: {prev_asst}\n" if prev_asst else ""
        prompt_text = (
            f"{ctx_line}"
            f"Customer just said: {user_text or '(audio turn)'}\n\n"
            "Classify into exactly ONE line:\n"
            "- Output `DIRECT` for greetings, short rapport, clarifications, or local CRM calculations/applications.\n"
            f"- Output `DELEGATE: <concise standalone question in {profile.router_language_name}>` "
            "ONLY for detailed product policy, tariffs, FX fees, grace days, NCA/FICA/POPIA rules, or travel perks."
        )
        prompt = self._runner.build_prompt(
            [
                {"role": "system", "content": "Output strictly `DIRECT` or `DELEGATE: <question>`."},
                {"role": "user", "content": prompt_text},
            ],
            add_generation_prompt=True,
        )
        raw = self._runner.complete_text(
            prompt,
            max_tokens=self.settings.max_router_tokens,
            temperature=0.0,
            extra_args={"generation_mode": "text"},
        )
        dec, query = _parse_router_query(raw, user_text)
        return dec, query, (time.perf_counter() - t0) * 1000.0

    def _run_query_extract_call(self, user_text: str) -> str:
        profile = get_language_profile(self.language)
        prev_asst = self.latest_assistant_text()
        prompt_text = (
            (f"Previous advisor turn: {prev_asst}\n" if prev_asst else "")
            + f"Customer message: {user_text or '(audio turn)'}\n"
            + f"Rewrite this as one crisp standalone banking knowledge question in {profile.router_language_name}."
        )
        prompt = self._runner.build_prompt(
            [
                {"role": "system", "content": "Output ONLY one concise standalone question."},
                {"role": "user", "content": prompt_text},
            ],
            add_generation_prompt=True,
        )
        raw = self._runner.complete_text(
            prompt,
            max_tokens=self.settings.max_router_tokens,
            temperature=0.0,
            extra_args={"generation_mode": "text"},
        )
        return raw.strip() or user_text.strip() or "(banking question)"

    def run_filler_and_query_batch(
        self,
        cancel_event: threading.Event,
        on_event: Callable[[dict[str, Any]], None],
        poll_mic_fn: Optional[Callable[[], list[dict[str, Any]]]] = None,
    ) -> BatchPlanOutcome:
        """Run parallel continuous-batch filler (stream 0) + query extractor (stream 1)."""
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

        with ThreadPoolExecutor(max_workers=2) as pool:
            q_fut = pool.submit(self._run_query_extract_call, user_text)
            f_fut = pool.submit(
                self._stream_omni_chat,
                persona=persona_stub,
                tools=None,
                force_interleaved=True,
                cancel_event=cancel_event,
                on_event=on_event,
                poll_mic_fn=poll_mic_fn,
                segment="filler",
                max_tokens=self.settings.max_filler_slots,
                extra_system_note=profile.filler_instruction,
            )
            query = q_fut.result()
            if not cancel_event.is_set() and query:
                on_event({"type": "delegation_query_ready", "query": query})
            filler_out = f_fut.result()

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
        cancel_event: threading.Event,
        on_event: Callable[[dict[str, Any]], None],
        poll_mic_fn: Optional[Callable[[], list[dict[str, Any]]]] = None,
    ) -> BatchPlanOutcome:
        """Speculatively run direct/tool generation in parallel with DIRECT/DELEGATE router."""
        user_text = self.latest_user_text()
        profile = get_language_profile(self.language)
        spec_cancel = threading.Event()
        buf_lock = threading.Lock()
        buffered_events: list[dict[str, Any]] = []
        flushed = False

        def combined_cancelled() -> bool:
            if cancel_event.is_set():
                spec_cancel.set()
            return spec_cancel.is_set() or cancel_event.is_set()

        class _LinkedEvent:
            def is_set(self) -> bool:
                return combined_cancelled()

            def set(self) -> None:
                cancel_event.set()
                spec_cancel.set()

        def direct_on_event(ev: dict[str, Any]) -> None:
            with buf_lock:
                if flushed:
                    on_event(ev)
                else:
                    buffered_events.append(ev)

        def guarded_direct_poll() -> list[dict[str, Any]]:
            if poll_mic_fn is None or spec_cancel.is_set() or cancel_event.is_set():
                return []
            return poll_mic_fn()

        with ThreadPoolExecutor(max_workers=2) as pool:
            r_fut = pool.submit(self._run_router_call, user_text)
            d_fut = pool.submit(
                self._stream_omni_chat,
                persona=persona,
                tools=tools,
                force_interleaved=not bool(tools),
                cancel_event=_LinkedEvent(),  # type: ignore[arg-type]
                on_event=direct_on_event,
                poll_mic_fn=guarded_direct_poll if poll_mic_fn is not None else None,
                segment="answer",
            )
            dec, query, route_ms = r_fut.result()

            if cancel_event.is_set():
                spec_cancel.set()
                d_out = d_fut.result()
                return BatchPlanOutcome(
                    decision="cancelled",
                    filler_turn=d_out.turn_record,
                    filler_audio_frames=0,
                    route_decision_ms=route_ms,
                )

            if dec == "direct":
                with buf_lock:
                    flushed = True
                    pending = list(buffered_events)
                    buffered_events.clear()
                for pev in pending:
                    on_event(pev)
                d_out = d_fut.result()
                if d_out.cancelled or cancel_event.is_set():
                    return BatchPlanOutcome(
                        decision="cancelled",
                        filler_turn=d_out.turn_record,
                        filler_audio_frames=0,
                        route_decision_ms=route_ms,
                    )
                if d_out.kind == "tool_calls":
                    return BatchPlanOutcome(
                        decision="tool_calls",
                        direct_outcome=d_out,
                        route_decision_ms=route_ms,
                    )
                self.history.append(
                    d_out.turn_record or TurnRecord(role="assistant", text=d_out.text)
                )
                return BatchPlanOutcome(
                    decision="direct",
                    direct_outcome=d_out,
                    route_decision_ms=route_ms,
                    ttft_ms=d_out.ttft_ms,
                    ttfa_direct_ms=d_out.ttfa_ms,
                    audio_chunks_sent=d_out.audio_chunks_sent,
                    first_chunk_samples=d_out.first_chunk_samples,
                    slots_per_sec=d_out.slots_per_sec,
                )

            # Router chose DELEGATE -> abort speculative direct generation & dispatch query now.
            spec_cancel.set()
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
                    max_tokens=self.settings.max_filler_slots,
                    extra_system_note=profile.filler_instruction,
                )
            d_fut.result()

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


class VllmOmniEngine:
    """Thread-safe in-process engine managing native `vllm_omni.AsyncOmni`."""

    def __init__(
        self,
        settings: EngineSettings,
        omni_client: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.settings = settings
        self._injected_omni = omni_client
        self._injected_tokenizer = tokenizer
        self._runner: Optional[_NativeOmniRunner] = None
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="vllm-lfm3"
        )
        self._loaded: bool = False
        self._load_lock = threading.Lock()
        self.available_models: list[str] = []

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

    def _resolve_deploy_config(self) -> str:
        cfg_path = Path(self.settings.vllm_deploy_config)
        if cfg_path.exists():
            return str(cfg_path)
        repo_cfg = Path(__file__).resolve().parent.parent / self.settings.vllm_deploy_config
        if repo_cfg.exists():
            return str(repo_cfg)
        raise FileNotFoundError(
            f"vLLM-Omni deploy config not found: {self.settings.vllm_deploy_config}"
        )

    def load(self) -> None:
        """Initialize `AsyncOmni` + `AutoTokenizer` natively in-process and run warmup."""
        with self._load_lock:
            if self._loaded:
                return
            t0 = time.perf_counter()
            if self._injected_omni is not None and self._injected_tokenizer is not None:
                self._runner = _NativeOmniRunner(
                    self._injected_omni,
                    self._injected_tokenizer,
                    timeout_sec=self.settings.vllm_timeout_sec,
                )
                self.available_models = [self.settings.model_id]
                self._loaded = True
                return

            from transformers import AutoTokenizer
            from vllm_omni import AsyncOmni

            model_dir = self._resolve_model_dir()
            deploy_cfg = self._resolve_deploy_config()
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            omni = AsyncOmni(
                model=model_dir,
                trust_remote_code=True,
                deploy_config=deploy_cfg,
                async_chunk=True,
                limit_mm_per_prompt={"audio": int(self.settings.vllm_max_audio_per_prompt)},
            )
            runner = _NativeOmniRunner(
                omni,
                tokenizer,
                timeout_sec=self.settings.vllm_timeout_sec,
            )
            warmup_prompt = runner.build_prompt(
                [{"role": "user", "content": "Hi"}],
                add_generation_prompt=True,
            )
            runner.complete_text(
                warmup_prompt,
                max_tokens=2,
                temperature=0.0,
                extra_args={"generation_mode": "text"},
            )
            asr_warmup_prompt = runner.build_live_asr_prompt(
                np.zeros((10240,), dtype=np.float32)
            )
            runner.complete_text(
                asr_warmup_prompt,
                max_tokens=4,
                temperature=0.0,
                extra_args={"generation_mode": "text"},
            )
            self._runner = runner
            self.available_models = [self.settings.model_id]
            self._loaded = True
            elapsed = time.perf_counter() - t0
            print(
                f"[VllmOmniEngine] Loaded native AsyncOmni ({self.settings.model_id}) in {elapsed:.2f}s",
                flush=True,
            )

    def shutdown(self) -> None:
        if self._runner is not None:
            self._runner.shutdown()
            self._runner = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def run_on_worker(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Future[Any]:
        return self._executor.submit(fn, *args, **kwargs)

    def create_pipeline(self) -> VllmStreamingConversationSession:
        if not self._loaded:
            self.load()
        assert self._runner is not None
        return VllmStreamingConversationSession(
            runner=self._runner,
            settings=self.settings,
        )
