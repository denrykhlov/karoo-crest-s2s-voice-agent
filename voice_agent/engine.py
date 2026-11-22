"""Single-model MLX inference engine for LFM3-Omni-2B (S2S + Tool Calling + ASR)."""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import mlx.core as mx
import numpy as np
from mlx_vlm import load as mlx_load
from mlx_vlm.models.lfm3_audio.audio import (
    extract_features,
    num_encoder_tokens,
)

from .codec_stream import StreamingLiquidCodecDecoder, float32_to_pcm16
from .config import EngineSettings, PersonaConfig, ToolSpec
from .protocol import (
    FUNC_RE,
    PARAM_RE,
    SPECIAL_TAG_RE,
    TOOL_BLOCK_RE,
    TurnOutcome,
    TurnRecord,
    clean_display_text,
    parse_tool_calls,
    render_tool_calls,
)


def truncate_assistant_turn_for_barge_in(
    turn: TurnRecord,
    played_frames: int,
    audio_out_token_id: int,
    audio_eos_code: int,
    im_end_id: int,
    tokenizer: Any,
) -> TurnRecord:
    """Truncate an interrupted spoken assistant turn to the audio frames actually heard."""
    if not turn.raw_token_ids or turn.raw_audio_codes is None:
        text = turn.text.rstrip()
        if text and not text.endswith("..."):
            text = f"{text}..."
        return TurnRecord(role="assistant", text=text)

    codes = np.asarray(turn.raw_audio_codes, dtype=np.int32)
    non_eos_total = int(np.sum(~np.any(codes == audio_eos_code, axis=-1)))
    if non_eos_total == 0:
        return TurnRecord(role="assistant", text=turn.text)

    keep_frames = max(1, min(int(played_frames), non_eos_total))
    new_token_ids: list[int] = []
    kept_codes: list[np.ndarray] = []
    kept_text_ids: list[int] = []

    code_idx = 0
    kept_non_eos = 0
    for tok in turn.raw_token_ids:
        if tok == audio_out_token_id:
            if code_idx >= codes.shape[0]:
                break
            row = codes[code_idx]
            code_idx += 1
            if np.any(row == audio_eos_code):
                continue
            if kept_non_eos < keep_frames:
                new_token_ids.append(tok)
                kept_codes.append(row)
                kept_non_eos += 1
                if kept_non_eos >= keep_frames:
                    break
        elif tok == im_end_id:
            continue
        else:
            new_token_ids.append(tok)
            kept_text_ids.append(tok)

    # Terminate speech cleanly with one EOS speech frame and <|im_end|>.
    new_token_ids.append(audio_out_token_id)
    kept_codes.append(np.full((12,), audio_eos_code, dtype=np.int32))
    new_token_ids.append(im_end_id)

    decoded = clean_display_text(
        tokenizer.decode(kept_text_ids, skip_special_tokens=True)
    )
    if decoded and not decoded.endswith("..."):
        decoded = f"{decoded}..."

    return TurnRecord(
        role="assistant",
        text=decoded,
        raw_token_ids=new_token_ids,
        raw_audio_codes=np.stack(kept_codes, axis=0).astype(np.int32),
    )


class Lfm3OmniEngine:
    """Thread-safe MLX execution wrapper around `LiquidAI/LFM3-Omni-2B`."""

    def __init__(self, settings: EngineSettings) -> None:
        self.settings = settings
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-lfm3"
        )
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        self.codec: Any = None
        self._loaded: bool = False
        self._load_lock = threading.Lock()
        self._im_end_id: int = -1
        self._newline_ids: list[int] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Synchronously load the model, processor, and LiquidCodec on the MLX worker."""
        future = self._executor.submit(self._load_sync)
        future.result()

    def _load_sync(self) -> None:
        with self._load_lock:
            if self._loaded:
                return
            t0 = time.perf_counter()
            model, processor = mlx_load(self.settings.model_id)
            codec = model.codec()
            tokenizer = getattr(processor, "tokenizer", processor)

            # Warm up Metal kernels with a minimal 1-frame codec decode.
            dummy = mx.zeros((12, 1), dtype=mx.int32)
            warm_wav = codec.decode(dummy)
            mx.eval(warm_wav)

            self.model = model
            self.processor = processor
            self.tokenizer = tokenizer
            self.codec = codec
            self._im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
            self._newline_ids = list(
                tokenizer.encode("\n", add_special_tokens=False)
            )
            self._loaded = True
            elapsed = time.perf_counter() - t0
            print(
                f"[Lfm3OmniEngine] Loaded {self.settings.model_id} + LiquidCodec in {elapsed:.2f}s",
                flush=True,
            )

    def shutdown(self) -> None:
        """Shut down the dedicated MLX worker thread."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def run_on_worker(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Future[Any]:
        """Execute any MLX operation serialized on the single Metal worker thread."""
        return self._executor.submit(fn, *args, **kwargs)

    def create_pipeline(self) -> Any:
        """Create a per-call streaming conversation session bound to this MLX engine."""
        from .live_pipeline import StreamingConversationSession

        return StreamingConversationSession(
            model=self.model,
            tokenizer=self.tokenizer,
            codec=self.codec,
            settings=self.settings,
        )

    def submit_turn(
        self,
        system_prompt: str,
        history: list[TurnRecord],
        tools: list[ToolSpec] | None,
        persona: PersonaConfig,
        force_interleaved: bool,
        cancel_event: threading.Event,
        on_event: Callable[[dict[str, Any]], None],
    ) -> Future[TurnOutcome]:
        """Schedule a streaming voice turn on the single MLX worker thread."""
        return self._executor.submit(
            self._stream_turn_sync,
            system_prompt,
            list(history),
            tools,
            persona,
            force_interleaved,
            cancel_event,
            on_event,
        )

    def submit_asr(
        self,
        waveform_16k: np.ndarray,
        cancel_event: threading.Event,
    ) -> Future[str]:
        """Schedule background ASR transcription on the single MLX worker thread."""
        return self._executor.submit(
            self._transcribe_audio_sync,
            waveform_16k,
            cancel_event,
        )

    def _encode_text(self, text: str) -> list[int]:
        if not text:
            return []
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _build_prompt_inputs(
        self,
        system_prompt: str,
        history: list[TurnRecord],
        tools: list[ToolSpec] | None,
    ) -> dict[str, Any]:
        cfg = self.model.config
        tool_schemas = [t.to_openai_schema() for t in tools] if tools else None

        system_rendered = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            tools=tool_schemas,
            add_generation_prompt=False,
            tokenize=False,
        )
        prompt_ids: list[int] = self._encode_text(system_rendered)

        # Keep raw audio for the most recent user turns and raw speech codes for
        # the most recent assistant turns so voice identity persists while bounding latency.
        user_audio_indices = [
            idx
            for idx, turn in enumerate(history)
            if turn.role == "user" and turn.user_audio is not None and turn.user_audio.size > 0
        ]
        keep_user_audio_set = set(
            user_audio_indices[-self.settings.max_audio_user_turns :]
        )

        asst_audio_indices = [
            idx
            for idx, turn in enumerate(history)
            if turn.role == "assistant"
            and turn.raw_token_ids
            and turn.raw_audio_codes is not None
            and turn.raw_audio_codes.shape[0] > 0
        ]
        keep_asst_audio_set = set(
            asst_audio_indices[-self.settings.max_audio_assistant_turns :]
        )

        pending_clips: list[np.ndarray] = []
        audio_clip_token_counts: list[int] = []
        if keep_user_audio_set:
            for idx in sorted(keep_user_audio_set):
                pending_clips.append(history[idx].user_audio.astype(np.float32, copy=False))
            features, mask = extract_features(pending_clips)
            audio_clip_token_counts = num_encoder_tokens(mask.sum(-1)).tolist()
        else:
            features = mask = None

        clip_cursor = 0
        out_code_chunks: list[np.ndarray] = []

        for idx, turn in enumerate(history):
            if turn.role == "user":
                if idx in keep_user_audio_set:
                    n_audio_tokens = audio_clip_token_counts[clip_cursor]
                    clip_cursor += 1
                    prompt_ids.extend(self._encode_text("<|im_start|>user\n<|vision_start|>"))
                    prompt_ids.extend([cfg.audio_token_id] * n_audio_tokens)
                    prompt_ids.extend(self._encode_text("<|vision_end|><|im_end|>\n"))
                else:
                    text = turn.text.strip() or "(spoken message)"
                    prompt_ids.extend(
                        self._encode_text(f"<|im_start|>user\n{text}<|im_end|>\n")
                    )

            elif turn.role == "assistant":
                if turn.tool_calls:
                    rendered_calls = render_tool_calls(turn.tool_calls)
                    prompt_ids.extend(
                        self._encode_text(
                            f"<|im_start|>assistant\n<think></think>{rendered_calls}<|im_end|>\n"
                        )
                    )
                elif idx in keep_asst_audio_set:
                    prompt_ids.extend(
                        self._encode_text("<|im_start|>assistant\n<think></think>")
                    )
                    raw_ids = list(turn.raw_token_ids or [])
                    raw_codes = np.asarray(turn.raw_audio_codes, dtype=np.int32)

                    # Validate placeholder/code alignment strictly.
                    placeholder_count = sum(
                        1 for t in raw_ids if t == cfg.audio_out_token_id
                    )
                    if placeholder_count == raw_codes.shape[0]:
                        if not raw_ids or raw_ids[-1] != self._im_end_id:
                            raw_ids.append(self._im_end_id)
                        prompt_ids.extend(raw_ids)
                        prompt_ids.extend(self._newline_ids)
                        out_code_chunks.append(raw_codes)
                    else:
                        fallback_text = turn.text.strip()
                        prompt_ids.extend(
                            self._encode_text(
                                f"{fallback_text}<|im_end|>\n"
                            )
                        )
                else:
                    prompt_ids.extend(
                        self._encode_text(
                            f"<|im_start|>assistant\n<think></think>{turn.text.strip()}<|im_end|>\n"
                        )
                    )

            elif turn.role == "tool":
                prompt_ids.extend(
                    self._encode_text(
                        f"<|im_start|>tool\n<tool_response>\n{turn.text}\n</tool_response><|im_end|>\n"
                    )
                )

        # Open assistant generation prompt with non-reasoning <think></think> block.
        prompt_ids.extend(
            self._encode_text("<|im_start|>assistant\n<think></think>")
        )

        kwargs: dict[str, Any] = {
            "input_ids": mx.array([prompt_ids], dtype=mx.int32),
        }
        if features is not None and mask is not None:
            kwargs["input_features"] = mx.array(features)
            kwargs["feature_attention_mask"] = mx.array(mask)
        if out_code_chunks:
            kwargs["audio_out_codes"] = mx.array(
                np.concatenate(out_code_chunks, axis=0), dtype=mx.int32
            )
        return kwargs

    def _stream_turn_sync(
        self,
        system_prompt: str,
        history: list[TurnRecord],
        tools: list[ToolSpec] | None,
        persona: PersonaConfig,
        force_interleaved: bool,
        cancel_event: threading.Event,
        on_event: Callable[[dict[str, Any]], None],
    ) -> TurnOutcome:
        if not self._loaded:
            self._load_sync()

        if cancel_event.is_set():
            return TurnOutcome(kind="cancelled", cancelled=True)

        cfg = self.model.config
        inputs = self._build_prompt_inputs(system_prompt, history, tools)
        input_ids = inputs.pop("input_ids")

        use_tools = bool(tools) and not force_interleaved
        generation_mode: str | None = None if use_tools else "interleaved"

        raw_token_ids: list[int] = (
            [cfg.mixed_start_token_id] if generation_mode == "interleaved" else []
        )
        raw_audio_codes: list[np.ndarray] = []
        text_token_ids: list[int] = []
        pending_speech_frames: list[np.ndarray] = []

        codec_decoder = StreamingLiquidCodecDecoder(self.codec)
        first_chunk_emitted = False
        emitted_text = ""
        text_streaming_allowed = generation_mode == "interleaved"
        mode_decided = generation_mode == "interleaved"

        t_start = time.perf_counter()
        t_first_slot: float | None = None
        t_first_audio: float | None = None
        total_non_eos_frames = 0
        slot_count = 0

        def flush_audio(finalize: bool = False) -> None:
            nonlocal first_chunk_emitted, t_first_audio
            if not pending_speech_frames and not (finalize and codec_decoder.started):
                return
            if pending_speech_frames:
                codes_chunk = np.stack(pending_speech_frames, axis=-1)  # [12, T]
                n_frames = len(pending_speech_frames)
                pending_speech_frames.clear()
            else:
                codes_chunk = np.zeros((12, 0), dtype=np.int32)
                n_frames = 0

            wav = codec_decoder.decode_chunk(codes_chunk, finalize=finalize)
            if wav.size > 0:
                pcm16 = float32_to_pcm16(wav)
                if t_first_audio is None:
                    t_first_audio = time.perf_counter()
                first_chunk_emitted = True
                on_event(
                    {
                        "type": "audio_chunk",
                        "pcm16": pcm16,
                        "frames": n_frames,
                        "samples": int(wav.size),
                    }
                )

        def maybe_emit_text(force: bool = False) -> None:
            nonlocal emitted_text
            if not text_streaming_allowed or not text_token_ids:
                return
            decoded_full = clean_display_text(
                self.tokenizer.decode(text_token_ids, skip_special_tokens=True)
            )
            if not force and decoded_full.endswith("\ufffd"):
                return
            if decoded_full.startswith(emitted_text):
                delta = decoded_full[len(emitted_text) :]
                if delta:
                    emitted_text = decoded_full
                    on_event({"type": "text_delta", "delta": delta, "text": decoded_full})

        def build_partial_turn_record() -> Optional[TurnRecord]:
            if not raw_token_ids and not raw_audio_codes:
                return None
            partial_text = clean_display_text(
                self.tokenizer.decode(text_token_ids, skip_special_tokens=True)
            )
            p_ids = list(raw_token_ids)
            p_codes = [np.asarray(c, dtype=np.int32) for c in raw_audio_codes]
            if p_codes and not np.any(p_codes[-1] == cfg.audio_eos_code):
                p_codes.append(np.full((12,), cfg.audio_eos_code, dtype=np.int32))
                p_ids.append(cfg.audio_out_token_id)
            if not p_ids or p_ids[-1] != self._im_end_id:
                p_ids.append(self._im_end_id)
            return TurnRecord(
                role="assistant",
                text=partial_text,
                raw_token_ids=p_ids,
                raw_audio_codes=(
                    np.stack(p_codes, axis=0).astype(np.int32) if p_codes else None
                ),
            )

        for token, codes in self.model.stream_slots(
            input_ids,
            max_tokens=self.settings.max_slots,
            generation_mode=generation_mode,
            temperature=self.settings.text_temperature,
            top_p=self.settings.text_top_p,
            repetition_penalty=self.settings.text_repetition_penalty,
            audio_temperature=self.settings.audio_temperature,
            audio_top_k=self.settings.audio_top_k,
            audio_repetition_penalty=self.settings.audio_repetition_penalty,
            audio_repetition_window=self.settings.audio_repetition_window,
            **inputs,
        ):
            if cancel_event.is_set():
                p_turn = build_partial_turn_record()
                return TurnOutcome(
                    kind="cancelled",
                    cancelled=True,
                    turn_record=p_turn,
                    text=p_turn.text if p_turn else "",
                    audio_frames=total_non_eos_frames,
                )

            if t_first_slot is None:
                t_first_slot = time.perf_counter()
            slot_count += 1
            raw_token_ids.append(token)

            if codes is not None:
                codes_np = np.asarray(codes, dtype=np.int32)
                raw_audio_codes.append(codes_np)
                if int(codes_np[0]) == cfg.audio_eos_code:
                    flush_audio(finalize=True)
                else:
                    total_non_eos_frames += 1
                    pending_speech_frames.append(codes_np)
                    target = (
                        self.settings.stream_chunk_frames
                        if first_chunk_emitted
                        else self.settings.first_chunk_frames
                    )
                    if len(pending_speech_frames) >= target:
                        flush_audio(finalize=False)
            else:
                text_token_ids.append(token)
                if not mode_decided:
                    if token == cfg.mixed_start_token_id:
                        mode_decided = True
                        text_streaming_allowed = True
                    elif token == cfg.audio_start_token_id:
                        mode_decided = True
                        text_streaming_allowed = False
                    else:
                        probe = self.tokenizer.decode(
                            text_token_ids, skip_special_tokens=True
                        ).lstrip()
                        if probe.startswith("<tool_call") or probe.startswith("<function"):
                            mode_decided = True
                            text_streaming_allowed = False
                        elif len(probe) >= 16 and "<" not in probe:
                            # Plain text reply (not a tool call). Buffer until end or stream smoothly.
                            mode_decided = True
                            text_streaming_allowed = True
                maybe_emit_text(force=False)

        if cancel_event.is_set():
            p_turn = build_partial_turn_record()
            return TurnOutcome(
                kind="cancelled",
                cancelled=True,
                turn_record=p_turn,
                text=p_turn.text if p_turn else "",
                audio_frames=total_non_eos_frames,
            )

        # Flush any trailing audio frames that were not terminated by an EOS frame.
        if not codec_decoder.finished:
            flush_audio(finalize=True)

        raw_full_text = self.tokenizer.decode(
            text_token_ids, skip_special_tokens=True
        )
        tool_calls = parse_tool_calls(raw_full_text) if use_tools else []
        if tool_calls:
            return TurnOutcome(
                kind="tool_calls",
                tool_calls=tool_calls,
                text=raw_full_text,
                prefill_ms=((t_first_slot - t_start) * 1000.0) if t_first_slot else 0.0,
            )

        clean_text = clean_display_text(raw_full_text)
        # Emit any remaining buffered text (unless it was silent plain text that needs TTS fallback).
        if not emitted_text and clean_text:
            text_streaming_allowed = True
            maybe_emit_text(force=True)

        # Guardrail: if tools were enabled (`generation_mode=None`) and the 2B model
        # produced a plain-text reply without entering <|mixed_start|> / <|audio_start|>,
        # synthesize its speech using LFM3's native TTS prompt so the caller always hears voice!
        if (
            total_non_eos_frames == 0
            and clean_text
            and not cancel_event.is_set()
        ):
            tts_codes, tts_raw_ids = self._synthesize_tts_streaming_sync(
                text=clean_text,
                persona=persona,
                cancel_event=cancel_event,
                on_event=on_event,
            )
            if cancel_event.is_set():
                return TurnOutcome(kind="cancelled", cancelled=True)
            if tts_codes is not None and tts_codes.shape[0] > 0:
                total_non_eos_frames = int(
                    np.sum(~np.any(tts_codes == cfg.audio_eos_code, axis=-1))
                )
                raw_audio_codes = [row for row in tts_codes]
                # Keep text-only representation in raw_token_ids? No: match audio_out_token_id!
                raw_token_ids = (
                    [cfg.audio_start_token_id]
                    + tts_raw_ids
                    + [self._im_end_id]
                )
                if t_first_audio is None:
                    t_first_audio = time.perf_counter()

        # Ensure raw_audio_codes includes trailing EOS if stopped by max_slots mid-speech.
        if raw_audio_codes and not np.any(raw_audio_codes[-1] == cfg.audio_eos_code):
            raw_audio_codes.append(np.full((12,), cfg.audio_eos_code, dtype=np.int32))
            raw_token_ids.append(cfg.audio_out_token_id)

        if not raw_token_ids or raw_token_ids[-1] != self._im_end_id:
            raw_token_ids.append(self._im_end_id)

        t_end = time.perf_counter()
        prefill_ms = ((t_first_slot - t_start) * 1000.0) if t_first_slot else 0.0
        ttfa_ms = ((t_first_audio - t_start) * 1000.0) if t_first_audio else prefill_ms
        gen_time = max(1e-4, t_end - (t_first_slot or t_start))
        slots_per_sec = slot_count / gen_time if slot_count else 0.0

        turn_record = TurnRecord(
            role="assistant",
            text=clean_text,
            raw_token_ids=raw_token_ids,
            raw_audio_codes=(
                np.stack(raw_audio_codes, axis=0).astype(np.int32)
                if raw_audio_codes
                else None
            ),
        )
        return TurnOutcome(
            kind="response",
            turn_record=turn_record,
            text=clean_text,
            audio_frames=total_non_eos_frames,
            prefill_ms=prefill_ms,
            ttfa_ms=ttfa_ms,
            slots_per_sec=slots_per_sec,
        )

    def _synthesize_tts_streaming_sync(
        self,
        text: str,
        persona: PersonaConfig,
        cancel_event: threading.Event,
        on_event: Callable[[dict[str, Any]], None],
    ) -> tuple[Optional[np.ndarray], list[int]]:
        """Fallback TTS synthesis via LFM3-Omni when `generation_mode=None` emitted text only."""
        cfg = self.model.config
        tts_prompt = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": f"Perform TTS in {persona.voice_accent} using a {persona.voice_gender} voice.",
                },
                {"role": "user", "content": text},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = mx.array([self._encode_text(tts_prompt)], dtype=mx.int32)

        codec_decoder = StreamingLiquidCodecDecoder(self.codec)
        pending: list[np.ndarray] = []
        all_codes: list[np.ndarray] = []
        out_token_ids: list[int] = []
        first_emitted = False

        def flush(finalize: bool = False) -> None:
            nonlocal first_emitted
            if not pending and not (finalize and codec_decoder.started):
                return
            if pending:
                chunk = np.stack(pending, axis=-1)
                n_frames = len(pending)
                pending.clear()
            else:
                chunk = np.zeros((12, 0), dtype=np.int32)
                n_frames = 0
            wav = codec_decoder.decode_chunk(chunk, finalize=finalize)
            if wav.size > 0:
                first_emitted = True
                on_event(
                    {
                        "type": "audio_chunk",
                        "pcm16": float32_to_pcm16(wav),
                        "frames": n_frames,
                        "samples": int(wav.size),
                    }
                )

        for token, codes in self.model.stream_slots(
            input_ids,
            max_tokens=self.settings.max_slots,
            generation_mode="audio",
            audio_temperature=self.settings.audio_temperature,
            audio_top_k=self.settings.audio_top_k,
            audio_repetition_penalty=self.settings.audio_repetition_penalty,
            audio_repetition_window=self.settings.audio_repetition_window,
        ):
            if cancel_event.is_set():
                return None, []
            if codes is not None:
                c_np = np.asarray(codes, dtype=np.int32)
                all_codes.append(c_np)
                out_token_ids.append(token)
                if int(c_np[0]) == cfg.audio_eos_code:
                    flush(finalize=True)
                else:
                    pending.append(c_np)
                    target = (
                        self.settings.stream_chunk_frames
                        if first_emitted
                        else self.settings.first_chunk_frames
                    )
                    if len(pending) >= target:
                        flush(finalize=False)

        if not codec_decoder.finished:
            flush(finalize=True)

        if not all_codes:
            return None, []
        return np.stack(all_codes, axis=0).astype(np.int32), out_token_ids

    def _transcribe_audio_sync(
        self,
        waveform_16k: np.ndarray,
        cancel_event: threading.Event,
    ) -> str:
        """Transcribe 16 kHz mono audio using the loaded LFM3-Omni model."""
        if not self._loaded:
            self._load_sync()
        if cancel_event.is_set() or waveform_16k.size == 0:
            return ""

        cfg = self.model.config
        features, mask = extract_features([waveform_16k.astype(np.float32, copy=False)])
        n_audio_tokens = int(num_encoder_tokens(mask.sum(-1))[0])

        prompt_ids: list[int] = []
        prompt_ids.extend(
            self._encode_text(
                "<|startoftext|><|im_start|>system\nPerform ASR in English.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|>"
            )
        )
        prompt_ids.extend([cfg.audio_token_id] * n_audio_tokens)
        prompt_ids.extend(
            self._encode_text(
                "<|vision_end|><|im_end|>\n<|im_start|>assistant\n<think></think>"
            )
        )

        tokens_out: list[int] = []
        for token, _ in self.model.stream_slots(
            mx.array([prompt_ids], dtype=mx.int32),
            max_tokens=256,
            generation_mode="text",
            temperature=0,
            input_features=mx.array(features),
            feature_attention_mask=mx.array(mask),
        ):
            if cancel_event.is_set():
                break
            tokens_out.append(token)

        return clean_display_text(
            self.tokenizer.decode(tokens_out, skip_special_tokens=True)
        )
