"""Pure-S2S always-streaming audio pipeline with zero external VAD (supports 5+ min calls)."""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Optional

import mlx.core as mx
import numpy as np
from mlx_vlm.models.lfm3_audio.depth import (
    embed_frames,
    normalize_eos,
    sample,
    sample_frame,
)

from .audio_stream import (
    StreamingAudioEncoder,
    StreamingLogMel,
    cache_token_count,
    clone_lfm3_cache,
    eval_cache,
    restore_lfm3_cache,
    snapshot_lfm3_cache,
)
from .batch_gen import BatchLfm3Generator
from .codec_stream import (
    StreamingAudioFlusher,
    StreamingLiquidCodecDecoder,
    float32_to_pcm16,
)
from .config import EngineSettings, PersonaConfig, ToolSpec
from .engine import truncate_assistant_turn_for_barge_in
from .languages import get_language_profile, normalize_language_code
from .protocol import (
    BatchPlanOutcome,
    TurnOutcome,
    TurnRecord,
    append_transcript_piece as _append_transcript_piece,
    clean_display_text,
    has_real_speech,
    is_likely_agent_echo,
    is_semantically_complete,
    parse_tool_calls,
    render_tool_calls,
    required_empty_chunks,
)

TEXT, AUDIO, INTERLEAVED = "text", "audio", "interleaved"


class LiveS2SListener:
    """Lightweight continuous `Perform live ASR.` decoder over 960 ms encoder blocks.

    Runs directly on `StreamingAudioEncoder` embeddings with zero external VAD.
    Resets its KV cache on idle pauses after `idle_reset_sec` (15 s) and
    unconditionally after `hard_reset_sec` (30 s) for O(1) memory across 5+ min calls.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        idle_reset_sec: float = 15.0,
        hard_reset_sec: float = 30.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = model.config
        self.backbone = model.language_model.model
        self.idle_reset_tokens = int(round(idle_reset_sec * 12.5))  # 12.5 tokens/s
        self.hard_reset_tokens = int(round(hard_reset_sec * 12.5))

        prompt_ids = tokenizer.encode(
            "<|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        self._prompt_ids = mx.array([prompt_ids], dtype=mx.int32)
        self._text_end_embed = self.backbone.embed_tokens(
            mx.array([[self.cfg.text_end_token_id]], dtype=mx.int32)
        )
        mx.eval(self._prompt_ids, self._text_end_embed)

        suppress = set(model.placeholder_token_ids())
        for tok_str in (
            "<|audio_start|>",
            "<|mixed_start|>",
            "<|im_end|>",
            "<|mixed_end|>",
            "<|endoftext|>",
        ):
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if tid is not None and tid >= 0:
                suppress.add(int(tid))
        suppress.discard(int(self.cfg.text_end_token_id))
        self._suppress_ids = mx.array(sorted(suppress), dtype=mx.int32)

        self.cache: Optional[list[Any]] = None
        self.pending_embed: Optional[mx.array] = None
        self.audio_tokens_in_window: int = 0
        self.reset_count: int = 0

    @property
    def window_seconds(self) -> float:
        return round(self.audio_tokens_in_window / 12.5, 1)

    def reset(self) -> None:
        """Reset the LiveASR context window cleanly."""
        self.cache = self.model.make_cache()
        self.pending_embed = self.backbone.embed_tokens(self._prompt_ids)
        self.audio_tokens_in_window = 0
        mx.eval(self.pending_embed)

    def maybe_reset(self, is_idle: bool) -> bool:
        """Roll over the LiveASR KV window at 15 s (idle) or 30 s (hard cap)."""
        if self.cache is None:
            return False
        should_reset = (
            (is_idle and self.audio_tokens_in_window >= self.idle_reset_tokens)
            or self.audio_tokens_in_window >= self.hard_reset_tokens
        )
        if should_reset:
            self.reset()
            self.reset_count += 1
            return True
        return False

    def step_block(self, embeds: mx.array, max_new_tokens: int = 32) -> str:
        """Decode one block (`[T, hidden_size]`, normally 12 tokens = 960 ms) of audio."""
        if embeds.shape[0] == 0:
            return ""
        if self.cache is None:
            self.reset()

        chunk_3d = embeds[None]
        inp = (
            chunk_3d
            if self.pending_embed is None
            else mx.concatenate([self.pending_embed, chunk_3d], axis=1)
        )
        self.pending_embed = None

        h = self.backbone(None, self.cache, inp)[:, -1]
        gen_tokens: list[int] = []
        text_end_id = int(self.cfg.text_end_token_id)

        for _ in range(max_new_tokens):
            logits = self.backbone.embed_tokens.as_linear(h).astype(mx.float32)
            logits[..., self._suppress_ids] = -mx.inf
            next_tok = int(mx.argmax(logits, axis=-1).item())
            if next_tok == text_end_id:
                break
            gen_tokens.append(next_tok)
            tok_embed = self.backbone.embed_tokens(mx.array([[next_tok]], dtype=mx.int32))
            h = self.backbone(None, self.cache, tok_embed)[:, -1]

        self.pending_embed = self._text_end_embed
        self.audio_tokens_in_window += int(embeds.shape[0])
        eval_cache(self.cache, extra=[h])

        if not gen_tokens:
            return ""
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

    def probe_silence(self, embeds_half: mx.array, max_new_tokens: int = 4) -> bool:
        """Non-mutating 480 ms trailing check returning True when LiveASR hears no words."""
        if embeds_half.shape[0] == 0 or self.cache is None:
            return False
        probe_cache = clone_lfm3_cache(self.cache)
        chunk_3d = embeds_half[None]
        inp = (
            chunk_3d
            if self.pending_embed is None
            else mx.concatenate([self.pending_embed, chunk_3d], axis=1)
        )
        h = self.backbone(None, probe_cache, inp)[:, -1]
        gen_tokens: list[int] = []
        text_end_id = int(self.cfg.text_end_token_id)
        for _ in range(max_new_tokens):
            logits = self.backbone.embed_tokens.as_linear(h).astype(mx.float32)
            logits[..., self._suppress_ids] = -mx.inf
            next_tok = int(mx.argmax(logits, axis=-1).item())
            if next_tok == text_end_id:
                break
            gen_tokens.append(next_tok)
            tok_embed = self.backbone.embed_tokens(mx.array([[next_tok]], dtype=mx.int32))
            h = self.backbone(None, probe_cache, tok_embed)[:, -1]
        text = (
            self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            if gen_tokens
            else ""
        )
        return not has_real_speech(text)


class StreamingConversationSession:
    """Single-model always-streaming S2S conversation state for 5+ minute calls.

    All methods must be called from the single MLX worker thread in `Lfm3OmniEngine`.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        codec: Any,
        settings: EngineSettings,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.codec = codec
        self.settings = settings
        self.cfg = model.config
        self.backbone = model.language_model.model

        self._im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
        self._newline_ids = list(tokenizer.encode("\n", add_special_tokens=False))

        self.mel_stream = StreamingLogMel()
        self.audio_encoder = StreamingAudioEncoder(model)
        self.listener = LiveS2SListener(
            model,
            tokenizer,
            idle_reset_sec=settings.listener_idle_reset_sec,
            hard_reset_sec=settings.listener_hard_reset_sec,
        )

        self.system_prompt: str = ""
        self.tools: list[ToolSpec] | None = None
        self.language: str = "en"
        self.conv_cache: Optional[list[Any]] = None
        self.pre_turn_snapshot: Optional[list[tuple[Any, ...]]] = None
        self.history: list[TurnRecord] = []
        self.compaction_count: int = 0

        self._embed_buf: Optional[mx.array] = None  # [T, 2048]
        self._pre_speech_blocks: deque[mx.array] = deque(
            maxlen=settings.pre_speech_blocks
        )

        self.user_speaking: bool = False
        self.user_turn_open: bool = False
        self.active_user_turn_id: Optional[str] = None
        self.active_user_transcript: str = ""
        self.active_user_embeds: list[mx.array] = []
        self.empty_chunk_streak: int = 0
        self.user_blocks_in_turn: int = 0
        self._last_speech_block_ts: float = 0.0
        self._half_stride_probed_blocks: int = -1
        self.pending_barge_in_blocks: Optional[list[mx.array]] = None
        self.max_user_blocks: int = max(
            4, int(round(settings.max_user_turn_sec / 0.96))
        )
        self.batch_gen = BatchLfm3Generator(
            model=model,
            tokenizer=tokenizer,
            codec=codec,
            settings=settings,
            tts_fallback_fn=self._synthesize_tts_streaming_sync,
        )

    @property
    def conv_tokens(self) -> int:
        return cache_token_count(self.conv_cache)

    def _encode_text(self, text: str) -> list[int]:
        if not text:
            return []
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _prefill_embeds(self, embeds: mx.array) -> mx.array:
        """Feed `[1, T, hidden_size]` into `self.conv_cache` and return last hidden `[1, hidden]`."""
        h = self.backbone(None, self.conv_cache, embeds)[:, -1]
        eval_cache(self.conv_cache, extra=[h])
        return h

    def _prefill_token_ids(self, token_ids: list[int]) -> Optional[mx.array]:
        if not token_ids:
            return None
        ids_mx = mx.array([token_ids], dtype=mx.int32)
        embeds = self.backbone.embed_tokens(ids_mx)
        return self._prefill_embeds(embeds)

    def _render_system_ids(
        self, system_prompt: str, tools: list[ToolSpec] | None
    ) -> list[int]:
        tool_schemas = [t.to_openai_schema() for t in tools] if tools else None
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            tools=tool_schemas,
            add_generation_prompt=False,
            tokenize=False,
        )
        return self._encode_text(rendered)

    def initialize_conversation(
        self,
        system_prompt: str,
        tools: list[ToolSpec] | None,
        language: str = "en",
    ) -> None:
        """Start a fresh call conversation cache with the system prompt and tool definitions."""
        self.system_prompt = system_prompt
        self.tools = list(tools) if tools else None
        self.language = normalize_language_code(language)
        self.mel_stream.reset()
        self.audio_encoder.reset()
        self.listener.reset()
        self.listener.reset_count = 0

        self.conv_cache = self.model.make_cache()
        self.pre_turn_snapshot = None
        self.history.clear()
        self.compaction_count = 0
        self._embed_buf = None
        self._pre_speech_blocks.clear()

        self.user_speaking = False
        self.user_turn_open = False
        self.active_user_turn_id = None
        self.active_user_transcript = ""
        self.active_user_embeds.clear()
        self.empty_chunk_streak = 0
        self.user_blocks_in_turn = 0
        self._last_speech_block_ts = 0.0
        self._half_stride_probed_blocks = -1
        self.pending_barge_in_blocks = None

        sys_ids = self._render_system_ids(system_prompt, self.tools)
        self._prefill_token_ids(sys_ids)

    def update_system_config(
        self,
        system_prompt: str,
        tools: list[ToolSpec] | None,
        language: Optional[str] = None,
    ) -> None:
        """Rebuild the persistent cache cleanly if persona, language, or tools toggle mid-call."""
        self.system_prompt = system_prompt
        self.tools = list(tools) if tools else None
        if language is not None:
            self.language = normalize_language_code(language)
        if not self.user_speaking and not self.user_turn_open:
            self._rebuild_cache_from_history()

    def _open_user_turn_in_cache(self, onset_blocks: list[mx.array]) -> None:
        prefix_ids = self._encode_text("<|im_start|>user\n<|vision_start|>")
        prefix_emb = self.backbone.embed_tokens(mx.array([prefix_ids], dtype=mx.int32))
        if onset_blocks:
            audio_emb = mx.concatenate(onset_blocks, axis=0)[None]
            combined = mx.concatenate([prefix_emb, audio_emb], axis=1)
        else:
            combined = prefix_emb
        self._prefill_embeds(combined)
        self.user_turn_open = True

    def _commit_current_user_turn(self, events: list[dict[str, Any]]) -> None:
        now = time.perf_counter()
        endpoint_ms = (
            round((now - self._last_speech_block_ts) * 1000.0, 1)
            if self._last_speech_block_ts > 0.0
            else None
        )
        turn_id = self.active_user_turn_id or f"usr-{uuid.uuid4().hex[:8]}"
        transcript = self.active_user_transcript.strip() or "(spoken message)"
        user_embeds = (
            mx.concatenate(self.active_user_embeds, axis=0)
            if self.active_user_embeds
            else None
        )
        if user_embeds is not None:
            mx.eval(user_embeds)
        self.history.append(
            TurnRecord(
                role="user",
                text=transcript,
                user_audio_embeds=user_embeds,
            )
        )
        events.append(
            {
                "type": "user_turn_committed",
                "turn_id": turn_id,
                "text": transcript,
                "endpoint_ms": endpoint_ms,
                "commit_ts": now,
            }
        )
        self.user_speaking = False
        self.active_user_turn_id = None
        self.active_user_transcript = ""
        self.active_user_embeds.clear()
        self.empty_chunk_streak = 0
        self.user_blocks_in_turn = 0
        self._last_speech_block_ts = 0.0
        self._half_stride_probed_blocks = -1

    def ingest_pcm_and_listen(
        self,
        pcm16_bytes: bytes,
        agent_speaking: bool = False,
        recent_agent_text: str = "",
        patience: str = "balanced",
    ) -> list[dict[str, Any]]:
        """Stream raw 16 kHz Int16 PCM into causal encoder -> LiveASR + conv_cache."""
        if not pcm16_bytes or self.conv_cache is None:
            return []
        samples = np.frombuffer(pcm16_bytes, dtype="<i2").astype(np.float32) / 32768.0
        mel = self.mel_stream.push(samples, is_final=False)
        if mel.shape[1] == 0:
            return []
        new_emb = self.audio_encoder.push(mel, is_final=False)
        if new_emb.shape[0] == 0:
            return []

        if self._embed_buf is None or self._embed_buf.shape[0] == 0:
            self._embed_buf = new_emb
        else:
            self._embed_buf = mx.concatenate([self._embed_buf, new_emb], axis=0)
        mx.eval(self._embed_buf)

        events: list[dict[str, Any]] = []
        chunk_tokens = self.settings.live_chunk_tokens

        while self._embed_buf is not None and self._embed_buf.shape[0] >= chunk_tokens:
            block = self._embed_buf[:chunk_tokens]
            self._embed_buf = self._embed_buf[chunk_tokens:]
            mx.eval(block)

            chunk_text = self.listener.step_block(block)
            speech = has_real_speech(chunk_text)

            if not self.user_speaking:
                if speech:
                    if agent_speaking and is_likely_agent_echo(
                        chunk_text, recent_agent_text, lang_code=self.language
                    ):
                        self._pre_speech_blocks.append(block)
                        self.listener.maybe_reset(is_idle=True)
                        continue

                    turn_id = f"usr-{uuid.uuid4().hex[:8]}"
                    self.user_speaking = True
                    self.active_user_turn_id = turn_id
                    self.active_user_transcript = chunk_text.strip()
                    self.empty_chunk_streak = 0
                    self.user_blocks_in_turn = 1
                    self._last_speech_block_ts = time.perf_counter()
                    self._half_stride_probed_blocks = -1

                    # Include 1 preceding block (960 ms) to capture speech onset before LiveASR fires.
                    onset_blocks = list(self._pre_speech_blocks)[-1:] + [block]
                    self._pre_speech_blocks.clear()
                    self.active_user_embeds = list(onset_blocks)

                    if agent_speaking:
                        self.pending_barge_in_blocks = list(onset_blocks)
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
                    self.active_user_transcript = _append_transcript_piece(
                        self.active_user_transcript, chunk_text
                    )
                    self.active_user_embeds.append(block)
                    if self.pending_barge_in_blocks is not None:
                        self.pending_barge_in_blocks.append(block)
                    else:
                        self._prefill_embeds(block[None])
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
                        self.active_user_embeds.append(block)
                        if self.pending_barge_in_blocks is not None:
                            self.pending_barge_in_blocks.append(block)
                        else:
                            self._prefill_embeds(block[None])

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

        # Half-stride (480 ms) non-mutating speculative silence check for snappy endpointing
        probe_toks = int(self.settings.half_stride_probe_tokens)
        if (
            self.user_speaking
            and self.empty_chunk_streak == 0
            and (patience or "balanced").lower() in ("snappy", "balanced")
            and self._embed_buf is not None
            and self._embed_buf.shape[0] >= probe_toks
            and self._half_stride_probed_blocks != self.user_blocks_in_turn
            and is_semantically_complete(self.active_user_transcript, lang_code=self.language)
        ):
            self._half_stride_probed_blocks = self.user_blocks_in_turn
            half_tail = self._embed_buf[:probe_toks]
            if self.listener.probe_silence(half_tail):
                self._embed_buf = self._embed_buf[probe_toks:]
                mx.eval(half_tail)
                self.active_user_embeds.append(half_tail)
                if self.pending_barge_in_blocks is not None:
                    self.pending_barge_in_blocks.append(half_tail)
                else:
                    self._prefill_embeds(half_tail[None])
                self._commit_current_user_turn(events)
                self.listener.maybe_reset(is_idle=True)

        return events

    def flush_ptt_audio(self, pcm16_bytes: bytes) -> list[dict[str, Any]]:
        """Process a complete Push-to-Talk capture via S2S encoder + LiveASR without VAD."""
        if not pcm16_bytes or self.conv_cache is None:
            return []
        self.mel_stream.reset()
        self.audio_encoder.reset()
        self._embed_buf = None
        self._pre_speech_blocks.clear()

        samples = np.frombuffer(pcm16_bytes, dtype="<i2").astype(np.float32) / 32768.0
        mel = self.mel_stream.push(samples, is_final=True)
        emb = self.audio_encoder.push(mel, is_final=True)
        # Re-initialize stream encoders cleanly for subsequent turns.
        self.mel_stream.reset()
        self.audio_encoder.reset()

        if emb.shape[0] == 0:
            return []

        transcript_parts: list[str] = []
        chunk_tokens = self.settings.live_chunk_tokens
        for i in range(0, int(emb.shape[0]), chunk_tokens):
            part = self.listener.step_block(emb[i : i + chunk_tokens])
            if has_real_speech(part):
                transcript_parts.append(part)
        self.listener.maybe_reset(is_idle=True)

        transcript = "".join(transcript_parts).strip() or "(spoken message)"
        turn_id = f"usr-{uuid.uuid4().hex[:8]}"
        self._open_user_turn_in_cache([emb])
        self.history.append(
            TurnRecord(role="user", text=transcript, user_audio_embeds=emb)
        )
        return [
            {"type": "user_speech_start", "turn_id": turn_id, "text": transcript},
            {"type": "user_turn_committed", "turn_id": turn_id, "text": transcript},
        ]

    def append_text_user_turn(self, text: str) -> None:
        """Append and prefill a typed user message into the live cache."""
        clean = (text or "").strip()
        if not clean:
            return
        self._prefill_token_ids(
            self._encode_text(f"<|im_start|>user\n{clean}<|im_end|>\n")
        )
        self.history.append(TurnRecord(role="user", text=clean))

    def append_tool_exchange(
        self, tool_calls: list[dict[str, Any]], tool_results_json: str
    ) -> None:
        """Append and prefill assistant tool_calls + tool response into the live cache."""
        self.history.append(TurnRecord(role="assistant", tool_calls=tool_calls))
        self.history.append(TurnRecord(role="tool", text=tool_results_json))
        rendered = (
            f"<|im_start|>assistant\n<think></think>{render_tool_calls(tool_calls)}<|im_end|>\n"
            f"<|im_start|>tool\n<tool_response>\n{tool_results_json}\n</tool_response><|im_end|>\n"
        )
        self._prefill_token_ids(self._encode_text(rendered))

    def _assistant_turn_embeds(self, turn: TurnRecord) -> mx.array:
        """Build exact `[1, T, hidden_size]` embeddings for a spoken assistant turn."""
        if turn.raw_token_ids and turn.raw_audio_codes is not None:
            raw_ids = list(turn.raw_token_ids)
            raw_codes = np.asarray(turn.raw_audio_codes, dtype=np.int32)
            placeholder_count = sum(
                1 for t in raw_ids if t == self.cfg.audio_out_token_id
            )
            if placeholder_count == raw_codes.shape[0]:
                if not raw_ids or raw_ids[-1] != self._im_end_id:
                    raw_ids.append(self._im_end_id)
                prefix_ids = self._encode_text("<|im_start|>assistant\n<think></think>")
                full_ids = prefix_ids + raw_ids + self._newline_ids
                embeds = self.backbone.embed_tokens(mx.array([full_ids], dtype=mx.int32))
                indices = np.flatnonzero(
                    np.asarray(full_ids) == self.cfg.audio_out_token_id
                )
                audio_embs = embed_frames(
                    self.model.codebook_embedding, mx.array(raw_codes, dtype=mx.int32)
                )
                flat = embeds.reshape(-1, embeds.shape[-1])
                flat[mx.array(indices)] = audio_embs.astype(embeds.dtype)
                return flat.reshape(embeds.shape)

        ids = self._encode_text(
            f"<|im_start|>assistant\n<think></think>{turn.text.strip()}<|im_end|>\n"
        )
        return self.backbone.embed_tokens(mx.array([ids], dtype=mx.int32))

    def finalize_greeting_turn(self) -> None:
        """Replace synthetic greeting trigger with natural caller pickup phrase in conv_cache."""
        if self.conv_cache is None or not self.history:
            return
        if self.history[0].role == "user":
            pickup = get_language_profile(self.language).call_pickup_phrase
            self.history[0] = TurnRecord(role="user", text=pickup)
        if not self.user_speaking and not self.user_turn_open:
            self._rebuild_cache_from_history()

    def rollback_interrupted_assistant_turn(
        self,
        played_frames: int,
        interrupted_turn: Optional[TurnRecord],
        replace_last_history: bool,
    ) -> Optional[TurnRecord]:
        """Roll back `conv_cache` to `pre_turn_snapshot`, replay heard frames, and open barge-in audio."""
        if self.conv_cache is None:
            return None

        if self.pre_turn_snapshot is None:
            if self.pending_barge_in_blocks is not None:
                if not self.user_turn_open:
                    self._open_user_turn_in_cache(self.pending_barge_in_blocks)
                else:
                    self._prefill_embeds(mx.concatenate(self.pending_barge_in_blocks, axis=0)[None])
                self.pending_barge_in_blocks = None
            return None

        # If caller spoke during the final trailing ~320 ms of a completed reply, keep the full turn.
        if (
            replace_last_history
            and interrupted_turn is not None
            and interrupted_turn.raw_audio_codes is not None
        ):
            total_non_eos = int(
                np.sum(
                    ~np.any(
                        np.asarray(interrupted_turn.raw_audio_codes) == self.cfg.audio_eos_code,
                        axis=-1,
                    )
                )
            )
            if total_non_eos > 0 and int(played_frames) >= max(1, total_non_eos - 4):
                self.pre_turn_snapshot = None
                if self.pending_barge_in_blocks is not None:
                    if not self.user_turn_open:
                        self._open_user_turn_in_cache(self.pending_barge_in_blocks)
                    else:
                        self._prefill_embeds(
                            mx.concatenate(self.pending_barge_in_blocks, axis=0)[None]
                        )
                    self.pending_barge_in_blocks = None
                return None

        restore_lfm3_cache(self.conv_cache, self.pre_turn_snapshot)
        self.pre_turn_snapshot = None

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
                audio_out_token_id=self.cfg.audio_out_token_id,
                audio_eos_code=self.cfg.audio_eos_code,
                im_end_id=self._im_end_id,
                tokenizer=self.tokenizer,
            )
            if replace_last_history and asst_idx is not None:
                self.history[asst_idx] = truncated
            elif self.history and self.history[-1].role == "user":
                self.history.insert(len(self.history) - 1, truncated)
            else:
                self.history.append(truncated)

            asst_embeds = self._assistant_turn_embeds(truncated)
            self._prefill_embeds(asst_embeds)
        elif replace_last_history and asst_idx is not None:
            if asst_idx > 1:
                self.history.pop(asst_idx)
            elif interrupted_turn is not None:
                preserved = TurnRecord(role="assistant", text=interrupted_turn.text)
                self.history[asst_idx] = preserved
                self._prefill_embeds(self._assistant_turn_embeds(preserved))

        if self.pending_barge_in_blocks is not None:
            self._open_user_turn_in_cache(self.pending_barge_in_blocks)
            self.pending_barge_in_blocks = None

        return truncated

    def latest_user_text(self) -> str:
        """Return the most recent user message text from conversation history."""
        for turn in reversed(self.history):
            if turn.role == "user" and turn.text.strip():
                return turn.text.strip()
        return ""

    def latest_user_audio_embeds(self) -> Optional[mx.array]:
        """Return the audio embeddings for the most recent spoken user turn, if any."""
        for turn in reversed(self.history):
            if turn.role == "user":
                return turn.user_audio_embeds
        return None

    def latest_assistant_text(self) -> str:
        """Return the most recent assistant spoken text for minimal router context."""
        for turn in reversed(self.history):
            if turn.role == "assistant" and turn.text.strip():
                return turn.text.strip()
        return ""

    def prepare_for_batched_generation(self) -> str:
        """Close any open user audio turn and snapshot `conv_cache` before B=2 generation."""
        if self.conv_cache is None:
            raise RuntimeError("Conversation cache is not initialized.")
        if self.user_turn_open:
            close_ids = self._encode_text("<|vision_end|><|im_end|>\n")
            self._prefill_token_ids(close_ids)
            self.user_turn_open = False
        self.pre_turn_snapshot = snapshot_lfm3_cache(self.conv_cache)
        return self.latest_user_text()

    def run_filler_and_query_batch(
        self,
        cancel_event: threading.Event,
        on_event: Callable[[dict[str, Any]], None],
        poll_mic_fn: Optional[Callable[[], list[dict[str, Any]]]] = None,
    ) -> BatchPlanOutcome:
        """Run B=2 parallel voice filler (Row 0) + Flametree query extraction (Row 1)."""
        latest_text = self.prepare_for_batched_generation()
        assert self.conv_cache is not None
        return self.batch_gen.run_filler_and_query_batch(
            conv_cache=self.conv_cache,
            language_code=self.language,
            latest_user_text=latest_text,
            cancel_event=cancel_event,
            on_event=on_event,
            poll_mic_fn=poll_mic_fn,
            user_audio_embeds=self.latest_user_audio_embeds(),
            prev_assistant_text=self.latest_assistant_text(),
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
        """Run B=2 speculative router (DIRECT vs DELEGATE) + direct speech or filler."""
        latest_text = self.prepare_for_batched_generation()
        assert self.conv_cache is not None
        plan = self.batch_gen.run_speculative_selective_batch(
            conv_cache=self.conv_cache,
            tools=tools,
            persona=persona,
            language_code=self.language,
            latest_user_text=latest_text,
            dynamic_filler_enabled=dynamic_filler_enabled,
            cancel_event=cancel_event,
            on_event=on_event,
            poll_mic_fn=poll_mic_fn,
            user_audio_embeds=self.latest_user_audio_embeds(),
            prev_assistant_text=self.latest_assistant_text(),
        )
        if plan.decision == "direct" and plan.direct_outcome and plan.direct_outcome.turn_record:
            turn_rec = plan.direct_outcome.turn_record
            self.history.append(turn_rec)
            if plan.updated_conv_cache is not None:
                self.conv_cache = plan.updated_conv_cache
            elif self.pre_turn_snapshot is not None:
                restore_lfm3_cache(self.conv_cache, self.pre_turn_snapshot)
                self._prefill_embeds(self._assistant_turn_embeds(turn_rec))
            eval_cache(self.conv_cache)
        elif plan.decision == "tool_calls" and self.pre_turn_snapshot is not None:
            restore_lfm3_cache(self.conv_cache, self.pre_turn_snapshot)
            self.pre_turn_snapshot = None
        return plan

    def append_delegated_knowledge_context(
        self,
        query: str,
        answer: str,
        source: str,
    ) -> None:
        """Inject retrieved Flametree Knowledge Expert answer cleanly into `conv_cache`."""
        import json as _json

        tool_calls = [
            {
                "type": "function",
                "function": {
                    "name": "consult_flametree_knowledge",
                    "arguments": {"question": query},
                },
            }
        ]
        payload = _json.dumps(
            {
                "ok": True,
                "source": source,
                "grounded_facts": answer,
                "delivery_note": (
                    "Speak the grounded facts naturally and concisely in the caller's active language. "
                    "Do not invent extra rates or fees."
                ),
            },
            ensure_ascii=False,
        )
        self.append_tool_exchange(tool_calls, payload)

    def generate_assistant_turn(
        self,
        persona: PersonaConfig,
        tools: list[ToolSpec] | None,
        force_interleaved: bool,
        cancel_event: threading.Event,
        on_event: Callable[[dict[str, Any]], None],
        poll_mic_fn: Optional[Callable[[], list[dict[str, Any]]]] = None,
        segment_name: str = "answer",
    ) -> TurnOutcome:
        """Close any open user audio turn and generate an assistant reply on `conv_cache`."""
        if cancel_event.is_set() or self.conv_cache is None:
            return TurnOutcome(kind="cancelled", cancelled=True)

        t_start = time.perf_counter()

        # 1. Close open user turn if needed, then capture pre-turn snapshot BEFORE assistant header.
        if self.user_turn_open:
            close_ids = self._encode_text("<|vision_end|><|im_end|>\n")
            self._prefill_token_ids(close_ids)
            self.user_turn_open = False

        self.pre_turn_snapshot = snapshot_lfm3_cache(self.conv_cache)

        use_tools = bool(tools) and not force_interleaved
        generation_mode: Optional[str] = None if use_tools else INTERLEAVED

        header_str = "<|im_start|>assistant\n<think></think>"
        header_ids = self._encode_text(header_str)
        if generation_mode == INTERLEAVED:
            header_ids.append(int(self.cfg.mixed_start_token_id))
            raw_token_ids = [int(self.cfg.mixed_start_token_id)]
            mode = INTERLEAVED
        else:
            raw_token_ids = []
            mode = TEXT

        hidden = self._prefill_token_ids(header_ids)
        assert hidden is not None

        eos_ids = self.cfg.eos_token_id
        eos_id_set = set(
            eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        )
        suppress = list(self.model.placeholder_token_ids())

        raw_audio_codes: list[np.ndarray] = []
        text_token_ids: list[int] = []

        flusher = StreamingAudioFlusher(
            self.codec,
            first_chunk_frames=self.settings.first_chunk_frames,
            stream_chunk_frames=self.settings.stream_chunk_frames,
            on_event=on_event,
            segment_name=segment_name,
        )
        emitted_text = ""
        text_streaming_allowed = generation_mode == INTERLEAVED
        mode_decided = generation_mode == INTERLEAVED

        text_history: deque[int] = deque(maxlen=32)
        audio_history: deque[mx.array] = deque(maxlen=self.settings.audio_repetition_window)
        slots_left = int(self.cfg.interleaved_n_text)
        speaking = False
        text_done = False

        t_first_slot: Optional[float] = None
        t_first_tok: Optional[float] = None
        total_non_eos_frames = 0
        slot_count = 0

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
                    on_event(
                        {
                            "type": "text_delta",
                            "delta": delta,
                            "text": decoded_full,
                            "segment": segment_name,
                        }
                    )

        def build_partial_turn() -> Optional[TurnRecord]:
            if not raw_token_ids and not raw_audio_codes:
                return None
            partial_text = clean_display_text(
                self.tokenizer.decode(text_token_ids, skip_special_tokens=True)
            )
            p_ids = list(raw_token_ids)
            p_codes = [np.asarray(c, dtype=np.int32) for c in raw_audio_codes]
            if p_codes and not np.any(p_codes[-1] == self.cfg.audio_eos_code):
                p_codes.append(np.full((12,), self.cfg.audio_eos_code, dtype=np.int32))
                p_ids.append(int(self.cfg.audio_out_token_id))
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

        last_unfed_embed: Optional[mx.array] = None
        last_token: Optional[int] = None
        finished = False

        for step in range(self.settings.max_slots):
            if cancel_event.is_set():
                p_turn = build_partial_turn()
                return TurnOutcome(
                    kind="cancelled",
                    cancelled=True,
                    turn_record=p_turn,
                    text=p_turn.text if p_turn else "",
                    audio_frames=total_non_eos_frames,
                )

            # Cooperative mid-generation listening so barge-in fires even during long outputs.
            if (
                poll_mic_fn is not None
                and step > 0
                and step % self.settings.coop_listen_slot_interval == 0
            ):
                mic_events = poll_mic_fn()
                if any(ev.get("type") == "barge_in_detected" for ev in mic_events):
                    cancel_event.set()
                    p_turn = build_partial_turn()
                    return TurnOutcome(
                        kind="cancelled",
                        cancelled=True,
                        turn_record=p_turn,
                        text=p_turn.text if p_turn else "",
                        audio_frames=total_non_eos_frames,
                    )

            interleaving = mode == INTERLEAVED
            if mode == AUDIO or (interleaving and speaking):
                audio_hist = (
                    mx.stack(list(audio_history))[None] if audio_history else None
                )
                codes = sample_frame(
                    self.model.depth_decoder,
                    hidden,
                    self.cfg.maskgit_schedule,
                    self.settings.audio_temperature,
                    self.settings.audio_top_k,
                    self.settings.audio_repetition_penalty,
                    audio_hist,
                    None,
                )[0]
                codes = normalize_eos(codes, self.cfg.audio_eos_code)
                codes_np = np.asarray(codes, dtype=np.int32)
                raw_token_ids.append(int(self.cfg.audio_out_token_id))
                raw_audio_codes.append(codes_np)
                if int(codes_np[0]) == self.cfg.audio_eos_code:
                    mode = TEXT
                    flusher.flush(finalize=True)
                else:
                    audio_history.append(codes)
                    total_non_eos_frames += 1
                    flusher.push_frame(codes_np)
                embed = embed_frames(self.model.codebook_embedding, codes)
                last_token = int(self.cfg.audio_out_token_id)
            else:
                hist_mx = mx.array([list(text_history)]) if text_history else None
                token = sample(
                    self.backbone.embed_tokens.as_linear(hidden),
                    self.settings.text_temperature,
                    None,
                    self.settings.text_repetition_penalty,
                    hist_mx,
                    suppress,
                    self.settings.text_top_p,
                    None,
                ).item()
                text_history.append(token)
                raw_token_ids.append(token)
                text_token_ids.append(token)
                last_token = token
                if t_first_tok is None and token not in (
                    self.cfg.mixed_start_token_id,
                    self.cfg.audio_start_token_id,
                ):
                    t_first_tok = time.perf_counter()

                if token == self.cfg.text_end_token_id:
                    if interleaving:
                        text_done = speaking = True
                    else:
                        finished = True
                elif token in eos_id_set:
                    finished = True
                elif interleaving:
                    pass
                elif token == self.cfg.audio_start_token_id:
                    mode = AUDIO
                elif token == self.cfg.mixed_start_token_id:
                    mode = INTERLEAVED
                    slots_left = int(self.cfg.interleaved_n_text)
                    speaking = text_done = False

                if not mode_decided:
                    if token == self.cfg.mixed_start_token_id:
                        mode_decided = True
                        text_streaming_allowed = True
                    elif token == self.cfg.audio_start_token_id:
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
                            mode_decided = True
                            text_streaming_allowed = True
                maybe_emit_text(force=False)
                embed = self.backbone.embed_tokens(mx.array([token], dtype=mx.int32))

            if t_first_slot is None:
                t_first_slot = time.perf_counter()
            slot_count += 1

            if interleaving and not text_done:
                slots_left -= 1
                if slots_left == 0:
                    speaking = not speaking
                    slots_left = (
                        int(self.cfg.interleaved_n_audio)
                        if speaking
                        else int(self.cfg.interleaved_n_text)
                    )

            if finished or step == self.settings.max_slots - 1:
                last_unfed_embed = embed
                break

            hidden = self.backbone(None, self.conv_cache, embed.reshape(1, 1, -1))[:, -1]

        if cancel_event.is_set():
            p_turn = build_partial_turn()
            return TurnOutcome(
                kind="cancelled",
                cancelled=True,
                turn_record=p_turn,
                text=p_turn.text if p_turn else "",
                audio_frames=total_non_eos_frames,
            )

        if not flusher.decoder.finished:
            flusher.flush(finalize=True)

        raw_full_text = self.tokenizer.decode(text_token_ids, skip_special_tokens=True)
        tool_calls = parse_tool_calls(raw_full_text) if use_tools else []
        if tool_calls:
            # Restore cache to pre-turn snapshot so caller executes tools and calls append_tool_exchange cleanly.
            if self.pre_turn_snapshot is not None:
                restore_lfm3_cache(self.conv_cache, self.pre_turn_snapshot)
                self.pre_turn_snapshot = None
            prefill_ms = ((t_first_slot - t_start) * 1000.0) if t_first_slot else 0.0
            ttft_ms = ((t_first_tok - t_start) * 1000.0) if t_first_tok else prefill_ms
            return TurnOutcome(
                kind="tool_calls",
                tool_calls=tool_calls,
                text=raw_full_text,
                prefill_ms=prefill_ms,
                ttft_ms=ttft_ms,
            )

        clean_text = clean_display_text(raw_full_text)
        if not emitted_text and clean_text:
            text_streaming_allowed = True
            maybe_emit_text(force=True)

        # Silent-text guardrail: if tools were enabled (`generation_mode=None`) and model
        # answered in plain text without speech tokens, synthesize TTS audio and replace cache turn.
        used_tts_fallback = False
        if total_non_eos_frames == 0 and clean_text and not cancel_event.is_set():
            def on_tts_event(ev: dict[str, Any]) -> None:
                if ev.get("type") == "audio_chunk":
                    flusher.chunks_sent += 1
                    if flusher.first_chunk_samples == 0:
                        flusher.first_chunk_samples = int(ev.get("samples") or 0)
                    if flusher.t_first_audio is None:
                        flusher.t_first_audio = time.perf_counter()
                    if "codec_ms" in ev:
                        flusher.codec_ms_list.append(float(ev["codec_ms"]))
                on_event(ev)

            tts_codes, _tts_ids = self._synthesize_tts_streaming_sync(
                text=clean_text,
                persona=persona,
                cancel_event=cancel_event,
                on_event=on_tts_event,
                segment_name=segment_name,
            )
            if cancel_event.is_set():
                p_turn = build_partial_turn()
                return TurnOutcome(
                    kind="cancelled",
                    cancelled=True,
                    turn_record=p_turn,
                    text=clean_text,
                    audio_frames=0,
                )
            if tts_codes is not None and tts_codes.shape[0] > 0:
                used_tts_fallback = True
                total_non_eos_frames = int(
                    np.sum(~np.any(tts_codes == self.cfg.audio_eos_code, axis=-1))
                )
                if flusher.t_first_audio is None:
                    flusher.t_first_audio = time.perf_counter()

        if used_tts_fallback:
            # Keep generated text tokens in conv_cache rather than blinding it with audio-only tokens.
            turn_record = TurnRecord(
                role="assistant",
                text=clean_text,
                raw_token_ids=None,
                raw_audio_codes=None,
            )
            need_eos_frame = False
        else:
            need_eos_frame = bool(
                raw_audio_codes and not np.any(raw_audio_codes[-1] == self.cfg.audio_eos_code)
            )
            if need_eos_frame:
                raw_audio_codes.append(
                    np.full((12,), self.cfg.audio_eos_code, dtype=np.int32)
                )
                raw_token_ids.append(int(self.cfg.audio_out_token_id))
            if not raw_token_ids or raw_token_ids[-1] != self._im_end_id:
                raw_token_ids.append(self._im_end_id)

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

        # Always advance conv_cache cleanly right past `<|im_end|>\n`.
        tail_embeds: list[mx.array] = []
        if last_unfed_embed is not None:
            tail_embeds.append(last_unfed_embed.reshape(1, 1, -1))
        if need_eos_frame:
            eos_codes = mx.full((12,), self.cfg.audio_eos_code, dtype=mx.int32)
            tail_embeds.append(
                embed_frames(self.model.codebook_embedding, eos_codes).reshape(1, 1, -1)
            )
        closing_ids: list[int] = []
        if last_token != self._im_end_id:
            closing_ids.append(self._im_end_id)
        closing_ids.extend(self._newline_ids)
        if closing_ids:
            tail_embeds.append(
                self.backbone.embed_tokens(mx.array([closing_ids], dtype=mx.int32))
            )
        if tail_embeds:
            self._prefill_embeds(mx.concatenate(tail_embeds, axis=1))

        self.history.append(turn_record)
        eval_cache(self.conv_cache)

        t_end = time.perf_counter()
        prefill_ms = ((t_first_slot - t_start) * 1000.0) if t_first_slot else 0.0
        ttft_ms = ((t_first_tok - t_start) * 1000.0) if t_first_tok else prefill_ms
        ttfa_ms = (
            ((flusher.t_first_audio - t_start) * 1000.0)
            if flusher.t_first_audio
            else prefill_ms
        )
        gen_time = max(1e-4, t_end - (t_first_slot or t_start))
        slots_per_sec = slot_count / gen_time if slot_count else 0.0
        avg_codec_ms = (
            sum(flusher.codec_ms_list) / len(flusher.codec_ms_list)
            if flusher.codec_ms_list
            else 0.0
        )

        return TurnOutcome(
            kind="response",
            turn_record=turn_record,
            text=clean_text,
            audio_frames=total_non_eos_frames,
            prefill_ms=prefill_ms,
            ttfa_ms=ttfa_ms,
            ttft_ms=ttft_ms,
            slots_per_sec=slots_per_sec,
            audio_chunks_sent=flusher.chunks_sent,
            first_chunk_samples=flusher.first_chunk_samples,
            codec_chunk_avg_ms=avg_codec_ms,
            gen_wall_ms=(t_end - t_start) * 1000.0,
        )

    def _synthesize_tts_streaming_sync(
        self,
        text: str,
        persona: PersonaConfig,
        cancel_event: threading.Event,
        on_event: Callable[[dict[str, Any]], None],
        segment_name: str = "answer",
    ) -> tuple[Optional[np.ndarray], list[int]]:
        """Fallback TTS synthesis via LFM3-Omni when `generation_mode=None` emitted plain text."""
        tts_accent = get_language_profile(self.language).tts_accent
        tts_prompt = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": f"Perform TTS in {tts_accent} using a {persona.voice_gender} voice.",
                },
                {"role": "user", "content": text},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = mx.array([self._encode_text(tts_prompt)], dtype=mx.int32)

        flusher = StreamingAudioFlusher(
            self.codec,
            first_chunk_frames=self.settings.first_chunk_frames,
            stream_chunk_frames=self.settings.stream_chunk_frames,
            on_event=on_event,
            segment_name=segment_name,
        )
        all_codes: list[np.ndarray] = []
        out_token_ids: list[int] = []

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
                if int(c_np[0]) == self.cfg.audio_eos_code:
                    flusher.flush(finalize=True)
                else:
                    flusher.push_frame(c_np)

        if not flusher.decoder.finished:
            flusher.flush(finalize=True)

        if not all_codes:
            return None, []
        return np.stack(all_codes, axis=0).astype(np.int32), out_token_ids

    def maybe_compact_context(self) -> bool:
        """Fold older turns to text once `conv_cache` exceeds `compaction_trigger_tokens`."""
        if (
            self.conv_cache is None
            or self.user_speaking
            or self.user_turn_open
            or self.conv_tokens < self.settings.compaction_trigger_tokens
        ):
            return False

        user_audio_idxs = [
            i
            for i, t in enumerate(self.history)
            if t.role == "user" and t.user_audio_embeds is not None
        ]
        keep_user_set = set(user_audio_idxs[-self.settings.keep_recent_user_audio_turns :])

        asst_audio_idxs = [
            i
            for i, t in enumerate(self.history)
            if t.role == "assistant"
            and t.raw_token_ids
            and t.raw_audio_codes is not None
        ]
        keep_asst_set = set(
            asst_audio_idxs[-self.settings.keep_recent_assistant_audio_turns :]
        )

        # Free heavy tensors on older turns beyond the recent audio window.
        for i, turn in enumerate(self.history):
            if turn.role == "user" and i not in keep_user_set:
                turn.user_audio_embeds = None
                turn.user_audio = None
            elif turn.role == "assistant" and i not in keep_asst_set:
                turn.raw_token_ids = None
                turn.raw_audio_codes = None

        self._rebuild_cache_from_history()
        self.compaction_count += 1
        return True

    def _rebuild_cache_from_history(self) -> None:
        """Re-encode compacted `self.history` into a fresh O(1)-bounded `conv_cache`."""
        new_cache = self.model.make_cache()
        chunks: list[mx.array] = []

        sys_ids = self._render_system_ids(self.system_prompt, self.tools)
        chunks.append(self.backbone.embed_tokens(mx.array([sys_ids], dtype=mx.int32)))

        for turn in self.history:
            if turn.role == "user":
                if turn.user_audio_embeds is not None and turn.user_audio_embeds.shape[0] > 0:
                    p_ids = self._encode_text("<|im_start|>user\n<|vision_start|>")
                    s_ids = self._encode_text("<|vision_end|><|im_end|>\n")
                    chunks.append(
                        self.backbone.embed_tokens(mx.array([p_ids], dtype=mx.int32))
                    )
                    chunks.append(turn.user_audio_embeds[None])
                    chunks.append(
                        self.backbone.embed_tokens(mx.array([s_ids], dtype=mx.int32))
                    )
                else:
                    ids = self._encode_text(
                        f"<|im_start|>user\n{turn.text.strip() or '(spoken message)'}<|im_end|>\n"
                    )
                    chunks.append(
                        self.backbone.embed_tokens(mx.array([ids], dtype=mx.int32))
                    )
            elif turn.role == "assistant":
                if turn.tool_calls:
                    rendered = render_tool_calls(turn.tool_calls)
                    ids = self._encode_text(
                        f"<|im_start|>assistant\n<think></think>{rendered}<|im_end|>\n"
                    )
                    chunks.append(
                        self.backbone.embed_tokens(mx.array([ids], dtype=mx.int32))
                    )
                else:
                    chunks.append(self._assistant_turn_embeds(turn))
            elif turn.role == "tool":
                ids = self._encode_text(
                    f"<|im_start|>tool\n<tool_response>\n{turn.text}\n</tool_response><|im_end|>\n"
                )
                chunks.append(
                    self.backbone.embed_tokens(mx.array([ids], dtype=mx.int32))
                )

        full_embeds = mx.concatenate(chunks, axis=1)
        self.backbone(None, new_cache, full_embeds)
        eval_cache(new_cache)
        self.conv_cache = new_cache
        self.pre_turn_snapshot = None
        mx.clear_cache()
