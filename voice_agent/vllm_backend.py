"""vLLM-Omni (Linux/CUDA) pure-S2S inference backend via OpenAI-compatible HTTP/SSE."""

from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

import httpx

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
    is_semantically_complete,
    parse_tool_calls,
    pcm16_to_wav_bytes,
    render_tool_calls,
    required_empty_chunks,
    strip_wav_header_if_present,
)


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

    max_k = min(len(prev_words), len(norm_new_words), 6)
    overlap_k = 0
    for k in range(max_k, 0, -1):
        if prev_words[-k:] == norm_new_words[:k]:
            overlap_k = k
            break

    if overlap_k >= len(orig_new_words):
        return ""
    return " ".join(orig_new_words[overlap_k:]).strip()


class VllmLiveS2SListener:
    """Pure-S2S always-streaming audio listener backed by vLLM-Omni `/v1/chat/completions`."""

    def __init__(
        self,
        client_factory: Callable[[], httpx.Client],
        model_id: str,
        idle_reset_sec: float = 15.0,
        hard_reset_sec: float = 30.0,
        block_bytes: int = 20480,  # 640 ms @ 16 kHz mono Int16
        overlap_bytes: int = 20480,  # 640 ms left overlap (1.28 s O(1) window)
    ) -> None:
        self._client_factory = client_factory
        self.model_id = model_id
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
            pcm16_block[-self.overlap_bytes :] if self.overlap_bytes > 0 else b""
        )
        wav_bytes = pcm16_to_wav_bytes(window_pcm, sample_rate=16000)
        b64_wav = base64.b64encode(wav_bytes).decode("ascii")
        # Native assistant-turn `Perform live ASR.` emits `<|text_end|>`/`<|im_end|>` on silence;
        # trailing empty text block satisfies `continue_final_message=True` in Jinja templates.
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "Perform live ASR."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{b64_wav}"},
                        },
                        {"type": "text", "text": ""},
                    ],
                },
            ],
            "add_generation_prompt": False,
            "continue_final_message": True,
            "modalities": ["text"],
            "temperature": 0.0,
            "max_tokens": 48,
            "stream": False,
            "sampling_params_list": [
                {
                    "temperature": 0.0,
                    "max_tokens": 48,
                    "extra_args": {"generation_mode": "text"},
                }
            ],
        }
        with self._client_factory() as client:
            resp = client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        raw_text = clean_display_text(str(msg.get("content") or ""))
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
    """Single-interface S2S conversation orchestrator running against `vllm-omni`."""

    def __init__(
        self,
        client_factory: Callable[[], httpx.Client],
        settings: EngineSettings,
    ) -> None:
        self._client_factory = client_factory
        self.settings = settings
        self.listener = VllmLiveS2SListener(
            client_factory=client_factory,
            model_id=settings.model_id,
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
        self.history.append(
            TurnRecord(
                role="assistant",
                text=render_tool_calls(calls),
                tool_calls=calls,
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
        self.listener.reset_window()
        profile = get_language_profile(self.language)
        wav_bytes = pcm16_to_wav_bytes(pcm16_bytes, sample_rate=16000)
        b64_wav = base64.b64encode(wav_bytes).decode("ascii")
        transcript = self._complete_text_chat(
            [
                {
                    "role": "system",
                    "content": f"Perform ASR in {profile.router_language_name}.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{b64_wav}"},
                        }
                    ],
                },
            ],
            max_tokens=256,
            temperature=0.0,
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
            prompt = (
                "Summarize the following Karoo Crest Bank telesales call state concisely "
                "(customer name, income, card tier preferences, questions answered, tool results):\n\n"
                + "\n".join(digest_lines[-16:])
            )
            payload = {
                "model": self.settings.model_id,
                "messages": [
                    {"role": "system", "content": "Produce a compact factual call memory summary."},
                    {"role": "user", "content": prompt},
                ],
                "modalities": ["text"],
                "temperature": 0.0,
                "max_tokens": 220,
                "stream": False,
                "sampling_params_list": [
                    {
                        "temperature": 0.0,
                        "max_tokens": 220,
                        "extra_args": {"generation_mode": "text"},
                    }
                ],
            }
            with self._client_factory() as client:
                resp = client.post("/v1/chat/completions", json=payload)
                resp.raise_for_status()
                msg = ((resp.json().get("choices") or [{}])[0].get("message") or {})
                summary = clean_display_text(str(msg.get("content") or ""))
                if summary:
                    self._rolling_summary = summary
        except Exception:
            self._rolling_summary = " | ".join(digest_lines[-6:])
        self.history = list(kept_tail)
        self.compaction_count += 1
        return True

    def _build_messages(
        self,
        *,
        include_tools: bool = True,
        extra_system_note: str = "",
        force_mixed_prefix: bool = False,
    ) -> tuple[list[dict[str, Any]], Optional[list[dict[str, Any]]]]:
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
            tool_names = ", ".join(t.name for t in self.tools)
            sys_parts.append(
                f"Available CRM tools ({tool_names}): when calling a tool, emit "
                "<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>."
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "\n\n".join(p for p in sys_parts if p)}
        ]

        # Only send raw audio on recent user turns so payloads remain compact on long calls.
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
                    wav_bytes = pcm16_to_wav_bytes(turn.user_pcm16, sample_rate=16000)
                    b64 = base64.b64encode(wav_bytes).decode("ascii")
                    content: Any = [
                        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}}
                    ]
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "user", "content": turn.text})
            elif turn.role == "assistant":
                messages.append({"role": "assistant", "content": turn.text})
            elif turn.role == "tool":
                messages.append({"role": "tool", "content": turn.text})

        if force_mixed_prefix:
            messages.append({"role": "assistant", "content": "<|mixed_start|>"})
        return messages, tool_schemas

    def _complete_text_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "model": self.settings.model_id,
            "messages": messages,
            "modalities": ["text"],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
            "sampling_params_list": [
                {
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                    "extra_args": {"generation_mode": "text"},
                }
            ],
        }
        with self._client_factory() as client:
            resp = client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        return clean_display_text(str(msg.get("content") or ""))

    def _stream_tts_fallback(
        self,
        text: str,
        persona: PersonaConfig,
        cancel_event: threading.Event,
        emit: Callable[[dict[str, Any]], None],
        t_start: float,
    ) -> tuple[int, float, int, int]:
        profile = get_language_profile(self.language)
        instructions = f"Perform TTS in {profile.tts_accent} using a {persona.voice_gender} voice."
        payload = {
            "model": self.settings.model_id,
            "input": text,
            "instructions": instructions,
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
            "extra_params": {
                "audio_temperature": float(self.settings.audio_temperature),
                "audio_top_k": int(self.settings.audio_top_k),
                "audio_repetition_penalty": float(self.settings.audio_repetition_penalty),
            },
        }
        chunks_sent = 0
        first_chunk_samples = 0
        ttfa_ms = 0.0
        total_pcm_bytes = 0
        is_first = True

        with self._client_factory() as client:
            with client.stream("POST", "/v1/audio/speech", json=payload) as resp:
                resp.raise_for_status()
                ctype = (resp.headers.get("content-type") or "").lower()
                if "text/event-stream" in ctype or "ndjson" in ctype or "json" in ctype:
                    for raw_line in resp.iter_lines():
                        if cancel_event.is_set():
                            resp.close()
                            break
                        line = raw_line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if line == "[DONE]":
                            break
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        b64 = obj.get("audio")
                        if b64 is None:
                            choice = (obj.get("choices") or [{}])[0]
                            delta = choice.get("delta") or {}
                            b64 = (delta.get("audio") or {}).get("data") or delta.get("content")
                        if not b64:
                            continue
                        pcm = base64.b64decode(b64)
                        if is_first:
                            pcm = strip_wav_header_if_present(pcm)
                            is_first = False
                        if not pcm:
                            continue
                        if ttfa_ms == 0.0:
                            ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                            first_chunk_samples = len(pcm) // 2
                        chunks_sent += 1
                        total_pcm_bytes += len(pcm)
                        emit({"type": "audio_chunk", "pcm16": pcm, "samples": len(pcm) // 2})
                else:
                    remainder = b""
                    for raw_bytes in resp.iter_bytes(chunk_size=3840):
                        if cancel_event.is_set():
                            resp.close()
                            break
                        if not raw_bytes:
                            continue
                        data_bytes = remainder + raw_bytes
                        if is_first:
                            data_bytes = strip_wav_header_if_present(data_bytes)
                            is_first = False
                        even_len = len(data_bytes) & ~1
                        pcm = data_bytes[:even_len]
                        remainder = data_bytes[even_len:]
                        if not pcm:
                            continue
                        if ttfa_ms == 0.0:
                            ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                            first_chunk_samples = len(pcm) // 2
                        chunks_sent += 1
                        total_pcm_bytes += len(pcm)
                        emit({"type": "audio_chunk", "pcm16": pcm, "samples": len(pcm) // 2})

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
        use_mixed_prefix = bool(force_interleaved and not tools)
        messages, tool_schemas = self._build_messages(
            include_tools=bool(tools),
            extra_system_note=extra_system_note,
            force_mixed_prefix=use_mixed_prefix,
        )
        audio_extra_args: dict[str, Any] = {
            "audio_temperature": float(self.settings.audio_temperature),
            "audio_top_k": int(self.settings.audio_top_k),
            "audio_repetition_penalty": float(self.settings.audio_repetition_penalty),
        }
        payload: dict[str, Any] = {
            "model": self.settings.model_id,
            "messages": messages,
            "modalities": ["text", "audio"],
            "audio": {"format": "pcm"},
            "temperature": float(self.settings.text_temperature),
            "top_p": float(self.settings.text_top_p),
            "max_tokens": int(max_tokens or self.settings.max_slots),
            "stream": True,
            "sampling_params_list": [
                {
                    "temperature": float(self.settings.text_temperature),
                    "top_p": float(self.settings.text_top_p),
                    "max_tokens": int(max_tokens or self.settings.max_slots),
                    "extra_args": audio_extra_args,
                }
            ],
        }
        if tool_schemas:
            payload["tools"] = tool_schemas
        if use_mixed_prefix:
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False

        raw_text_acc = ""
        clean_text_acc = ""
        openai_tool_calls: dict[int, dict[str, Any]] = {}
        ttft_ms = 0.0
        ttfa_ms = 0.0
        chunks_sent = 0
        first_chunk_samples = 0
        total_audio_bytes = 0
        is_first_audio = True
        last_mic_poll = time.perf_counter()

        def check_barge_in() -> None:
            if poll_mic_fn is None:
                return
            evs = poll_mic_fn()
            for ev in evs:
                if ev.get("type") == "barge_in_detected":
                    cancel_event.set()
                    return

        with self._client_factory() as client:
            with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if cancel_event.is_set():
                        resp.close()
                        break
                    now_poll = time.perf_counter()
                    if now_poll - last_mic_poll >= 0.50:
                        last_mic_poll = now_poll
                        check_barge_in()
                        if cancel_event.is_set():
                            resp.close()
                            break

                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue

                    modality = chunk.get("modality")
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}

                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            idx = int(tc.get("index", 0))
                            slot = openai_tool_calls.setdefault(
                                idx, {"type": "function", "function": {"name": "", "arguments": ""}}
                            )
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["function"]["name"] += str(fn["name"])
                            if fn.get("arguments"):
                                slot["function"]["arguments"] += str(fn["arguments"])

                    audio_b64: Optional[str] = None
                    if modality == "audio":
                        audio_b64 = delta.get("content") or (delta.get("audio") or {}).get("data")
                    elif isinstance(delta.get("audio"), dict) and delta["audio"].get("data"):
                        audio_b64 = delta["audio"]["data"]

                    if audio_b64:
                        pcm = base64.b64decode(audio_b64)
                        if is_first_audio:
                            pcm = strip_wav_header_if_present(pcm)
                            is_first_audio = False
                        if pcm:
                            if ttfa_ms == 0.0:
                                ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                                first_chunk_samples = len(pcm) // 2
                            chunks_sent += 1
                            total_audio_bytes += len(pcm)
                            on_event(
                                {
                                    "type": "audio_chunk",
                                    "pcm16": pcm,
                                    "samples": len(pcm) // 2,
                                }
                            )

                    if modality in (None, "text"):
                        piece = delta.get("content")
                        if piece:
                            raw_text_acc += str(piece)
                            if ttft_ms == 0.0:
                                ttft_ms = (time.perf_counter() - t_start) * 1000.0
                            new_clean = clean_display_text(raw_text_acc)
                            if "<tool_call>" not in new_clean and new_clean != clean_text_acc:
                                diff = (
                                    new_clean[len(clean_text_acc) :]
                                    if new_clean.startswith(clean_text_acc)
                                    else new_clean
                                )
                                clean_text_acc = new_clean
                                on_event(
                                    {
                                        "type": "text_delta",
                                        "delta": diff,
                                        "text": clean_text_acc,
                                        "segment": segment,
                                    }
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

        parsed_calls = parse_tool_calls(raw_text_acc)
        if not parsed_calls and openai_tool_calls:
            for idx in sorted(openai_tool_calls):
                fn = openai_tool_calls[idx]["function"]
                raw_args = fn.get("arguments") or "{}"
                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    parsed_args = {}
                parsed_calls.append(
                    {
                        "type": "function",
                        "function": {"name": fn.get("name", ""), "arguments": parsed_args},
                    }
                )

        if parsed_calls:
            return TurnOutcome(
                kind="tool_calls",
                text="",
                tool_calls=parsed_calls,
                ttft_ms=ttft_ms,
                gen_wall_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        final_text = clean_display_text(raw_text_acc) or clean_text_acc
        used_tts_fallback = False
        if final_text and chunks_sent == 0 and not cancel_event.is_set():
            used_tts_fallback = True
            fb_frames, fb_ttfa, fb_chunks, fb_first = self._stream_tts_fallback(
                final_text, persona, cancel_event, on_event, t_start
            )
            chunks_sent += fb_chunks
            total_audio_bytes += fb_frames * 2 * 1920
            if ttfa_ms == 0.0:
                ttfa_ms = fb_ttfa
                first_chunk_samples = fb_first

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
            slots_per_sec=round((len(final_text.split()) + audio_frames) / max(wall_ms / 1000.0, 1e-3), 1),
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
        prompt = (
            f"{ctx_line}"
            f"Customer just said: {user_text or '(audio turn)'}\n\n"
            "Classify into exactly ONE line:\n"
            "- Output `DIRECT` for greetings, short rapport, clarifications, or local CRM calculations/applications.\n"
            f"- Output `DELEGATE: <concise standalone question in {profile.router_language_name}>` "
            "ONLY for detailed product policy, tariffs, FX fees, grace days, NCA/FICA/POPIA rules, or travel perks."
        )
        raw = self._complete_text_chat(
            [
                {"role": "system", "content": "Output strictly `DIRECT` or `DELEGATE: <question>`."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.settings.max_router_tokens,
            temperature=0.0,
        )
        dec, query = _parse_router_query(raw, user_text)
        return dec, query, (time.perf_counter() - t0) * 1000.0

    def _run_query_extract_call(self, user_text: str) -> str:
        profile = get_language_profile(self.language)
        prev_asst = self.latest_assistant_text()
        prompt = (
            (f"Previous advisor turn: {prev_asst}\n" if prev_asst else "")
            + f"Customer message: {user_text or '(audio turn)'}\n"
            + f"Rewrite this as one crisp standalone banking knowledge question in {profile.router_language_name}."
        )
        raw = self._complete_text_chat(
            [
                {"role": "system", "content": "Output ONLY one concise standalone question."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.settings.max_router_tokens,
            temperature=0.0,
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

        with ThreadPoolExecutor(max_workers=2) as pool:
            r_fut = pool.submit(self._run_router_call, user_text)
            d_fut = pool.submit(
                self._stream_omni_chat,
                persona=persona,
                tools=tools,
                force_interleaved=not bool(tools),
                cancel_event=_LinkedEvent(),  # type: ignore[arg-type]
                on_event=direct_on_event,
                poll_mic_fn=poll_mic_fn,
                segment="answer",
            )
            dec, query, route_ms = r_fut.result()

            if cancel_event.is_set():
                spec_cancel.set()
                d_out = d_fut.result()
                return BatchPlanOutcome(
                    decision="cancelled",
                    filler_turn=d_out.turn_record,
                    filler_audio_frames=d_out.audio_frames,
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
                        filler_audio_frames=d_out.audio_frames,
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

            # Router chose DELEGATE -> cancel speculative direct generation & dispatch query now.
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


class VllmOmniEngine:
    """Thread-safe HTTP/SSE client engine targeting a running `vllm serve --omni` server."""

    def __init__(
        self,
        settings: EngineSettings,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.settings = settings
        self._transport = transport
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="vllm-lfm3"
        )
        self._loaded: bool = False
        self._load_lock = threading.Lock()
        self.available_models: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _make_client(self) -> httpx.Client:
        headers: dict[str, str] = {}
        if self.settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.vllm_api_key}"
        return httpx.Client(
            base_url=self.settings.vllm_base_url.rstrip("/"),
            timeout=httpx.Timeout(self.settings.vllm_timeout_sec, connect=10.0),
            headers=headers,
            transport=self._transport,
        )

    def load(self) -> None:
        """Verify reachability of the vLLM-Omni OpenAI API server (`GET /v1/models`)."""
        with self._load_lock:
            if self._loaded:
                return
            t0 = time.perf_counter()
            last_exc: Optional[Exception] = None
            max_attempts = 1 if self._transport is not None else 15
            for attempt in range(max_attempts):
                try:
                    with self._make_client() as client:
                        resp = client.get("/v1/models")
                        resp.raise_for_status()
                        data = resp.json()
                    models = [m.get("id", "") for m in (data.get("data") or []) if m.get("id")]
                    self.available_models = models
                    self._loaded = True
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[VllmOmniEngine] Connected to {self.settings.vllm_base_url} "
                        f"(models={models}) in {elapsed:.2f}s",
                        flush=True,
                    )
                    return
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        time.sleep(1.0)
            raise RuntimeError(
                f"Failed to connect to vLLM-Omni at {self.settings.vllm_base_url}: {last_exc}"
            )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def run_on_worker(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Future[Any]:
        return self._executor.submit(fn, *args, **kwargs)

    def create_pipeline(self) -> VllmStreamingConversationSession:
        if not self._loaded:
            self.load()
        return VllmStreamingConversationSession(
            client_factory=self._make_client,
            settings=self.settings,
        )
