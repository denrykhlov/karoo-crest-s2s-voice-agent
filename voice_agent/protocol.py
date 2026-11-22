"""Pure-Python conversation records, tool parsing, S2S endpointing & WAV helpers (no MLX)."""

from __future__ import annotations

import io
import json
import re
import struct
import wave
from dataclasses import dataclass, field
from typing import Any, Optional

from .languages import get_language_profile

TOOL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
FUNC_RE = re.compile(
    r"^\s*<function=([^>\n]+)>(.*?)</function>\s*$",
    re.DOTALL,
)
PARAM_RE = re.compile(
    r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>",
    re.DOTALL,
)
SPECIAL_TAG_RE = re.compile(r"<think>.*?</think>|<\|[^|>]+\|>", re.DOTALL)

_ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse LFM3 `<tool_call><function=...><parameter=...>...</parameter></function></tool_call>` blocks."""
    calls: list[dict[str, Any]] = []
    for block in TOOL_BLOCK_RE.findall(text):
        match = FUNC_RE.match(block)
        if not match:
            continue
        func_name = match.group(1).strip()
        body = match.group(2)
        if not func_name:
            continue
        arguments: dict[str, Any] = {}
        for key, raw_value in PARAM_RE.findall(body):
            param_key = key.strip()
            val_str = raw_value.strip()
            try:
                arguments[param_key] = json.loads(val_str)
            except json.JSONDecodeError:
                arguments[param_key] = raw_value
        calls.append(
            {
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": arguments,
                },
            }
        )
    return calls


def render_tool_calls(calls: list[dict[str, Any]]) -> str:
    """Render tool calls back into LFM3 chat-template format."""
    blocks: list[str] = []
    for call in calls:
        fn = call.get("function", call)
        name = fn["name"]
        args = fn.get("arguments") or {}
        lines = [f"<tool_call>\n<function={name}>"]
        if isinstance(args, dict):
            for k, v in args.items():
                rendered = v if isinstance(v, str) else json.dumps(v)
                lines.append(f"<parameter={k}>\n{rendered}\n</parameter>")
        lines.append("</function>\n</tool_call>")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def clean_display_text(text: str) -> str:
    """Strip any residual special tags or thinking markers for UI display."""
    cleaned = SPECIAL_TAG_RE.sub("", text or "")
    return cleaned.strip()


@dataclass(slots=True)
class TurnRecord:
    """One turn stored in conversation history for multi-turn voice continuity."""

    role: str  # "user" | "assistant" | "tool"
    text: str = ""
    user_audio: Any = None  # optional 16 kHz mono float32 ndarray
    user_audio_embeds: Any = None  # optional backend-specific embeddings (e.g. MLX [T, H])
    raw_token_ids: Optional[list[int]] = None
    raw_audio_codes: Any = None  # optional [N_audio_out, 12] ndarray incl. EOS
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_name: Optional[str] = None
    user_pcm16: Optional[bytes] = None  # optional 16 kHz mono Int16 PCM for HTTP S2S backends


@dataclass(slots=True)
class TurnOutcome:
    """Result returned by streaming conversation engines."""

    kind: str  # "response" | "tool_calls" | "cancelled"
    turn_record: Optional[TurnRecord] = None
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    audio_frames: int = 0
    prefill_ms: float = 0.0
    ttfa_ms: float = 0.0
    ttft_ms: float = 0.0
    slots_per_sec: float = 0.0
    audio_chunks_sent: int = 0
    first_chunk_samples: int = 0
    codec_chunk_avg_ms: float = 0.0
    gen_wall_ms: float = 0.0
    cancelled: bool = False
    used_tts_fallback: bool = False


@dataclass(slots=True)
class BatchPlanOutcome:
    """Result from parallel B=2 filler/routing/direct execution (MLX or vLLM)."""

    decision: str  # "direct" | "delegate" | "tool_calls" | "cancelled"
    delegation_query: str = ""
    filler_text: str = ""
    filler_turn: Optional[TurnRecord] = None
    filler_audio_frames: int = 0
    direct_outcome: Optional[TurnOutcome] = None
    updated_conv_cache: Optional[list[Any]] = None
    fork_merge_ms: float = 0.0
    route_decision_ms: float = 0.0
    ttft_ms: float = 0.0
    ttfa_filler_ms: float = 0.0
    ttfa_direct_ms: float = 0.0
    codec_chunk_ms_list: list[float] = field(default_factory=list)
    audio_chunks_sent: int = 0
    first_chunk_samples: int = 0
    slots_per_sec: float = 0.0
    batch_steps: int = 0


def has_real_speech(text: str) -> bool:
    """Return True when a LiveASR chunk contains spoken alphanumeric content."""
    return bool(_ALNUM_RE.search(text or ""))


def is_likely_agent_echo(
    heard_text: str,
    recent_agent_text: str,
    lang_code: str = "en",
) -> bool:
    """Suppress short speaker bleed matching what the agent just spoke."""
    if not recent_agent_text:
        return False
    heard_words = _WORD_RE.findall(heard_text.lower())
    if not heard_words or len(heard_words) > 6:
        return False
    agent_words = _WORD_RE.findall(recent_agent_text.lower())
    if not agent_words:
        return False

    exempt = get_language_profile(lang_code).barge_in_exempt_words
    if len(heard_words) == 1:
        w = heard_words[0]
        if w in exempt:
            return False
        return w in agent_words[-8:]

    tail = agent_words[-24:]
    n = len(heard_words)
    for i in range(max(0, len(tail) - n + 1)):
        if tail[i : i + n] == heard_words:
            return True

    overlap = sum(1 for w in heard_words if w in set(tail))
    return (overlap / len(heard_words)) >= 0.85


def is_semantically_complete(transcript: str, lang_code: str = "en") -> bool:
    """Check whether the live S2S transcript looks like a finished conversational turn."""
    stripped = (transcript or "").strip()
    if not stripped:
        return False
    if stripped.endswith(("...", "…", ",", "-", ":", "¿", "¡")):
        return False
    words = _WORD_RE.findall(stripped.lower())
    if not words:
        return False
    cont_words = get_language_profile(lang_code).continuation_words
    if words[-1] in cont_words:
        return False
    return stripped.endswith((".", "?", "!"))


def required_empty_chunks(
    transcript: str,
    patience: str,
    lang_code: str = "en",
) -> int:
    """Number of empty LiveASR blocks to wait before closing the user turn."""
    complete = is_semantically_complete(transcript, lang_code=lang_code)
    mode = (patience or "balanced").lower()
    if mode == "snappy":
        return 1
    if mode == "patient":
        return 2 if complete else 3
    return 1 if complete else 2


def append_transcript_piece(existing: str, piece: str) -> str:
    """Merge an incremental ASR piece onto an existing transcript buffer."""
    if not piece:
        return existing
    if not existing:
        return piece.lstrip()
    if existing.endswith((" ", "\n")) or piece.startswith(
        (" ", ".", ",", "?", "!", "'", ":")
    ):
        return existing + piece
    return existing + " " + piece


def pcm16_to_wav_bytes(pcm16: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw mono 16-bit signed PCM bytes into an in-memory WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


def strip_wav_header_if_present(raw: bytes) -> bytes:
    """Extract raw PCM payload when a chunk begins with a RIFF/WAVE header."""
    if len(raw) >= 44 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        pos = 12
        end = len(raw)
        while pos + 8 <= end:
            chunk_id = raw[pos : pos + 4]
            chunk_size = struct.unpack_from("<I", raw, pos + 4)[0]
            pos += 8
            if chunk_id == b"data":
                return raw[pos:]
            pos += chunk_size + (chunk_size & 1)
        return raw[44:]
    return raw
