"""Pure-Python conversation records, tool parsing, S2S endpointing & WAV helpers (no MLX)."""

from __future__ import annotations

import io
import json
import re
import struct
import wave
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

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


def pcm16_to_float32(pcm16_bytes: bytes) -> np.ndarray:
    """Decode little-endian 16-bit mono PCM bytes into float32 samples in [-1.0, 1.0]."""
    if not pcm16_bytes:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(pcm16_bytes, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Encode mono float32 samples in [-1.0, 1.0] to little-endian signed 16-bit PCM bytes."""
    if audio.size == 0:
        return b""
    clipped = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2", copy=False).tobytes()


def normalize_tool_calls_for_template(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def parse_router_query(raw_router_text: str, fallback_text: str) -> tuple[str, str]:
    """Parse `DIRECT`, `TOOL`, or `DELEGATE: <question>` into `(kind, standalone_question)`."""
    cleaned = clean_display_text(raw_router_text).strip()
    upper = cleaned.upper()
    if "DELEGATE" in upper:
        idx = upper.find("DELEGATE")
        tail = cleaned[idx + len("DELEGATE") :].lstrip(" :-\n\t")
        return "delegate", (tail.strip() or fallback_text.strip() or "(banking question)")
    if upper.startswith("TOOL") or re.search(r"\bTOOL\b", upper):
        return "tool", ""
    return "direct", ""


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
