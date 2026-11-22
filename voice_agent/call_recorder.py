"""Per-call persistence for traces, metrics, transcripts, CRM/delegation logs, and WAV audio."""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import threading
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO, Optional

CALL_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_SECRET_KEY_TOKENS = ("token", "secret", "password", "authorization", "api_key", "cookie")


def generate_call_id() -> str:
    """Return a chronologically sortable UTC call identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"call-{stamp}-{uuid.uuid4().hex[:8]}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_call_id(call_id: str) -> str:
    """Validate that `call_id` is a safe single path segment."""
    if (
        not isinstance(call_id, str)
        or not call_id
        or call_id in (".", "..")
        or not CALL_ID_PATTERN.fullmatch(call_id)
    ):
        raise ValueError("Invalid call_id")
    return call_id


def _redact_secrets(obj: Any) -> Any:
    """Recursively strip any accidental secret-bearing keys before persistence."""
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if any(tok in kl for tok in _SECRET_KEY_TOKENS) and kl != "live_token_present":
                continue
            cleaned[k] = _redact_secrets(v)
        return cleaned
    if isinstance(obj, list):
        return [_redact_secrets(x) for x in obj]
    return obj


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp_path, path)


def _atomic_write_json(path: Path, data: Any) -> None:
    payload = json.dumps(_redact_secrets(data), indent=2, ensure_ascii=False) + "\n"
    _atomic_write_text(path, payload)


class CallRecorder:
    """Thread-safe incremental recorder writing all artifacts for one call into `<base_dir>/<call_id>/`."""

    def __init__(
        self,
        call_id: str,
        base_dir: str | Path,
        initial_metadata: dict[str, Any],
        record_audio: bool = True,
    ) -> None:
        self.call_id = validate_call_id(call_id)
        base_path = Path(base_dir).resolve()
        call_dir = (base_path / self.call_id).resolve()
        if call_dir.parent != base_path:
            raise ValueError("Invalid call_id path")

        self.base_dir = base_path
        self.call_dir = call_dir
        self.record_audio = bool(record_audio)

        self._lock = threading.Lock()
        self._start_monotonic = time.monotonic()
        self._started_at_utc = _utc_now_iso()
        self._trace_seq = 0
        self._finalized = False
        self._warned_io_error = False

        self.call_dir.mkdir(parents=True, exist_ok=True)
        self._trace_path = self.call_dir / "trace.jsonl"
        self._trace_fp: Optional[IO[str]] = open(
            self._trace_path, "a", encoding="utf-8", buffering=1
        )

        self._audio_dir = self.call_dir / "audio"
        self._caller_wav: Optional[wave.Wave_write] = None
        self._agent_wav: Optional[wave.Wave_write] = None
        self._caller_audio_bytes = 0
        self._agent_audio_bytes = 0

        self._transcript_entries: list[dict[str, Any]] = []
        self._delegation_log: list[dict[str, Any]] = []
        self._crm_state: dict[str, Any] = {
            "qualification": None,
            "repayment": None,
            "application": None,
            "callback": None,
            "tool_log": [],
        }
        self._metrics_report: dict[str, Any] = {"call_id": self.call_id}

        sanitized_init = _redact_secrets(copy.deepcopy(initial_metadata or {}))
        flametree_init = copy.deepcopy(sanitized_init.get("flametree") or {})
        flametree_init.setdefault("session_id", None)
        flametree_init.setdefault("session_url", "")
        flametree_init.setdefault("last_error", None)

        initial_settings = copy.deepcopy(sanitized_init.get("initial_settings") or {})

        self._metadata: dict[str, Any] = {
            "call_id": self.call_id,
            "status": "active",
            "started_at_utc": self._started_at_utc,
            "ended_at_utc": None,
            "duration_sec": 0.0,
            "end_reason": None,
            "app_title": sanitized_init.get("app_title", ""),
            "company_name": sanitized_init.get("company_name", ""),
            "model_id": sanitized_init.get("model_id", ""),
            "pipeline_mode": sanitized_init.get("pipeline_mode", "pure_s2s_streaming"),
            "batch_parallel_mode": sanitized_init.get(
                "batch_parallel_mode", "heterogeneous_b2"
            ),
            "engine": copy.deepcopy(sanitized_init.get("engine") or {}),
            "initial_settings": initial_settings,
            "current_settings": copy.deepcopy(
                sanitized_init.get("current_settings") or initial_settings
            ),
            "config_changes": [],
            "client_info": copy.deepcopy(sanitized_init.get("client_info") or {}),
            "flametree": flametree_init,
            "audio": {
                "enabled": self.record_audio,
                "caller_file": None,
                "caller_sample_rate": 16000,
                "caller_audio_sec": 0.0,
                "caller_bytes": 0,
                "agent_file": None,
                "agent_sample_rate": 24000,
                "agent_audio_sec": 0.0,
                "agent_bytes": 0,
            },
            "counts": {
                "user_turns": 0,
                "assistant_turns": 0,
                "interrupted_turns": 0,
                "barge_in_count": 0,
                "delegation_count": 0,
                "tool_call_count": 0,
                "compaction_count": 0,
                "listener_resets": 0,
                "error_count": 0,
                "trace_events": 0,
            },
            "final_context_stats": {},
        }

        with self._lock:
            self._write_metadata_locked()
            self._write_transcript_locked()
            self._write_metrics_locked()
            self._write_crm_locked()

    def _warn_once(self, context: str, exc: Exception) -> None:
        if not self._warned_io_error:
            self._warned_io_error = True
            print(
                f"[CallRecorder] I/O warning ({self.call_id}, {context}): {exc}",
                file=sys.stderr,
                flush=True,
            )

    def _offset_ms(self) -> float:
        return round(max(0.0, (time.monotonic() - self._start_monotonic) * 1000.0), 1)

    def _refresh_counts_locked(self) -> None:
        user_turns = sum(1 for e in self._transcript_entries if e.get("role") == "user")
        ast_entries = [e for e in self._transcript_entries if e.get("role") == "assistant"]
        interrupted_turns = sum(1 for e in ast_entries if e.get("interrupted"))

        tool_calls = len(self._crm_state.get("tool_log") or [])
        if tool_calls == 0:
            tool_calls = sum(1 for e in self._transcript_entries if e.get("role") == "tool")

        delegations = len(self._delegation_log)
        if delegations == 0:
            delegations = sum(
                1 for e in self._transcript_entries if e.get("role") == "delegation"
            )

        counts = self._metadata["counts"]
        counts["user_turns"] = user_turns
        counts["assistant_turns"] = len(ast_entries)
        counts["interrupted_turns"] = interrupted_turns
        counts["delegation_count"] = delegations
        counts["tool_call_count"] = tool_calls
        counts["trace_events"] = self._trace_seq
        if "barge_in_count" in self._metrics_report:
            counts["barge_in_count"] = max(
                int(counts.get("barge_in_count") or 0),
                int(self._metrics_report.get("barge_in_count") or 0),
            )

        audio_meta = self._metadata["audio"]
        audio_meta["caller_bytes"] = self._caller_audio_bytes
        audio_meta["caller_audio_sec"] = round(
            self._caller_audio_bytes / (2.0 * 16000.0), 3
        )
        audio_meta["agent_bytes"] = self._agent_audio_bytes
        audio_meta["agent_audio_sec"] = round(
            self._agent_audio_bytes / (2.0 * 24000.0), 3
        )
        self._metadata["duration_sec"] = round(
            max(0.0, time.monotonic() - self._start_monotonic), 3
        )

    def _write_metadata_locked(self) -> None:
        try:
            self._refresh_counts_locked()
            _atomic_write_json(self.call_dir / "metadata.json", self._metadata)
        except Exception as exc:
            self._warn_once("metadata.json", exc)

    def _write_metrics_locked(self) -> None:
        try:
            _atomic_write_json(self.call_dir / "metrics.json", self._metrics_report)
        except Exception as exc:
            self._warn_once("metrics.json", exc)

    def _write_crm_locked(self) -> None:
        try:
            payload = {
                "call_id": self.call_id,
                "crm_state": self._crm_state,
                "delegation_log": self._delegation_log,
            }
            _atomic_write_json(self.call_dir / "crm_and_delegation.json", payload)
        except Exception as exc:
            self._warn_once("crm_and_delegation.json", exc)

    @staticmethod
    def _format_offset(offset_ms: float) -> str:
        total_sec = max(0.0, float(offset_ms) / 1000.0)
        mins = int(total_sec // 60)
        secs = total_sec - (mins * 60)
        return f"{mins:02d}:{secs:04.1f}"

    def _format_transcript_text_locked(self) -> str:
        cur = self._metadata.get("current_settings") or {}
        ft = self._metadata.get("flametree") or {}
        lines = [
            f"Call ID: {self.call_id}",
            f"Started UTC: {self._started_at_utc}",
        ]
        if self._metadata.get("ended_at_utc"):
            lines.append(
                f"Ended UTC: {self._metadata['ended_at_utc']} "
                f"({self._metadata.get('end_reason') or 'ended'}, "
                f"{self._metadata.get('duration_sec', 0.0):.2f}s)"
            )
        lines.append(
            f"Persona: {cur.get('persona_name') or cur.get('persona_id') or '-'} | "
            f"Language: {cur.get('language') or '-'} | "
            f"Delegation: {cur.get('delegation_mode') or '-'}"
        )
        if ft.get("session_id"):
            lines.append(f"Flametree Session: {ft['session_id']}")
        lines.append("-" * 64)

        for entry in self._transcript_entries:
            ts = self._format_offset(entry.get("call_offset_ms") or 0.0)
            role = entry.get("role")
            speaker = entry.get("speaker") or role or "Unknown"
            text = str(entry.get("text") or "").strip()
            flags: list[str] = []
            if entry.get("interrupted") or entry.get("truncated"):
                flags.append("interrupted/truncated")
            suffix = f" [{' · '.join(flags)}]" if flags else ""
            metric_tag = ""
            if role == "assistant":
                m = entry.get("metrics") or {}
                m_parts: list[str] = []
                if m.get("ttft_ms") is not None:
                    m_parts.append(f"TTFT: {round(float(m['ttft_ms']))} ms")
                if m.get("ttfa_server_ms") is not None:
                    m_parts.append(f"TTFA: {round(float(m['ttfa_server_ms']))} ms")
                if m.get("ttfa_perceived_ms") is not None:
                    m_parts.append(f"Perceived: {round(float(m['ttfa_perceived_ms']))} ms")
                if m_parts:
                    metric_tag = f" ({' | '.join(m_parts)})"
            if role == "delegation":
                lines.append(f"[{ts}] [Delegation · {speaker}] {text}")
            elif role == "tool":
                lines.append(f"[{ts}] [Tool · {speaker}] {text}")
            else:
                lines.append(f"[{ts}] {speaker}{metric_tag}: {text}{suffix}")
        return "\n".join(lines) + "\n"

    def _write_transcript_locked(self) -> None:
        try:
            payload = {
                "call_id": self.call_id,
                "started_at_utc": self._started_at_utc,
                "entries": self._transcript_entries,
            }
            _atomic_write_json(self.call_dir / "transcript.json", payload)
            _atomic_write_text(
                self.call_dir / "transcript.txt",
                self._format_transcript_text_locked(),
            )
        except Exception as exc:
            self._warn_once("transcript", exc)

    def record_event(self, event: str, data: Optional[dict[str, Any]] = None) -> None:
        """Append one timestamped event line to `trace.jsonl`."""
        with self._lock:
            if self._trace_fp is None:
                return
            try:
                self._trace_seq += 1
                row = {
                    "seq": self._trace_seq,
                    "ts_utc": _utc_now_iso(),
                    "call_offset_ms": self._offset_ms(),
                    "event": str(event),
                    "data": _redact_secrets(data or {}),
                }
                self._trace_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._trace_fp.flush()
                if event in ("barge_in_detected", "barge_in_post_gen", "barge_in_mid_gen"):
                    self._metadata["counts"]["barge_in_count"] = (
                        int(self._metadata["counts"].get("barge_in_count") or 0) + 1
                    )
            except Exception as exc:
                self._warn_once("trace.jsonl", exc)

    def record_caller_audio(self, pcm16_16k: bytes) -> None:
        """Append raw 16 kHz mono Int16 PCM caller audio to `audio/caller_16k.wav`."""
        if not self.record_audio or not pcm16_16k:
            return
        with self._lock:
            if self._finalized:
                return
            try:
                if self._caller_wav is None:
                    self._audio_dir.mkdir(parents=True, exist_ok=True)
                    wav_path = self._audio_dir / "caller_16k.wav"
                    writer = wave.open(str(wav_path), "wb")
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(16000)
                    self._caller_wav = writer
                    self._metadata["audio"]["caller_file"] = "audio/caller_16k.wav"
                self._caller_wav.writeframes(pcm16_16k)
                self._caller_audio_bytes += len(pcm16_16k)
            except Exception as exc:
                self._warn_once("caller_16k.wav", exc)

    def record_agent_audio(self, pcm16_24k: bytes) -> None:
        """Append streamed 24 kHz mono Int16 PCM agent audio to `audio/agent_24k.wav`."""
        if not self.record_audio or not pcm16_24k:
            return
        with self._lock:
            if self._finalized:
                return
            try:
                if self._agent_wav is None:
                    self._audio_dir.mkdir(parents=True, exist_ok=True)
                    wav_path = self._audio_dir / "agent_24k.wav"
                    writer = wave.open(str(wav_path), "wb")
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(24000)
                    self._agent_wav = writer
                    self._metadata["audio"]["agent_file"] = "audio/agent_24k.wav"
                self._agent_wav.writeframes(pcm16_24k)
                self._agent_audio_bytes += len(pcm16_24k)
            except Exception as exc:
                self._warn_once("agent_24k.wav", exc)

    def record_transcript_entry(
        self,
        turn_id: str,
        role: str,
        speaker: str,
        text: str,
        *,
        language: Optional[str] = None,
        route_decision: Optional[str] = None,
        filler_text: Optional[str] = None,
        truncated: bool = False,
        interrupted: bool = False,
        extra: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, Any]] = None,
        upsert_by_turn_id: bool = True,
    ) -> None:
        """Record or update one dialogue/tool/delegation entry in `transcript.json` and `transcript.txt`."""
        with self._lock:
            now_utc = _utc_now_iso()
            offset_ms = self._offset_ms()
            sanitized_extra = _redact_secrets(copy.deepcopy(extra)) if extra else {}
            sanitized_metrics = _redact_secrets(copy.deepcopy(metrics)) if metrics else {}

            target_idx: Optional[int] = None
            if upsert_by_turn_id and turn_id:
                for idx in range(len(self._transcript_entries) - 1, -1, -1):
                    existing = self._transcript_entries[idx]
                    if existing.get("turn_id") == turn_id and existing.get("role") == role:
                        target_idx = idx
                        break

            if target_idx is not None:
                entry = self._transcript_entries[target_idx]
                entry["speaker"] = speaker or entry.get("speaker")
                entry["text"] = text
                entry["updated_at_utc"] = now_utc
                if language is not None:
                    entry["language"] = language
                if route_decision is not None:
                    entry["route_decision"] = route_decision
                if filler_text is not None:
                    entry["filler_text"] = filler_text
                entry["truncated"] = bool(truncated)
                entry["interrupted"] = bool(interrupted)
                if sanitized_extra:
                    merged_extra = dict(entry.get("extra") or {})
                    merged_extra.update(sanitized_extra)
                    entry["extra"] = merged_extra
                if sanitized_metrics:
                    merged_metrics = dict(entry.get("metrics") or {})
                    for mk, mv in sanitized_metrics.items():
                        if mv is not None:
                            merged_metrics[mk] = mv
                    entry["metrics"] = merged_metrics
            else:
                entry = {
                    "turn_id": turn_id,
                    "role": role,
                    "speaker": speaker,
                    "text": text,
                    "ts_utc": now_utc,
                    "call_offset_ms": offset_ms,
                    "language": language,
                    "route_decision": route_decision,
                    "filler_text": filler_text,
                    "truncated": bool(truncated),
                    "interrupted": bool(interrupted),
                    "extra": sanitized_extra,
                    "metrics": {
                        mk: mv for mk, mv in sanitized_metrics.items() if mv is not None
                    },
                }
                self._transcript_entries.append(entry)

            self._write_transcript_locked()
            self._write_metadata_locked()

    def update_transcript_metrics(
        self,
        turn_id: str,
        metrics: dict[str, Any],
        role: str = "assistant",
    ) -> None:
        """Merge late-arriving metrics (e.g. perceived TTFA) into an existing transcript entry."""
        if not turn_id or not metrics:
            return
        with self._lock:
            sanitized = _redact_secrets(copy.deepcopy(metrics))
            for idx in range(len(self._transcript_entries) - 1, -1, -1):
                entry = self._transcript_entries[idx]
                if entry.get("turn_id") == turn_id and entry.get("role") == role:
                    merged = dict(entry.get("metrics") or {})
                    changed = False
                    for mk, mv in sanitized.items():
                        if mv is not None and merged.get(mk) != mv:
                            merged[mk] = mv
                            changed = True
                    if changed:
                        entry["metrics"] = merged
                        entry["updated_at_utc"] = _utc_now_iso()
                        self._write_transcript_locked()
                    return

    def record_metrics(self, metrics_report: dict[str, Any]) -> None:
        """Persist current `metrics.json` report and sync call counters in `metadata.json`."""
        with self._lock:
            self._metrics_report = {
                "call_id": self.call_id,
                **copy.deepcopy(metrics_report or {}),
            }
            self._write_metrics_locked()
            self._write_metadata_locked()

    def record_crm_and_delegation(
        self,
        crm_state: dict[str, Any],
        delegation_log: list[dict[str, Any]],
    ) -> None:
        """Persist `crm_and_delegation.json` and update call counts."""
        with self._lock:
            self._crm_state = _redact_secrets(copy.deepcopy(crm_state or {}))
            self._delegation_log = _redact_secrets(copy.deepcopy(delegation_log or []))
            self._write_crm_locked()
            self._write_metadata_locked()

    def record_config_change(
        self,
        changes: dict[str, Any],
        current_settings: dict[str, Any],
        rebuild_needed: bool,
    ) -> None:
        """Record a mid-call configuration change in `metadata.json` and `trace.jsonl`."""
        with self._lock:
            change_entry = {
                "ts_utc": _utc_now_iso(),
                "call_offset_ms": self._offset_ms(),
                "changes": _redact_secrets(copy.deepcopy(changes)),
                "rebuild_needed": bool(rebuild_needed),
            }
            self._metadata["config_changes"].append(change_entry)
            self._metadata["current_settings"] = _redact_secrets(
                copy.deepcopy(current_settings)
            )
            self._write_metadata_locked()
        self.record_event(
            "config_changed",
            {
                "changes": changes,
                "current_settings": current_settings,
                "rebuild_needed": bool(rebuild_needed),
            },
        )

    def update_flametree_status(
        self,
        live: bool,
        session_id: Optional[str],
        session_url: str,
        last_error: Optional[str],
    ) -> None:
        """Update Flametree session metadata while latching non-empty session_id/session_url."""
        with self._lock:
            ft = self._metadata["flametree"]
            ft["live"] = bool(live)
            if session_id:
                ft["session_id"] = session_id
            if session_url:
                ft["session_url"] = session_url
            if last_error is not None:
                ft["last_error"] = last_error
            elif session_id:
                ft["last_error"] = None
            self._write_metadata_locked()
            self._write_transcript_locked()

    def record_context_stats(self, stats: dict[str, Any]) -> None:
        """Update latest context and listener stats in `metadata.json`."""
        with self._lock:
            cleaned = _redact_secrets(copy.deepcopy(stats or {}))
            self._metadata["final_context_stats"] = cleaned
            counts = self._metadata["counts"]
            if "compactions" in cleaned:
                counts["compaction_count"] = int(cleaned["compactions"] or 0)
            if "listener_resets" in cleaned:
                counts["listener_resets"] = int(cleaned["listener_resets"] or 0)
            self._write_metadata_locked()

    def record_error(self, source: str, message: str) -> None:
        """Increment error counter and append an `error` trace event."""
        with self._lock:
            self._metadata["counts"]["error_count"] = (
                int(self._metadata["counts"].get("error_count") or 0) + 1
            )
            self._write_metadata_locked()
        self.record_event("error", {"source": source, "message": str(message)})

    def _close_wavs_locked(self) -> None:
        for attr in ("_caller_wav", "_agent_wav"):
            writer = getattr(self, attr)
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:
                    self._warn_once(f"close_{attr}", exc)
                setattr(self, attr, None)

    def finalize(
        self,
        end_reason: str,
        metrics_report: Optional[dict[str, Any]] = None,
        crm_state: Optional[dict[str, Any]] = None,
        delegation_log: Optional[list[dict[str, Any]]] = None,
        context_stats: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Close WAV/trace handles and atomically finalize all per-call artifacts."""
        with self._lock:
            if self._finalized:
                return copy.deepcopy(self._metadata)

            if metrics_report is not None:
                self._metrics_report = {
                    "call_id": self.call_id,
                    **copy.deepcopy(metrics_report),
                }
            if crm_state is not None:
                self._crm_state = _redact_secrets(copy.deepcopy(crm_state))
            if delegation_log is not None:
                self._delegation_log = _redact_secrets(copy.deepcopy(delegation_log))
            if context_stats:
                cleaned_ctx = _redact_secrets(copy.deepcopy(context_stats))
                self._metadata["final_context_stats"] = cleaned_ctx
                if "compactions" in cleaned_ctx:
                    self._metadata["counts"]["compaction_count"] = int(
                        cleaned_ctx["compactions"] or 0
                    )
                if "listener_resets" in cleaned_ctx:
                    self._metadata["counts"]["listener_resets"] = int(
                        cleaned_ctx["listener_resets"] or 0
                    )

            self._close_wavs_locked()

            ended_at_utc = _utc_now_iso()
            duration_sec = round(max(0.0, time.monotonic() - self._start_monotonic), 3)
            self._metadata["status"] = "completed"
            self._metadata["ended_at_utc"] = ended_at_utc
            self._metadata["duration_sec"] = duration_sec
            self._metadata["end_reason"] = end_reason

            # Append final `call_ended` event before closing trace.jsonl.
            if self._trace_fp is not None:
                try:
                    self._trace_seq += 1
                    row = {
                        "seq": self._trace_seq,
                        "ts_utc": ended_at_utc,
                        "call_offset_ms": round(duration_sec * 1000.0, 1),
                        "event": "call_ended",
                        "data": {
                            "end_reason": end_reason,
                            "duration_sec": duration_sec,
                            "caller_audio_sec": round(
                                self._caller_audio_bytes / (2.0 * 16000.0), 3
                            ),
                            "agent_audio_sec": round(
                                self._agent_audio_bytes / (2.0 * 24000.0), 3
                            ),
                        },
                    }
                    self._trace_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                    self._trace_fp.flush()
                    self._trace_fp.close()
                except Exception as exc:
                    self._warn_once("finalize_trace", exc)
                finally:
                    self._trace_fp = None

            self._write_metadata_locked()
            self._write_metrics_locked()
            self._write_transcript_locked()
            self._write_crm_locked()
            self._finalized = True
            return copy.deepcopy(self._metadata)


def _read_json_if_exists(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def list_recorded_calls(base_dir: str | Path, limit: int = 50) -> list[dict[str, Any]]:
    """List recorded call summaries under `base_dir` ordered newest-first."""
    base_path = Path(base_dir).resolve()
    if not base_path.is_dir():
        return []
    capped = max(1, min(int(limit), 200))
    entries: list[dict[str, Any]] = []
    for child in base_path.iterdir():
        if not child.is_dir():
            continue
        meta = _read_json_if_exists(child / "metadata.json")
        if meta is None:
            continue
        entries.append(
            {
                "call_id": meta.get("call_id", child.name),
                "status": meta.get("status", "unknown"),
                "started_at_utc": meta.get("started_at_utc"),
                "ended_at_utc": meta.get("ended_at_utc"),
                "duration_sec": meta.get("duration_sec", 0.0),
                "end_reason": meta.get("end_reason"),
                "current_settings": meta.get("current_settings") or {},
                "flametree_session_id": (meta.get("flametree") or {}).get("session_id"),
                "counts": meta.get("counts") or {},
                "audio": meta.get("audio") or {},
                "call_dir": str(child),
            }
        )
    entries.sort(
        key=lambda item: (
            str(item.get("started_at_utc") or ""),
            str(item.get("call_id") or ""),
        ),
        reverse=True,
    )
    return entries[:capped]


def load_recorded_call(
    base_dir: str | Path,
    call_id: str,
    include_trace: bool = True,
    trace_limit: int = 1000,
) -> Optional[dict[str, Any]]:
    """Load persisted artifacts for one validated `call_id`, or return None if absent."""
    validated = validate_call_id(call_id)
    base_path = Path(base_dir).resolve()
    call_dir = (base_path / validated).resolve()
    if call_dir.parent != base_path:
        raise ValueError("Invalid call_id")
    if not call_dir.is_dir():
        return None
    metadata = _read_json_if_exists(call_dir / "metadata.json")
    if metadata is None:
        return None

    metrics = _read_json_if_exists(call_dir / "metrics.json")
    transcript = _read_json_if_exists(call_dir / "transcript.json")
    crm_and_delegation = _read_json_if_exists(call_dir / "crm_and_delegation.json")

    trace_path = call_dir / "trace.jsonl"
    trace_count = 0
    trace_events: list[dict[str, Any]] = []
    max_trace = max(0, int(trace_limit))
    if trace_path.is_file():
        with open(trace_path, "r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                trace_count += 1
                if include_trace and len(trace_events) < max_trace:
                    try:
                        trace_events.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass

    bundle: dict[str, Any] = {
        "call_id": metadata.get("call_id", validated),
        "call_dir": str(call_dir),
        "metadata": metadata,
        "metrics": metrics,
        "transcript": transcript,
        "crm_and_delegation": crm_and_delegation,
        "trace_count": trace_count,
    }
    if include_trace:
        bundle["trace_events"] = trace_events
    return bundle
