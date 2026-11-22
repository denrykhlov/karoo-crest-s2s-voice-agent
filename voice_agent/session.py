"""Per-call session orchestrator connecting WebSocket audio/control to pure-S2S + B=2 delegation."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Any, Callable, Optional

from fastapi import WebSocket, WebSocketDisconnect

from .call_recorder import CallRecorder, generate_call_id
from .config import PersonaConfig, ToolSpec, VoiceAgentConfig
from .delegation import DelegationResult, KnowledgeDelegate, build_delegate
from .languages import (
    build_localized_greeting_trigger,
    build_localized_system_prompt,
    normalize_language_code,
)
from .metrics import DeliveryStatsTracker, TurnDeliveryMetrics
from .protocol import BatchPlanOutcome, TurnOutcome, TurnRecord


class CallSession:
    """Manages one active browser voice call with pure-S2S streaming & Flametree delegation."""

    def __init__(
        self,
        websocket: WebSocket,
        config: VoiceAgentConfig,
        engine: Any,
    ) -> None:
        self.ws = websocket
        self.config = config
        self.engine = engine
        self.loop = asyncio.get_running_loop()

        client_addr = getattr(websocket, "client", None)
        headers = getattr(websocket, "headers", None)
        self._client_info: dict[str, Any] = {
            "host": getattr(client_addr, "host", None),
            "port": getattr(client_addr, "port", None),
            "user_agent": headers.get("user-agent") if headers else None,
        }

        self.persona: PersonaConfig = config.get_persona(config.default_persona_id)
        self.tools_enabled: bool = True
        self.input_mode: str = "live"  # "live" (always-on S2S) | "ptt" (push-to-talk)
        self.patience: str = config.default_turn_patience
        self.language: str = normalize_language_code(config.default_language)
        self.delegation_mode: str = config.default_delegation_mode
        self.dynamic_filler: bool = bool(config.default_dynamic_filler)

        self.pipeline: Any = None
        self.delegate: KnowledgeDelegate = build_delegate(config.flametree)
        self.stats = DeliveryStatsTracker()
        self.delegation_log: list[dict[str, Any]] = []
        self.recorder: Optional[CallRecorder] = None

        self.call_active: bool = False
        self.muted: bool = False
        self.call_start_ts: float = 0.0
        self._call_id: str = ""
        self.crm_state: dict[str, Any] = self._empty_crm_state()
        self._last_context_stats: dict[str, Any] = {}

        self._pcm_lock = threading.Lock()
        self._pcm_buffer = bytearray()
        self._ptt_active: bool = False
        self._ptt_chunks: list[bytes] = []

        self._cancel_event = threading.Event()
        self._generating: bool = False
        self._drain_in_flight: bool = False
        self._active_task: Optional[asyncio.Task[None]] = None
        self._turn_lock = asyncio.Lock()

        self._agent_speaking: bool = False
        self._played_samples_24k: int = 0
        self._recent_agent_text: str = ""
        self._last_assistant_turn: Optional[TurnRecord] = None
        self._last_assistant_turn_id: Optional[str] = None
        self._ui_user_turns: set[str] = set()
        self._closed: bool = False

        self._turn_counter: int = 0
        self._last_endpoint_ms: Optional[float] = None
        self._turn_start_wall: dict[str, float] = {}

    @staticmethod
    def _empty_crm_state() -> dict[str, Any]:
        return {
            "qualification": None,
            "repayment": None,
            "application": None,
            "callback": None,
            "tool_log": [],
        }

    def _current_settings_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona.id,
            "persona_name": self.persona.name,
            "voice_gender": self.persona.voice_gender,
            "language": self.language,
            "delegation_mode": self.delegation_mode,
            "dynamic_filler": self.dynamic_filler,
            "patience": self.patience,
            "input_mode": self.input_mode,
            "tools_enabled": self.tools_enabled,
        }

    def _finalize_recorder(self, end_reason: str) -> None:
        rec = self.recorder
        if rec is None:
            return
        self.recorder = None
        rec.finalize(
            end_reason=end_reason,
            metrics_report=self.stats.summary(include_all_turns=True),
            crm_state=self.crm_state,
            delegation_log=self.delegation_log,
            context_stats=self._last_context_stats,
        )
        print(
            f"[CallSession] Finalized {rec.call_id} ({end_reason}) -> {rec.call_dir}",
            flush=True,
        )

    def _localized_system_prompt(self) -> str:
        return build_localized_system_prompt(
            base_prompt=self.persona.system_prompt,
            lang_code=self.language,
            persona_name=self.persona.name,
            voice_gender=self.persona.voice_gender,
            delegation_mode=self.delegation_mode,
        )

    def _ensure_pipeline(self) -> Any:
        if self.pipeline is None:
            if not self.engine.is_loaded:
                self.engine.load()
            self.pipeline = self.engine.create_pipeline()
        return self.pipeline

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            await self.ws.send_json(payload)
        except Exception:
            self._closed = True

    async def _send_bytes(self, data: bytes) -> None:
        if self._closed or not data:
            return
        try:
            await self.ws.send_bytes(data)
            if self.recorder is not None and self.call_active:
                self.recorder.record_agent_audio(data)
        except Exception:
            self._closed = True

    def _active_tools(self) -> list[ToolSpec]:
        return list(self.config.tools) if self.tools_enabled else []

    def _tool_map(self) -> dict[str, ToolSpec]:
        return {t.name: t for t in self.config.tools}

    def _take_pcm_buffer(self) -> bytes:
        with self._pcm_lock:
            if not self._pcm_buffer:
                return b""
            data = bytes(self._pcm_buffer)
            self._pcm_buffer.clear()
            return data

    def _clear_pcm_buffer(self) -> None:
        with self._pcm_lock:
            self._pcm_buffer.clear()

    async def _emit_context_stats(self) -> None:
        if self.pipeline is None:
            return
        elapsed = (
            int(time.monotonic() - self.call_start_ts) if self.call_start_ts else 0
        )
        stats_dict = {
            "tokens": self.pipeline.conv_tokens,
            "max_tokens": 32768,
            "compactions": self.pipeline.compaction_count,
            "listener_sec": self.pipeline.listener.window_seconds,
            "listener_resets": self.pipeline.listener.reset_count,
            "call_sec": elapsed,
            "target_sec": self.config.target_call_duration_sec,
        }
        self._last_context_stats = stats_dict
        if self.recorder is not None:
            self.recorder.record_context_stats(stats_dict)
            self.recorder.record_event("context_stats", stats_dict)
        await self._send_json({"type": "context_stats", **stats_dict})

    async def _emit_delegation_status(self) -> None:
        sid = self.delegate.current_session_id
        session_url = (
            f"{self.config.flametree.base_url.rstrip('/')}/sessions/{sid}"
            f"?tenant_id={self.config.flametree.tenant_id}"
            if sid and self.config.flametree.tenant_id
            else ""
        )
        if self.recorder is not None:
            self.recorder.update_flametree_status(
                live=self.delegate.is_live,
                session_id=sid,
                session_url=session_url,
                last_error=self.delegate.last_error,
            )
        await self._send_json(
            {
                "type": "delegation_status",
                "call_id": self._call_id if self.call_active else None,
                "live": self.delegate.is_live,
                "agent_id": self.config.flametree.agent_id,
                "environment": self.config.flametree.environment,
                "tenant_name": self.config.flametree.tenant_name,
                "mode": self.delegation_mode,
                "dynamic_filler": self.dynamic_filler,
                "language": self.language,
                "session_id": sid,
                "session_url": session_url,
                "error": self.delegate.last_error,
            }
        )

    async def _open_delegate_session_bg(self) -> None:
        sid: Optional[str] = None
        err_msg: Optional[str] = None
        try:
            sid = await self.delegate.start_session(self._call_id, self.language)
        except Exception as exc:
            err_msg = str(exc)
        if self.recorder is not None:
            if sid:
                self.recorder.record_event(
                    "flametree_session_opened",
                    {
                        "session_id": sid,
                        "language": self.language,
                        "agent_id": self.config.flametree.agent_id,
                    },
                )
            elif not self.delegate.is_live:
                self.recorder.record_event(
                    "flametree_fallback_mode",
                    {
                        "reason": "no_live_token_or_disabled",
                        "language": self.language,
                    },
                )
            else:
                self.recorder.record_event(
                    "flametree_session_error",
                    {"error": err_msg or self.delegate.last_error or "unknown"},
                )
        if not self._closed:
            await self._emit_delegation_status()

    async def _emit_delivery_stats(self) -> None:
        if self.recorder is not None:
            self.recorder.record_metrics(self.stats.summary(include_all_turns=True))
        await self._send_json(
            {
                "type": "delivery_stats",
                "stats": self.stats.summary(),
            }
        )

    async def run(self) -> None:
        """Main WebSocket receive loop for the browser call session."""
        await self._send_json({"type": "call_state", "state": "idle"})
        await self._emit_delegation_status()
        await self._emit_delivery_stats()
        try:
            while not self._closed:
                message = await self.ws.receive()
                msg_type = message.get("type")
                if msg_type == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await self._on_audio_bytes(message["bytes"])
                elif message.get("text") is not None:
                    await self._on_control_text(message["text"])
        except WebSocketDisconnect:
            pass
        finally:
            self._closed = True
            self._cancel_event.set()
            if self._active_task and not self._active_task.done():
                self._active_task.cancel()
            had_ft_session = bool(self.delegate.current_session_id)
            try:
                await self.delegate.close_session()
            except Exception:
                pass
            if had_ft_session and self.recorder is not None:
                self.recorder.record_event("flametree_session_closed", {})
            self._finalize_recorder("websocket_disconnect")

    async def _on_control_text(self, raw_text: str) -> None:
        try:
            msg = json.loads(raw_text)
        except json.JSONDecodeError:
            return
        kind = msg.get("type")

        if kind == "start_call":
            if "persona_id" in msg:
                self.persona = self.config.get_persona(msg.get("persona_id"))
            if "tools_enabled" in msg:
                self.tools_enabled = bool(msg["tools_enabled"])
            if "turn_patience" in msg and msg["turn_patience"] in (
                "snappy",
                "balanced",
                "patient",
            ):
                self.patience = msg["turn_patience"]
            if "input_mode" in msg and msg["input_mode"] in ("live", "ptt"):
                self.input_mode = msg["input_mode"]
            if "language" in msg:
                self.language = normalize_language_code(msg.get("language"))
            if "delegation_mode" in msg and msg["delegation_mode"] in (
                "selective_tool",
                "always_delegate",
            ):
                self.delegation_mode = msg["delegation_mode"]
            if "dynamic_filler" in msg:
                self.dynamic_filler = bool(msg["dynamic_filler"])
            await self._start_call()

        elif kind == "hang_up":
            await self._hang_up()

        elif kind == "set_config":
            rebuild_needed = False
            changes: dict[str, Any] = {}
            if "persona_id" in msg:
                new_p = self.config.get_persona(msg.get("persona_id"))
                if new_p.id != self.persona.id:
                    self.persona = new_p
                    rebuild_needed = True
                    changes["persona_id"] = new_p.id
            if "tools_enabled" in msg:
                new_t = bool(msg["tools_enabled"])
                if new_t != self.tools_enabled:
                    self.tools_enabled = new_t
                    rebuild_needed = True
                    changes["tools_enabled"] = new_t
            if "language" in msg:
                new_l = normalize_language_code(msg.get("language"))
                if new_l != self.language:
                    self.language = new_l
                    rebuild_needed = True
                    changes["language"] = new_l
            if "delegation_mode" in msg and msg["delegation_mode"] in (
                "selective_tool",
                "always_delegate",
            ):
                if msg["delegation_mode"] != self.delegation_mode:
                    self.delegation_mode = msg["delegation_mode"]
                    rebuild_needed = True
                    changes["delegation_mode"] = self.delegation_mode
            if "dynamic_filler" in msg:
                new_df = bool(msg["dynamic_filler"])
                if new_df != self.dynamic_filler:
                    self.dynamic_filler = new_df
                    changes["dynamic_filler"] = new_df
            if "turn_patience" in msg and msg["turn_patience"] in (
                "snappy",
                "balanced",
                "patient",
            ):
                if msg["turn_patience"] != self.patience:
                    self.patience = msg["turn_patience"]
                    changes["turn_patience"] = self.patience
            if "input_mode" in msg and msg["input_mode"] in ("live", "ptt"):
                if msg["input_mode"] != self.input_mode:
                    self.input_mode = msg["input_mode"]
                    changes["input_mode"] = self.input_mode
                self._clear_pcm_buffer()
                self._ptt_active = False
                self._ptt_chunks.clear()
            if "muted" in msg:
                new_m = bool(msg["muted"])
                if new_m != self.muted:
                    self.muted = new_m
                    changes["muted"] = new_m
                if self.muted:
                    self._clear_pcm_buffer()

            if changes and self.recorder is not None:
                self.recorder.record_config_change(
                    changes, self._current_settings_dict(), rebuild_needed
                )

            await self._emit_delegation_status()
            if rebuild_needed and self.call_active and self.pipeline is not None:
                await asyncio.wrap_future(
                    self.engine.run_on_worker(
                        self.pipeline.update_system_config,
                        self._localized_system_prompt(),
                        self._active_tools() or None,
                        self.language,
                    ),
                    loop=self.loop,
                )

        elif kind == "ptt_start":
            if not self.call_active:
                return
            if self._agent_speaking or self._generating:
                await self._trigger_external_barge_in()
            self._ptt_active = True
            self._ptt_chunks.clear()
            if self.recorder is not None:
                self.recorder.record_event("ptt_start", {"input_mode": self.input_mode})
            await self._send_json({"type": "s2s_listen_state", "speaking": True})

        elif kind == "ptt_end":
            if not self.call_active or not self._ptt_active:
                return
            self._ptt_active = False
            await self._send_json({"type": "s2s_listen_state", "speaking": False})
            raw_pcm = b"".join(self._ptt_chunks)
            self._ptt_chunks.clear()
            if self.recorder is not None:
                self.recorder.record_event(
                    "ptt_end",
                    {
                        "pcm_bytes": len(raw_pcm),
                        "duration_sec": round(len(raw_pcm) / (2.0 * 16000.0), 3),
                    },
                )
            if len(raw_pcm) >= 1600 * 2:
                self._schedule_ptt_turn(raw_pcm)

        elif kind == "user_text":
            text = str(msg.get("text") or "").strip()
            if not text:
                return
            if not self.call_active:
                await self._init_call_state(ringing=False)
            if self._agent_speaking or self._generating:
                await self._trigger_external_barge_in()
            self._schedule_text_turn(text)

        elif kind == "playback_started":
            turn_id = str(msg.get("turn_id") or "")
            t0 = self._turn_start_wall.get(turn_id)
            if t0 is not None:
                perceived_ms = (time.perf_counter() - t0) * 1000.0
                rounded_perc = round(perceived_ms, 1)
                if self.recorder is not None:
                    self.recorder.record_event(
                        "playback_started",
                        {
                            "turn_id": turn_id,
                            "ttfa_perceived_ms": rounded_perc,
                            "buffered_samples": msg.get("buffered_samples"),
                        },
                    )
                    self.recorder.update_transcript_metrics(
                        turn_id, {"ttfa_perceived_ms": rounded_perc}
                    )
                await self._send_json(
                    {
                        "type": "live_ttfa",
                        "turn_id": turn_id,
                        "ttfa_perceived_ms": rounded_perc,
                    }
                )
                if self.stats.update_perceived_ttfa(turn_id, perceived_ms):
                    await self._emit_delivery_stats()

        elif kind == "playback_progress":
            samples = int(msg.get("played_samples") or 0)
            if samples >= self._played_samples_24k:
                self._played_samples_24k = samples

        elif kind == "playback_done":
            samples = int(msg.get("played_samples") or 0)
            if samples >= self._played_samples_24k:
                self._played_samples_24k = samples
            self._agent_speaking = False
            if self.recorder is not None:
                self.recorder.record_event(
                    "playback_done",
                    {
                        "played_samples_24k": self._played_samples_24k,
                        "played_sec": round(self._played_samples_24k / 24000.0, 3),
                    },
                )
            if self.call_active:
                await self._send_json({"type": "call_state", "state": "listening"})
                if self.pipeline is not None and not self._generating:
                    compacted = await asyncio.wrap_future(
                        self.engine.run_on_worker(self.pipeline.maybe_compact_context),
                        loop=self.loop,
                    )
                    if compacted:
                        if self.recorder is not None:
                            self.recorder.record_event(
                                "context_compacted",
                                {
                                    "conv_tokens": self.pipeline.conv_tokens,
                                    "compactions": self.pipeline.compaction_count,
                                },
                            )
                        await self._emit_context_stats()

    async def _on_audio_bytes(self, pcm16_bytes: bytes) -> None:
        if not self.call_active or self.muted:
            return
        if self.input_mode == "ptt":
            if self._ptt_active:
                self._ptt_chunks.append(pcm16_bytes)
                if self.recorder is not None:
                    self.recorder.record_caller_audio(pcm16_bytes)
            return

        if self.recorder is not None:
            self.recorder.record_caller_audio(pcm16_bytes)

        with self._pcm_lock:
            self._pcm_buffer.extend(pcm16_bytes)
            buf_len = len(self._pcm_buffer)

        if (
            not self._generating
            and not self._drain_in_flight
            and buf_len >= 10240
        ):
            self._drain_in_flight = True
            asyncio.create_task(self._drain_live_pcm_loop())

    async def _drain_live_pcm_loop(self) -> None:
        try:
            while (
                self.call_active
                and not self._closed
                and not self._generating
                and self.input_mode == "live"
            ):
                chunk = self._take_pcm_buffer()
                if not chunk:
                    break
                pipeline = self._ensure_pipeline()
                events = await asyncio.wrap_future(
                    self.engine.run_on_worker(
                        pipeline.ingest_pcm_and_listen,
                        chunk,
                        self._agent_speaking,
                        self._recent_agent_text,
                        self.patience,
                    ),
                    loop=self.loop,
                )
                if events:
                    await self._handle_live_events(events)

                with self._pcm_lock:
                    if len(self._pcm_buffer) < 10240:
                        break
        except Exception as exc:
            if self.recorder is not None:
                self.recorder.record_error("s2s_listener", str(exc))
            await self._send_json({"type": "error", "message": f"S2S listener error: {exc}"})
        finally:
            self._drain_in_flight = False

    async def _handle_live_events(self, events: list[dict[str, Any]]) -> None:
        should_respond = False
        for ev in events:
            et = ev.get("type")
            if et == "barge_in_detected":
                await self._handle_post_gen_barge_in()
            elif et == "user_speech_start":
                tid = ev["turn_id"]
                self._ui_user_turns.add(tid)
                if self.recorder is not None:
                    self.recorder.record_event(
                        "user_speech_start", {"turn_id": tid, "text": ev["text"]}
                    )
                await self._send_json({"type": "s2s_listen_state", "speaking": True})
                await self._send_json(
                    {
                        "type": "user_turn",
                        "turn_id": tid,
                        "text": ev["text"],
                        "pending_asr": True,
                    }
                )
            elif et == "user_speech_delta":
                tid = ev["turn_id"]
                if tid in self._ui_user_turns:
                    if self.recorder is not None:
                        self.recorder.record_event(
                            "user_speech_delta", {"turn_id": tid, "text": ev["text"]}
                        )
                    await self._send_json(
                        {
                            "type": "user_transcript_update",
                            "turn_id": tid,
                            "text": ev["text"],
                            "pending_asr": True,
                        }
                    )
            elif et == "user_turn_committed":
                tid = ev["turn_id"]
                self._last_endpoint_ms = ev.get("endpoint_ms")
                if self.recorder is not None:
                    self.recorder.record_event(
                        "user_turn_committed",
                        {
                            "turn_id": tid,
                            "text": ev["text"],
                            "endpoint_ms": ev.get("endpoint_ms"),
                        },
                    )
                    self.recorder.record_transcript_entry(
                        turn_id=tid,
                        role="user",
                        speaker="Caller",
                        text=ev["text"],
                        language=self.language,
                        extra={
                            "input_mode": "live",
                            "endpoint_ms": ev.get("endpoint_ms"),
                        },
                    )
                await self._send_json({"type": "s2s_listen_state", "speaking": False})
                if tid in self._ui_user_turns:
                    await self._send_json(
                        {
                            "type": "user_transcript_update",
                            "turn_id": tid,
                            "text": ev["text"],
                            "pending_asr": False,
                        }
                    )
                else:
                    self._ui_user_turns.add(tid)
                    await self._send_json(
                        {
                            "type": "user_turn",
                            "turn_id": tid,
                            "text": ev["text"],
                            "pending_asr": False,
                        }
                    )
                should_respond = True

        if should_respond and self.call_active and not self._closed:
            await self._emit_context_stats()
            self._schedule_agent_response()

    async def _handle_post_gen_barge_in(self) -> None:
        """Roll back `conv_cache` when user interrupts after generation finished."""
        t0 = time.perf_counter()
        played_frames = self._played_samples_24k // 1920
        prev_turn_id = self._last_assistant_turn_id
        self._agent_speaking = False
        await self._send_json({"type": "barge_in"})

        truncated = None
        if self.pipeline is not None:
            truncated = await asyncio.wrap_future(
                self.engine.run_on_worker(
                    self.pipeline.rollback_interrupted_assistant_turn,
                    played_frames,
                    self._last_assistant_turn,
                    True,
                ),
                loop=self.loop,
            )
            if truncated is not None and prev_turn_id:
                await self._send_json(
                    {
                        "type": "assistant_turn_truncated",
                        "turn_id": prev_turn_id,
                        "text": truncated.text,
                    }
                )
                if self.recorder is not None:
                    self.recorder.record_transcript_entry(
                        turn_id=prev_turn_id,
                        role="assistant",
                        speaker=self.persona.name,
                        text=truncated.text,
                        language=self.language,
                        truncated=True,
                        interrupted=True,
                    )
        self._last_assistant_turn = None
        self._last_assistant_turn_id = None
        cancel_ms = (time.perf_counter() - t0) * 1000.0
        if self.recorder is not None:
            self.recorder.record_event(
                "barge_in_post_gen",
                {
                    "turn_id": prev_turn_id,
                    "played_samples_24k": self._played_samples_24k,
                    "played_frames": played_frames,
                    "cancel_ms": round(cancel_ms, 1),
                    "truncated_text": truncated.text if truncated is not None else None,
                },
            )
        self.stats.mark_barge_in(cancel_ms=cancel_ms)
        await self._emit_delivery_stats()

    async def _trigger_external_barge_in(self) -> None:
        """Cancel active generation/playback for PTT or typed-text interruption."""
        self._cancel_event.set()
        if self._active_task and not self._active_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._active_task), timeout=5.0)
            except Exception:
                self._active_task.cancel()
        self._active_task = None
        if self._agent_speaking:
            await self._handle_post_gen_barge_in()

    async def _init_call_state(self, ringing: bool = True) -> None:
        await self._trigger_external_barge_in()
        if self.recorder is not None:
            self._finalize_recorder("new_call_replaced")
        try:
            await self.delegate.close_session()
        except Exception:
            pass
        self.delegate = build_delegate(self.config.flametree)
        self._call_id = generate_call_id()

        self.call_active = True
        self.muted = False
        self.call_start_ts = time.monotonic()
        self._clear_pcm_buffer()
        self._ptt_active = False
        self._ptt_chunks.clear()
        self._agent_speaking = False
        self._played_samples_24k = 0
        self._recent_agent_text = ""
        self._last_assistant_turn = None
        self._last_assistant_turn_id = None
        self._ui_user_turns.clear()
        self.crm_state = self._empty_crm_state()
        self.stats.reset()
        self.delegation_log.clear()
        self._turn_counter = 0
        self._last_endpoint_ms = None
        self._turn_start_wall.clear()
        self._last_context_stats = {}

        if self.config.calls_dir:
            initial_settings = self._current_settings_dict()
            pub_ft = self.config.flametree.to_public_dict()
            pub_ft["live"] = self.delegate.is_live
            pub_ft["session_id"] = None
            pub_ft["session_url"] = ""
            pub_ft["last_error"] = None
            self.recorder = CallRecorder(
                call_id=self._call_id,
                base_dir=self.config.calls_dir,
                initial_metadata={
                    "app_title": self.config.app_title,
                    "company_name": self.config.company_name,
                    "backend": self.config.engine.backend,
                    "model_id": self.config.engine.model_id,
                    "pipeline_mode": "pure_s2s_streaming",
                    "batch_parallel_mode": (
                        "vllm_continuous_batch"
                        if self.config.engine.backend == "vllm"
                        else "heterogeneous_b2"
                    ),
                    "engine": {
                        "first_chunk_frames": self.config.engine.first_chunk_frames,
                        "stream_chunk_frames": self.config.engine.stream_chunk_frames,
                        "compaction_trigger_tokens": self.config.engine.compaction_trigger_tokens,
                        "max_slots": self.config.engine.max_slots,
                    },
                    "initial_settings": initial_settings,
                    "current_settings": initial_settings,
                    "client_info": self._client_info,
                    "flametree": pub_ft,
                },
                record_audio=self.config.record_call_audio,
            )
            self.recorder.record_event(
                "call_started",
                {
                    "call_id": self._call_id,
                    "ringing": ringing,
                    "settings": initial_settings,
                },
            )
            print(
                f"[CallSession] Started {self._call_id} -> {self.recorder.call_dir}",
                flush=True,
            )

        await self._send_json({"type": "crm_state", "state": self.crm_state})
        await self._send_json(
            {
                "type": "call_started",
                "call_id": self._call_id,
                "artifacts_dir": str(self.recorder.call_dir) if self.recorder else "",
            }
        )
        await self._emit_delegation_status()
        await self._emit_delivery_stats()
        if ringing:
            await self._send_json({"type": "call_state", "state": "ringing"})

        # Open Flametree runtime session in background so local S2S greeting suffers 0 delay.
        asyncio.create_task(self._open_delegate_session_bg())

        pipeline = self._ensure_pipeline()
        await asyncio.wrap_future(
            self.engine.run_on_worker(
                pipeline.initialize_conversation,
                self._localized_system_prompt(),
                self._active_tools() or None,
                self.language,
            ),
            loop=self.loop,
        )
        await self._emit_context_stats()

    async def _start_call(self) -> None:
        await self._init_call_state(ringing=True)
        self._cancel_event = threading.Event()
        self._active_task = asyncio.create_task(self._run_greeting_turn())

    async def _hang_up(self) -> None:
        self.call_active = False
        self._agent_speaking = False
        self._ptt_active = False
        self._ptt_chunks.clear()
        self._clear_pcm_buffer()
        self._cancel_event.set()
        if self._active_task and not self._active_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._active_task), timeout=5.0)
            except Exception:
                self._active_task.cancel()
        self._active_task = None
        had_ft_session = bool(self.delegate.current_session_id)
        try:
            await self.delegate.close_session()
        except Exception:
            pass
        if had_ft_session and self.recorder is not None:
            self.recorder.record_event("flametree_session_closed", {})
        self._finalize_recorder("hang_up")
        await self._send_json({"type": "barge_in"})
        await self._send_json({"type": "s2s_listen_state", "speaking": False})
        await self._send_json({"type": "call_state", "state": "idle"})

    def _schedule_agent_response(self) -> None:
        prev_task = self._active_task
        cancel_ev = threading.Event()
        self._cancel_event = cancel_ev
        self._active_task = asyncio.create_task(
            self._run_response_after_prev(prev_task, cancel_ev)
        )

    def _schedule_ptt_turn(self, raw_pcm: bytes) -> None:
        prev_task = self._active_task
        cancel_ev = threading.Event()
        self._cancel_event = cancel_ev

        async def _run() -> None:
            if prev_task and not prev_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(prev_task), timeout=5.0)
                except Exception:
                    pass
            async with self._turn_lock:
                if not self.call_active or cancel_ev.is_set():
                    return
                self._last_endpoint_ms = 0.0
                pipeline = self._ensure_pipeline()
                events = await asyncio.wrap_future(
                    self.engine.run_on_worker(pipeline.flush_ptt_audio, raw_pcm),
                    loop=self.loop,
                )
                for ev in events:
                    if ev["type"] == "user_turn_committed":
                        if self.recorder is not None:
                            self.recorder.record_event(
                                "ptt_turn_committed",
                                {"turn_id": ev["turn_id"], "text": ev["text"]},
                            )
                            self.recorder.record_transcript_entry(
                                turn_id=ev["turn_id"],
                                role="user",
                                speaker="Caller",
                                text=ev["text"],
                                language=self.language,
                                extra={"input_mode": "ptt"},
                            )
                        await self._send_json(
                            {
                                "type": "user_turn",
                                "turn_id": ev["turn_id"],
                                "text": ev["text"],
                                "pending_asr": False,
                            }
                        )
                await self._send_json({"type": "call_state", "state": "thinking"})
                await self._execute_agent_turn(
                    cancel_ev=cancel_ev,
                    force_interleaved=False,
                    allow_tools=self.tools_enabled,
                    is_greeting=False,
                )

        self._active_task = asyncio.create_task(_run())

    def _schedule_text_turn(self, text: str) -> None:
        prev_task = self._active_task
        cancel_ev = threading.Event()
        self._cancel_event = cancel_ev

        async def _run() -> None:
            if prev_task and not prev_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(prev_task), timeout=5.0)
                except Exception:
                    pass
            async with self._turn_lock:
                if not self.call_active or cancel_ev.is_set():
                    return
                self._last_endpoint_ms = 0.0
                pipeline = self._ensure_pipeline()
                await asyncio.wrap_future(
                    self.engine.run_on_worker(pipeline.append_text_user_turn, text),
                    loop=self.loop,
                )
                tid = f"usr-{uuid.uuid4().hex[:8]}"
                if self.recorder is not None:
                    self.recorder.record_event(
                        "user_text_turn", {"turn_id": tid, "text": text}
                    )
                    self.recorder.record_transcript_entry(
                        turn_id=tid,
                        role="user",
                        speaker="Caller",
                        text=text,
                        language=self.language,
                        extra={"input_mode": "text"},
                    )
                await self._send_json(
                    {
                        "type": "user_turn",
                        "turn_id": tid,
                        "text": text,
                        "pending_asr": False,
                    }
                )
                await self._send_json({"type": "call_state", "state": "thinking"})
                await self._execute_agent_turn(
                    cancel_ev=cancel_ev,
                    force_interleaved=False,
                    allow_tools=self.tools_enabled,
                    is_greeting=False,
                )

        self._active_task = asyncio.create_task(_run())

    async def _run_greeting_turn(self) -> None:
        async with self._turn_lock:
            if not self.call_active or self._cancel_event.is_set():
                return
            self._last_endpoint_ms = None
            pipeline = self._ensure_pipeline()
            greeting_trigger = build_localized_greeting_trigger(
                lang_code=self.language,
                persona_name=self.persona.name,
                company_name=self.config.company_name,
                product_name=(
                    self.config.product_summary.product_name
                    if self.config.product_summary
                    else "Ubuntu Rewards Credit Card"
                ),
            )
            await asyncio.wrap_future(
                self.engine.run_on_worker(
                    pipeline.append_text_user_turn, greeting_trigger
                ),
                loop=self.loop,
            )
            await self._execute_agent_turn(
                cancel_ev=self._cancel_event,
                force_interleaved=True,
                allow_tools=False,
                is_greeting=True,
            )

    async def _run_response_after_prev(
        self,
        prev_task: Optional[asyncio.Task[None]],
        cancel_ev: threading.Event,
    ) -> None:
        if prev_task and not prev_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(prev_task), timeout=5.0)
            except Exception:
                pass
        async with self._turn_lock:
            if not self.call_active or cancel_ev.is_set():
                return
            await self._send_json({"type": "call_state", "state": "thinking"})
            await self._execute_agent_turn(
                cancel_ev=cancel_ev,
                force_interleaved=False,
                allow_tools=self.tools_enabled,
                is_greeting=False,
            )

    async def _run_worker_stream(
        self,
        worker_fn: Callable[..., Any],
        args_builder: Callable[[Callable[[dict[str, Any]], None]], tuple[Any, ...]],
        cancel_ev: threading.Event,
        turn_id: str,
        stream_state: dict[str, Any],
        on_delegation_query: Optional[Callable[[str], None]] = None,
    ) -> Any:
        """Execute a blocking MLX streaming function and pump audio/text events immediately."""
        event_queue: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue()

        def on_engine_event(ev: dict[str, Any]) -> None:
            if ev.get("type") == "text_delta":
                self._recent_agent_text = ev.get("text", "")
            self.loop.call_soon_threadsafe(event_queue.put_nowait, ev)

        self._generating = True
        future = self.engine.run_on_worker(worker_fn, *args_builder(on_engine_event))
        future.add_done_callback(
            lambda _f: self.loop.call_soon_threadsafe(event_queue.put_nowait, None)
        )

        try:
            while True:
                ev = await event_queue.get()
                if ev is None:
                    break
                if cancel_ev.is_set():
                    continue
                ev_type = ev.get("type")
                if ev_type == "delegation_query_ready":
                    q_text = str(ev.get("query") or "")
                    if self.recorder is not None and q_text:
                        self.recorder.record_event(
                            "delegation_query_ready",
                            {"turn_id": turn_id, "query": q_text},
                        )
                    if on_delegation_query is not None:
                        on_delegation_query(q_text)
                elif ev_type == "text_delta":
                    if stream_state.get("first_token_ts") is None:
                        now_tok = time.perf_counter()
                        stream_state["first_token_ts"] = now_tok
                        t_start_turn = float(stream_state.get("t_turn_start") or now_tok)
                        live_ttft_ms = round((now_tok - t_start_turn) * 1000.0, 1)
                        if self.recorder is not None:
                            self.recorder.record_event(
                                "first_token_sent",
                                {"turn_id": turn_id, "ttft_ms": live_ttft_ms},
                            )
                        await self._send_json(
                            {
                                "type": "live_ttft",
                                "turn_id": turn_id,
                                "ttft_ms": live_ttft_ms,
                            }
                        )
                    if not stream_state["turn_started_ui"]:
                        stream_state["turn_started_ui"] = True
                        await self._send_json(
                            {
                                "type": "assistant_turn_start",
                                "turn_id": turn_id,
                                "persona_name": self.persona.name,
                            }
                        )
                    seg = ev.get("segment", "answer")
                    raw_text = str(ev.get("text") or "")
                    prefix = stream_state.get("filler_prefix", "")
                    if seg == "filler":
                        stream_state["filler_prefix"] = raw_text
                        display_text = raw_text
                    else:
                        display_text = (
                            f"{prefix} {raw_text}".strip() if prefix else raw_text
                        )
                    stream_state["display_text"] = display_text
                    await self._send_json(
                        {
                            "type": "text_delta",
                            "turn_id": turn_id,
                            "delta": ev["delta"],
                            "text": display_text,
                            "segment": seg,
                        }
                    )
                elif ev_type == "audio_chunk":
                    if not stream_state["turn_started_ui"]:
                        stream_state["turn_started_ui"] = True
                        await self._send_json(
                            {
                                "type": "assistant_turn_start",
                                "turn_id": turn_id,
                                "persona_name": self.persona.name,
                            }
                        )
                    if not stream_state["speech_started_ui"]:
                        stream_state["speech_started_ui"] = True
                        self._agent_speaking = True
                        await self._send_json(
                            {"type": "call_state", "state": "speaking"}
                        )
                    stream_state["chunks_sent"] = int(stream_state.get("chunks_sent") or 0) + 1
                    is_first_chunk = stream_state["first_audio_ts"] is None
                    if is_first_chunk:
                        stream_state["first_audio_ts"] = time.perf_counter()
                        stream_state["first_chunk_samples"] = int(
                            ev.get("samples") or len(ev["pcm16"]) // 2
                        )
                    await self._send_bytes(ev["pcm16"])
                    if is_first_chunk:
                        t_start_turn = float(
                            stream_state.get("t_turn_start") or stream_state["first_audio_ts"]
                        )
                        srv_ttfa_ms = round(
                            (stream_state["first_audio_ts"] - t_start_turn) * 1000.0,
                            1,
                        )
                        if self.recorder is not None:
                            self.recorder.record_event(
                                "first_audio_chunk_sent",
                                {
                                    "turn_id": turn_id,
                                    "ttfa_server_ms": srv_ttfa_ms,
                                    "first_chunk_samples": stream_state["first_chunk_samples"],
                                },
                            )
                        await self._send_json(
                            {
                                "type": "live_ttfa",
                                "turn_id": turn_id,
                                "ttfa_server_ms": srv_ttfa_ms,
                            }
                        )

            return await asyncio.wrap_future(future, loop=self.loop)
        finally:
            self._generating = False

    async def _handle_mid_gen_barge_in(
        self,
        turn_id: str,
        stream_state: dict[str, Any],
        interrupted_turn: Optional[TurnRecord],
        mid_gen_mic_events: list[dict[str, Any]],
        filler_frames: int = 0,
        turn_ctx: Optional[dict[str, Any]] = None,
    ) -> None:
        """Rollback conv_cache, record interrupted metrics & cancel latency, forward events."""
        t0 = time.perf_counter()
        self._agent_speaking = False
        await self._send_json({"type": "barge_in"})
        played_frames = max(0, (self._played_samples_24k // 1920) - int(filler_frames))
        trunc_disp: Optional[str] = None
        prefix = stream_state.get("filler_prefix", "")
        if self.pipeline is not None:
            truncated = await asyncio.wrap_future(
                self.engine.run_on_worker(
                    self.pipeline.rollback_interrupted_assistant_turn,
                    played_frames,
                    interrupted_turn if played_frames > 0 else None,
                    False,
                ),
                loop=self.loop,
            )
            if truncated is not None and stream_state.get("turn_started_ui"):
                trunc_disp = f"{prefix} {truncated.text}".strip() if prefix else truncated.text
            elif stream_state.get("turn_started_ui") and stream_state.get("display_text"):
                trunc_disp = str(stream_state.get("display_text") or "")

        cancel_ms = (time.perf_counter() - t0) * 1000.0
        int_metrics: Optional[dict[str, Any]] = None
        if turn_ctx is not None and (
            stream_state.get("first_audio_ts") is not None
            or stream_state.get("turn_started_ui")
        ):
            t0_turn = float(turn_ctx.get("t_turn_start") or t0)
            first_tok = stream_state.get("first_token_ts")
            first_aud = stream_state.get("first_audio_ts")
            gen_wall_ms = (t0 - t0_turn) * 1000.0
            raw_codes = (
                getattr(interrupted_turn, "raw_audio_codes", None)
                if interrupted_turn is not None
                else None
            )
            partial_frames = (
                (
                    int(raw_codes.shape[0])
                    if hasattr(raw_codes, "shape")
                    else len(raw_codes)
                )
                if raw_codes is not None
                else 0
            ) + int(filler_frames)
            eff_frames = max(played_frames + int(filler_frames), partial_frames)
            audio_sec = (eff_frames * 1920) / 24000.0
            srv_ttfa = ((first_aud - t0_turn) * 1000.0) if first_aud else gen_wall_ms
            eff_ttft = float(
                turn_ctx.get("ttft_ms")
                or (((first_tok - t0_turn) * 1000.0) if first_tok else 0.0)
            )
            after_bi = bool(turn_ctx.get("after_barge_in"))
            recorded_int = self.stats.record_turn(
                TurnDeliveryMetrics(
                    turn_id=turn_id,
                    turn_index=self._turn_counter,
                    language=self.language,
                    delegation_mode=self.delegation_mode,
                    route_decision=str(turn_ctx.get("route_decision") or "direct"),
                    after_barge_in=after_bi,
                    interrupted=True,
                    endpoint_ms=turn_ctx.get("endpoint_ms"),
                    prefill_ms=round(float(turn_ctx.get("prefill_ms") or 0.0), 1),
                    ttft_ms=round(eff_ttft, 1),
                    ttfa_server_ms=round(srv_ttfa, 1),
                    ttfa_filler_ms=turn_ctx.get("ttfa_filler_ms"),
                    delegation_ms=turn_ctx.get("delegation_ms"),
                    delegation_source=turn_ctx.get("delegation_source"),
                    barge_in_cancel_ms=turn_ctx.get("barge_in_cancel_ms"),
                    ttfa_after_barge_in_ms=round(srv_ttfa, 1) if after_bi else None,
                    audio_chunks_sent=int(stream_state.get("chunks_sent") or 0),
                    first_chunk_samples=int(stream_state.get("first_chunk_samples") or 0),
                    audio_duration_sec=round(audio_sec, 2),
                    gen_rtf=round((gen_wall_ms / 1000.0) / audio_sec, 3) if audio_sec > 0 else 0.0,
                )
            )
            int_metrics = {
                "ttft_ms": recorded_int.ttft_ms,
                "ttfa_server_ms": recorded_int.ttfa_server_ms,
                "ttfa_perceived_ms": recorded_int.ttfa_perceived_ms,
                "route_decision": recorded_int.route_decision,
            }

        if trunc_disp is not None and stream_state.get("turn_started_ui"):
            payload_trunc: dict[str, Any] = {
                "type": "assistant_turn_truncated",
                "turn_id": turn_id,
                "text": trunc_disp,
            }
            if int_metrics is not None:
                payload_trunc["metrics"] = int_metrics
            await self._send_json(payload_trunc)
            if self.recorder is not None and trunc_disp:
                self.recorder.record_transcript_entry(
                    turn_id=turn_id,
                    role="assistant",
                    speaker=self.persona.name,
                    text=trunc_disp,
                    language=self.language,
                    filler_text=prefix or None,
                    truncated=True,
                    interrupted=True,
                    metrics=int_metrics,
                )

        if self.recorder is not None:
            self.recorder.record_event(
                "barge_in_mid_gen",
                {
                    "turn_id": turn_id,
                    "played_samples_24k": self._played_samples_24k,
                    "played_frames": played_frames,
                    "filler_frames": int(filler_frames),
                    "cancel_ms": round(cancel_ms, 1),
                    "truncated_text": trunc_disp,
                },
            )
        self.stats.mark_barge_in(cancel_ms=cancel_ms)
        await self._emit_delivery_stats()

        if mid_gen_mic_events:
            filtered = [
                e for e in mid_gen_mic_events if e.get("type") != "barge_in_detected"
            ]
            if filtered:
                await self._handle_live_events(filtered)

    async def _run_crm_tool_rounds(
        self,
        initial_calls: list[dict[str, Any]],
        pipeline: Any,
        turn_id: str,
        stream_state: dict[str, Any],
        poll_mic_sync: Callable[[], list[dict[str, Any]]],
        mid_gen_mic_events: list[dict[str, Any]],
        cancel_ev: threading.Event,
    ) -> Optional[TurnOutcome]:
        """Execute CRM tool calls and generate spoken confirmation response."""
        tool_map = self._tool_map()
        calls = initial_calls
        for round_idx in range(3):
            if cancel_ev.is_set() or not self.call_active:
                return TurnOutcome(kind="cancelled", cancelled=True)

            results: list[dict[str, Any]] = []
            for call in calls:
                fn_info = call.get("function", call)
                fn_name = fn_info.get("name", "")
                fn_args = fn_info.get("arguments") or {}
                spec = tool_map.get(fn_name)
                if spec is None:
                    res: dict[str, Any] = {"ok": False, "error": f"Unknown tool: {fn_name}"}
                else:
                    try:
                        res = spec.handler(**fn_args)
                    except Exception as err:
                        res = {"ok": False, "error": str(err)}
                results.append({"tool": fn_name, "arguments": fn_args, "result": res})
                self._update_crm_state(fn_name, fn_args, res)
                if self.recorder is not None:
                    self.recorder.record_event(
                        "tool_called",
                        {
                            "turn_id": turn_id,
                            "round": round_idx,
                            "tool": fn_name,
                            "arguments": fn_args,
                            "result": res,
                        },
                    )
                    summary_txt = res.get("summary") or json.dumps(
                        res, ensure_ascii=False
                    )
                    self.recorder.record_transcript_entry(
                        turn_id=f"{turn_id}-tool-{round_idx}-{fn_name}",
                        role="tool",
                        speaker=fn_name,
                        text=f"args={json.dumps(fn_args, ensure_ascii=False)} -> {summary_txt}",
                        upsert_by_turn_id=False,
                    )
                await self._send_json(
                    {
                        "type": "tool_event",
                        "tool": fn_name,
                        "arguments": fn_args,
                        "result": res,
                    }
                )
            if self.recorder is not None:
                self.recorder.record_crm_and_delegation(
                    self.crm_state, self.delegation_log
                )
            await self._send_json({"type": "crm_state", "state": self.crm_state})
            await asyncio.wrap_future(
                self.engine.run_on_worker(
                    pipeline.append_tool_exchange,
                    calls,
                    json.dumps(results, ensure_ascii=False),
                ),
                loop=self.loop,
            )

            outcome: TurnOutcome = await self._run_worker_stream(
                pipeline.generate_assistant_turn,
                lambda cb: (
                    self.persona,
                    None,
                    True,
                    cancel_ev,
                    cb,
                    poll_mic_sync,
                    "answer",
                ),
                cancel_ev=cancel_ev,
                turn_id=turn_id,
                stream_state=stream_state,
            )
            if outcome.cancelled or cancel_ev.is_set():
                return outcome
            if outcome.kind == "tool_calls" and outcome.tool_calls and round_idx < 2:
                calls = outcome.tool_calls
                continue
            return outcome
        return None

    def _record_batch_plan(self, turn_id: str, plan: BatchPlanOutcome) -> None:
        if self.recorder is None:
            return
        self.recorder.record_event(
            "batch_plan_completed",
            {
                "turn_id": turn_id,
                "decision": plan.decision,
                "delegation_query": plan.delegation_query,
                "filler_text": plan.filler_text,
                "filler_audio_frames": plan.filler_audio_frames,
                "fork_merge_ms": round(float(plan.fork_merge_ms or 0.0), 1),
                "ttft_ms": round(float(plan.ttft_ms or 0.0), 1),
                "ttfa_filler_ms": (
                    round(float(plan.ttfa_filler_ms), 1)
                    if plan.ttfa_filler_ms is not None
                    else None
                ),
                "slots_per_sec": round(float(plan.slots_per_sec or 0.0), 1),
            },
        )

    def _record_delegation_result(
        self,
        turn_id: str,
        del_ev: dict[str, Any],
        del_res: DelegationResult,
    ) -> None:
        if self.recorder is None:
            return
        self.recorder.record_event(
            "delegation_completed",
            {
                "turn_id": turn_id,
                "query": del_ev["query"],
                "answer": del_ev["answer"],
                "source": del_ev["source"],
                "latency_ms": del_ev["latency_ms"],
                "session_id": del_res.session_id,
                "error": del_ev["error"],
            },
        )
        self.recorder.record_transcript_entry(
            turn_id=f"{turn_id}-del",
            role="delegation",
            speaker=str(del_ev["source"]),
            text=f"Q: {del_ev['query']} | A: {del_ev['answer']}",
            language=self.language,
            extra={
                "latency_ms": del_ev["latency_ms"],
                "session_id": del_res.session_id,
                "error": del_ev["error"],
            },
            upsert_by_turn_id=False,
        )
        self.recorder.record_crm_and_delegation(self.crm_state, self.delegation_log)

    async def _execute_agent_turn(
        self,
        cancel_ev: threading.Event,
        force_interleaved: bool,
        allow_tools: bool,
        is_greeting: bool = False,
    ) -> None:
        pipeline = self._ensure_pipeline()
        tools = self._active_tools() if allow_tools else None
        turn_id = f"ast-{uuid.uuid4().hex[:8]}"
        self._turn_counter += 1
        t_turn_start = time.perf_counter()
        self._turn_start_wall[turn_id] = t_turn_start
        self._played_samples_24k = 0
        self._last_assistant_turn = None
        self._last_assistant_turn_id = turn_id

        mid_gen_mic_events: list[dict[str, Any]] = []
        stream_state: dict[str, Any] = {
            "t_turn_start": t_turn_start,
            "turn_started_ui": False,
            "speech_started_ui": False,
            "first_token_ts": None,
            "first_audio_ts": None,
            "chunks_sent": 0,
            "first_chunk_samples": 0,
            "filler_prefix": "",
            "display_text": "",
        }
        delegation_task: Optional[asyncio.Task[DelegationResult]] = None
        filler_frames_count = 0

        def poll_mic_sync() -> list[dict[str, Any]]:
            if self.input_mode != "live" or self.muted:
                return []
            chunk = self._take_pcm_buffer()
            if not chunk:
                return []
            evs = pipeline.ingest_pcm_and_listen(
                chunk,
                agent_speaking=True,
                recent_agent_text=self._recent_agent_text,
                patience=self.patience,
            )
            mid_gen_mic_events.extend(evs)
            return evs

        def start_delegation_async(q: str) -> None:
            nonlocal delegation_task
            if delegation_task is None and q:
                if self.recorder is not None:
                    self.recorder.record_event(
                        "delegation_started",
                        {"turn_id": turn_id, "query": q, "language": self.language},
                    )
                delegation_task = asyncio.create_task(
                    self.delegate.query(q, self.language)
                )

        endpoint_ms = self._last_endpoint_ms
        self._last_endpoint_ms = None
        after_barge_in, barge_in_cancel_ms = self.stats.consume_pending_barge_in()

        if self.recorder is not None:
            self.recorder.record_event(
                "assistant_turn_started",
                {
                    "turn_id": turn_id,
                    "turn_index": self._turn_counter,
                    "is_greeting": is_greeting,
                    "language": self.language,
                    "delegation_mode": self.delegation_mode,
                    "dynamic_filler": self.dynamic_filler,
                    "endpoint_ms": endpoint_ms,
                    "after_barge_in": after_barge_in,
                },
            )

        route_decision = "greeting" if is_greeting else "direct"
        prefill_ms = 0.0
        ttft_ms = 0.0
        ttfa_filler_ms: Optional[float] = None
        ttfa_answer_ms: Optional[float] = None
        delegation_ms: Optional[float] = None
        delegation_source: Optional[str] = None
        turn_ctx: dict[str, Any] = {
            "t_turn_start": t_turn_start,
            "route_decision": route_decision,
            "endpoint_ms": endpoint_ms,
            "after_barge_in": after_barge_in,
            "barge_in_cancel_ms": barge_in_cancel_ms,
            "prefill_ms": prefill_ms,
            "ttft_ms": ttft_ms,
            "ttfa_filler_ms": ttfa_filler_ms,
            "delegation_ms": delegation_ms,
            "delegation_source": delegation_source,
        }
        final_outcome: Optional[TurnOutcome] = None
        plan: Optional[BatchPlanOutcome] = None

        try:
            if is_greeting or force_interleaved:
                final_outcome = await self._run_worker_stream(
                    pipeline.generate_assistant_turn,
                    lambda cb: (
                        self.persona,
                        tools if not force_interleaved else None,
                        force_interleaved,
                        cancel_ev,
                        cb,
                        poll_mic_sync,
                        "answer",
                    ),
                    cancel_ev=cancel_ev,
                    turn_id=turn_id,
                    stream_state=stream_state,
                )
                if final_outcome.cancelled or cancel_ev.is_set():
                    await self._handle_mid_gen_barge_in(
                        turn_id,
                        stream_state,
                        final_outcome.turn_record,
                        mid_gen_mic_events,
                        turn_ctx=turn_ctx,
                    )
                    return
                if final_outcome.kind == "tool_calls" and final_outcome.tool_calls:
                    route_decision = "tool_calls"
                    turn_ctx["route_decision"] = route_decision
                    final_outcome = await self._run_crm_tool_rounds(
                        final_outcome.tool_calls,
                        pipeline,
                        turn_id,
                        stream_state,
                        poll_mic_sync,
                        mid_gen_mic_events,
                        cancel_ev,
                    )
                    if final_outcome is None or final_outcome.cancelled or cancel_ev.is_set():
                        await self._handle_mid_gen_barge_in(
                            turn_id,
                            stream_state,
                            final_outcome.turn_record if final_outcome else None,
                            mid_gen_mic_events,
                            turn_ctx=turn_ctx,
                        )
                        return

            elif self.delegation_mode == "always_delegate":
                route_decision = "delegate"
                turn_ctx["route_decision"] = route_decision
                if self.dynamic_filler:
                    plan = await self._run_worker_stream(
                        pipeline.run_filler_and_query_batch,
                        lambda cb: (cancel_ev, cb, poll_mic_sync),
                        cancel_ev=cancel_ev,
                        turn_id=turn_id,
                        stream_state=stream_state,
                        on_delegation_query=start_delegation_async,
                    )
                    self._record_batch_plan(turn_id, plan)
                    if plan.decision == "cancelled" or cancel_ev.is_set():
                        if delegation_task and not delegation_task.done():
                            delegation_task.cancel()
                        await self._handle_mid_gen_barge_in(
                            turn_id,
                            stream_state,
                            plan.filler_turn,
                            mid_gen_mic_events,
                            filler_frames=plan.filler_audio_frames,
                            turn_ctx=turn_ctx,
                        )
                        return
                    filler_frames_count = plan.filler_audio_frames
                    prefill_ms = plan.fork_merge_ms
                    ttft_ms = plan.ttft_ms
                    ttfa_filler_ms = plan.ttfa_filler_ms
                    turn_ctx["prefill_ms"] = prefill_ms
                    turn_ctx["ttft_ms"] = ttft_ms
                    turn_ctx["ttfa_filler_ms"] = ttfa_filler_ms
                    query_text = plan.delegation_query
                else:
                    query_text = await asyncio.wrap_future(
                        self.engine.run_on_worker(pipeline.prepare_for_batched_generation),
                        loop=self.loop,
                    )
                    start_delegation_async(query_text)

                if delegation_task is None:
                    start_delegation_async(query_text)
                assert delegation_task is not None
                del_res = await delegation_task
                if cancel_ev.is_set():
                    await self._handle_mid_gen_barge_in(
                        turn_id,
                        stream_state,
                        plan.filler_turn if plan else None,
                        mid_gen_mic_events,
                        filler_frames=filler_frames_count,
                        turn_ctx=turn_ctx,
                    )
                    return

                delegation_ms = round(del_res.latency_ms, 1)
                delegation_source = del_res.source
                turn_ctx["delegation_ms"] = delegation_ms
                turn_ctx["delegation_source"] = delegation_source
                await self._emit_delegation_status()
                del_ev = {
                    "type": "delegation_event",
                    "turn_id": turn_id,
                    "query": del_res.query or query_text,
                    "answer": del_res.answer,
                    "source": del_res.source,
                    "latency_ms": delegation_ms,
                    "error": del_res.error,
                    "language": self.language,
                }
                self.delegation_log.append(del_ev)
                self._record_delegation_result(turn_id, del_ev, del_res)
                await self._send_json(del_ev)

                await asyncio.wrap_future(
                    self.engine.run_on_worker(
                        pipeline.append_delegated_knowledge_context,
                        del_res.query or query_text,
                        del_res.answer,
                        del_res.source,
                    ),
                    loop=self.loop,
                )
                ans_t0 = time.perf_counter()
                final_outcome = await self._run_worker_stream(
                    pipeline.generate_assistant_turn,
                    lambda cb: (
                        self.persona,
                        None,
                        True,
                        cancel_ev,
                        cb,
                        poll_mic_sync,
                        "answer",
                    ),
                    cancel_ev=cancel_ev,
                    turn_id=turn_id,
                    stream_state=stream_state,
                )
                ttfa_answer_ms = (
                    round((ans_t0 - t_turn_start) * 1000.0 + final_outcome.ttfa_ms, 1)
                    if final_outcome
                    else None
                )
                if final_outcome.cancelled or cancel_ev.is_set():
                    await self._handle_mid_gen_barge_in(
                        turn_id,
                        stream_state,
                        final_outcome.turn_record,
                        mid_gen_mic_events,
                        filler_frames=filler_frames_count,
                        turn_ctx=turn_ctx,
                    )
                    return

            else:
                # selective_tool mode: MLX B=2 speculative router + direct OR filler + delegation
                plan = await self._run_worker_stream(
                    pipeline.run_speculative_selective_batch,
                    lambda cb: (
                        self.persona,
                        tools,
                        self.dynamic_filler,
                        cancel_ev,
                        cb,
                        poll_mic_sync,
                    ),
                    cancel_ev=cancel_ev,
                    turn_id=turn_id,
                    stream_state=stream_state,
                    on_delegation_query=start_delegation_async,
                )
                self._record_batch_plan(turn_id, plan)
                if plan.decision == "cancelled" or cancel_ev.is_set():
                    if delegation_task and not delegation_task.done():
                        delegation_task.cancel()
                    await self._handle_mid_gen_barge_in(
                        turn_id,
                        stream_state,
                        plan.filler_turn,
                        mid_gen_mic_events,
                        filler_frames=plan.filler_audio_frames,
                        turn_ctx=turn_ctx,
                    )
                    return

                if plan.decision == "tool_calls" and plan.direct_outcome:
                    route_decision = "tool_calls"
                    turn_ctx["route_decision"] = route_decision
                    final_outcome = await self._run_crm_tool_rounds(
                        plan.direct_outcome.tool_calls,
                        pipeline,
                        turn_id,
                        stream_state,
                        poll_mic_sync,
                        mid_gen_mic_events,
                        cancel_ev,
                    )
                    if final_outcome is None or final_outcome.cancelled or cancel_ev.is_set():
                        await self._handle_mid_gen_barge_in(
                            turn_id,
                            stream_state,
                            final_outcome.turn_record if final_outcome else None,
                            mid_gen_mic_events,
                            turn_ctx=turn_ctx,
                        )
                        return

                elif plan.decision == "delegate":
                    route_decision = "delegate"
                    filler_frames_count = plan.filler_audio_frames
                    prefill_ms = plan.fork_merge_ms
                    ttft_ms = plan.ttft_ms
                    ttfa_filler_ms = plan.ttfa_filler_ms or None
                    turn_ctx["route_decision"] = route_decision
                    turn_ctx["prefill_ms"] = prefill_ms
                    turn_ctx["ttft_ms"] = ttft_ms
                    turn_ctx["ttfa_filler_ms"] = ttfa_filler_ms
                    query_text = plan.delegation_query
                    if delegation_task is None:
                        start_delegation_async(query_text)
                    assert delegation_task is not None
                    del_res = await delegation_task
                    if cancel_ev.is_set():
                        await self._handle_mid_gen_barge_in(
                            turn_id,
                            stream_state,
                            plan.filler_turn,
                            mid_gen_mic_events,
                            filler_frames=filler_frames_count,
                            turn_ctx=turn_ctx,
                        )
                        return

                    delegation_ms = round(del_res.latency_ms, 1)
                    delegation_source = del_res.source
                    turn_ctx["delegation_ms"] = delegation_ms
                    turn_ctx["delegation_source"] = delegation_source
                    await self._emit_delegation_status()
                    del_ev = {
                        "type": "delegation_event",
                        "turn_id": turn_id,
                        "query": del_res.query or query_text,
                        "answer": del_res.answer,
                        "source": del_res.source,
                        "latency_ms": delegation_ms,
                        "error": del_res.error,
                        "language": self.language,
                    }
                    self.delegation_log.append(del_ev)
                    self._record_delegation_result(turn_id, del_ev, del_res)
                    await self._send_json(del_ev)

                    await asyncio.wrap_future(
                        self.engine.run_on_worker(
                            pipeline.append_delegated_knowledge_context,
                            del_res.query or query_text,
                            del_res.answer,
                            del_res.source,
                        ),
                        loop=self.loop,
                    )
                    ans_t0 = time.perf_counter()
                    final_outcome = await self._run_worker_stream(
                        pipeline.generate_assistant_turn,
                        lambda cb: (
                            self.persona,
                            None,
                            True,
                            cancel_ev,
                            cb,
                            poll_mic_sync,
                            "answer",
                        ),
                        cancel_ev=cancel_ev,
                        turn_id=turn_id,
                        stream_state=stream_state,
                    )
                    ttfa_answer_ms = (
                        round((ans_t0 - t_turn_start) * 1000.0 + final_outcome.ttfa_ms, 1)
                        if final_outcome
                        else None
                    )
                    if final_outcome.cancelled or cancel_ev.is_set():
                        await self._handle_mid_gen_barge_in(
                            turn_id,
                            stream_state,
                            final_outcome.turn_record,
                            mid_gen_mic_events,
                            filler_frames=filler_frames_count,
                            turn_ctx=turn_ctx,
                        )
                        return

                else:
                    route_decision = "direct"
                    final_outcome = plan.direct_outcome

        except Exception as exc:
            self._generating = False
            if delegation_task and not delegation_task.done():
                delegation_task.cancel()
            if self.recorder is not None:
                self.recorder.record_error("inference", str(exc))
            await self._send_json({"type": "error", "message": f"Inference error: {exc}"})
            await self._send_json({"type": "call_state", "state": "listening"})
            return

        if final_outcome is None:
            return

        # Completed full spoken response (direct, CRM tool response, or filler + Flametree answer).
        filler_prefix = stream_state.get("filler_prefix", "")
        combined_text = (
            f"{filler_prefix} {final_outcome.text}".strip()
            if filler_prefix and final_outcome.text
            else (final_outcome.text or filler_prefix)
        )
        self._last_assistant_turn = final_outcome.turn_record
        self._recent_agent_text = combined_text

        total_audio_frames = int(final_outcome.audio_frames) + int(filler_frames_count)
        audio_sec = (total_audio_frames * 1920) / 24000.0
        total_wall_ms = (time.perf_counter() - t_turn_start) * 1000.0
        first_audio_ts = stream_state.get("first_audio_ts")
        ttfa_server_ms = (
            (first_audio_ts - t_turn_start) * 1000.0
            if first_audio_ts is not None
            else (plan.ttfa_filler_ms if plan and plan.ttfa_filler_ms else final_outcome.ttfa_ms)
        )
        eff_prefill_ms = prefill_ms or final_outcome.prefill_ms
        eff_ttft_ms = ttft_ms or final_outcome.ttft_ms or eff_prefill_ms
        chunks_sent = max(
            int(final_outcome.audio_chunks_sent)
            + (plan.audio_chunks_sent if plan and route_decision == "delegate" else 0),
            int(stream_state.get("chunks_sent") or 0),
        )
        first_samples = (
            (
                plan.first_chunk_samples
                if plan and plan.first_chunk_samples > 0
                else final_outcome.first_chunk_samples
            )
            or int(stream_state.get("first_chunk_samples") or 0)
        )
        codec_avg_ms = final_outcome.codec_chunk_avg_ms
        if plan and plan.codec_chunk_ms_list and route_decision == "delegate":
            combined_cms = plan.codec_chunk_ms_list + (
                [final_outcome.codec_chunk_avg_ms] if final_outcome.codec_chunk_avg_ms > 0 else []
            )
            codec_avg_ms = sum(combined_cms) / len(combined_cms) if combined_cms else 0.0

        gen_rtf = (total_wall_ms / 1000.0) / audio_sec if audio_sec > 0 else 0.0
        slots_per_sec = final_outcome.slots_per_sec or (plan.slots_per_sec if plan else 0.0)

        turn_metrics = TurnDeliveryMetrics(
            turn_id=turn_id,
            turn_index=self._turn_counter,
            language=self.language,
            delegation_mode=self.delegation_mode,
            route_decision=route_decision,
            after_barge_in=after_barge_in,
            endpoint_ms=endpoint_ms,
            prefill_ms=round(eff_prefill_ms, 1),
            ttft_ms=round(eff_ttft_ms, 1),
            ttfa_server_ms=round(ttfa_server_ms, 1),
            ttfa_filler_ms=round(ttfa_filler_ms, 1) if ttfa_filler_ms is not None else None,
            ttfa_answer_ms=round(ttfa_answer_ms, 1) if ttfa_answer_ms is not None else None,
            delegation_ms=delegation_ms,
            delegation_source=delegation_source,
            barge_in_cancel_ms=barge_in_cancel_ms,
            ttfa_after_barge_in_ms=(
                round(ttfa_server_ms, 1) if after_barge_in else None
            ),
            audio_chunks_sent=chunks_sent,
            first_chunk_samples=first_samples,
            codec_chunk_avg_ms=round(codec_avg_ms, 2),
            audio_duration_sec=round(audio_sec, 2),
            gen_rtf=round(gen_rtf, 3),
            slots_per_sec=round(slots_per_sec, 1),
        )
        self.stats.record_turn(turn_metrics)
        msg_metrics: dict[str, Any] = {
            "ttft_ms": turn_metrics.ttft_ms,
            "ttfa_server_ms": turn_metrics.ttfa_server_ms,
            "ttfa_perceived_ms": turn_metrics.ttfa_perceived_ms,
            "ttfa_filler_ms": turn_metrics.ttfa_filler_ms,
            "ttfa_answer_ms": turn_metrics.ttfa_answer_ms,
            "delegation_ms": turn_metrics.delegation_ms,
            "route_decision": route_decision,
        }

        if stream_state["turn_started_ui"]:
            await self._send_json(
                {
                    "type": "assistant_turn_end",
                    "turn_id": turn_id,
                    "text": combined_text,
                    "metrics": msg_metrics,
                }
            )
        await self._send_json({"type": "audio_end", "turn_id": turn_id})

        if self.recorder is not None:
            used_tts_fallback = bool(getattr(final_outcome, "used_tts_fallback", False))
            self.recorder.record_event(
                "assistant_turn_completed",
                {
                    "turn_id": turn_id,
                    "route_decision": route_decision,
                    "text": combined_text,
                    "filler_text": filler_prefix or None,
                    "audio_frames": total_audio_frames,
                    "audio_sec": round(audio_sec, 2),
                    "audio_chunks_sent": chunks_sent,
                    "used_tts_fallback": used_tts_fallback,
                    "ttfa_server_ms": round(ttfa_server_ms, 1),
                    "ttft_ms": round(eff_ttft_ms, 1),
                    "ttfa_perceived_ms": turn_metrics.ttfa_perceived_ms,
                    "gen_rtf": round(gen_rtf, 3),
                    "slots_per_sec": round(slots_per_sec, 1),
                },
            )
            self.recorder.record_transcript_entry(
                turn_id=turn_id,
                role="assistant",
                speaker=self.persona.name,
                text=combined_text,
                language=self.language,
                route_decision=route_decision,
                filler_text=filler_prefix or None,
                extra={
                    "is_greeting": is_greeting,
                    "audio_sec": round(audio_sec, 2),
                    "used_tts_fallback": used_tts_fallback,
                },
                metrics=msg_metrics,
            )

        await self._send_json(
            {
                "type": "metrics",
                "turn_id": turn_id,
                "prefill_ms": round(eff_prefill_ms, 1),
                "ttft_ms": round(eff_ttft_ms, 1),
                "ttfa_ms": round(ttfa_server_ms, 1),
                "slots_per_sec": round(slots_per_sec, 1),
                "audio_frames": total_audio_frames,
                "audio_sec": round(audio_sec, 2),
                "audio_chunks_sent": chunks_sent,
                "route_decision": route_decision,
                "delegation_ms": delegation_ms,
            }
        )
        await self._emit_delivery_stats()
        await self._emit_context_stats()

        if is_greeting and self.pipeline is not None and not cancel_ev.is_set():
            await asyncio.wrap_future(
                self.engine.run_on_worker(self.pipeline.finalize_greeting_turn),
                loop=self.loop,
            )
            if self.recorder is not None:
                self.recorder.record_event("greeting_finalized", {"turn_id": turn_id})

        if not stream_state["speech_started_ui"]:
            self._agent_speaking = False
            await self._send_json({"type": "call_state", "state": "listening"})

        # Drain any PCM that queued right as generation finished.
        with self._pcm_lock:
            has_queued = len(self._pcm_buffer) >= 10240
        if has_queued and not self._drain_in_flight and self.input_mode == "live":
            self._drain_in_flight = True
            asyncio.create_task(self._drain_live_pcm_loop())

    def _update_crm_state(
        self, fn_name: str, args: dict[str, Any], result: dict[str, Any]
    ) -> None:
        entry = {
            "timestamp": time.strftime("%H:%M:%S"),
            "tool": fn_name,
            "arguments": args,
            "result": result,
        }
        self.crm_state["tool_log"].append(entry)
        if fn_name == "check_pre_qualification" and result.get("ok"):
            self.crm_state["qualification"] = result
        elif fn_name == "calculate_monthly_repayment" and result.get("ok"):
            self.crm_state["repayment"] = result
        elif fn_name == "submit_card_application" and result.get("ok"):
            self.crm_state["application"] = result
        elif fn_name == "schedule_callback" and result.get("ok"):
            self.crm_state["callback"] = result
