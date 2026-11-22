"""Tests for the Linux/CUDA vLLM-Omni HTTP/SSE backend (`voice_agent/vllm_backend.py`)."""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import threading
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from telesales_agent import build_config, build_personas, build_tools
from voice_agent.backend_factory import build_engine
from voice_agent.config import EngineSettings
from voice_agent.protocol import TurnRecord, pcm16_to_wav_bytes
from voice_agent.server import create_app
from voice_agent.session import CallSession
from voice_agent.vllm_backend import (
    VllmLiveS2SListener,
    VllmOmniEngine,
    VllmStreamingConversationSession,
)


def _pcm_frame_bytes(frames: int = 2, amplitude: int = 400) -> bytes:
    """Create valid 24 kHz mono PCM16 bytes (1920 samples per LFM3 codec frame)."""
    samples = frames * 1920
    t = np.arange(samples, dtype=np.float32)
    wave_i16 = (np.sin(2.0 * np.pi * 440.0 * t / 24000.0) * amplitude).astype(np.int16)
    return wave_i16.tobytes()


def _sse_chunk(content: str, *, modality: str | None = "text") -> bytes:
    payload = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "modality": modality,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def _sse_tool_chunk(name: str, arguments: str) -> bytes:
    payload = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "modality": "text",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


class MockVllmOmniRouter:
    """Thread-safe `httpx.MockTransport` handler implementing vLLM-Omni LFM3-Omni wire protocol."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.asr_sequence: list[str] = []
        self.asr_idx: int = 0
        self.router_decision: str = "DIRECT"
        self.query_extract_text: str = "What is the annual fee and lounge benefit on Ubuntu Platinum?"
        self.tool_call_on_next_answer: tuple[str, dict[str, Any]] | None = None
        self.omit_chat_audio: bool = False
        self.tts_mode: str = "raw_pcm"  # "raw_pcm" | "sse"
        self.answer_text: str = "Hello! I am Naledi from Karoo Crest Bank. How can I help today?"
        self.filler_text: str = "Let me check our policy details for you right away."
        self.summary_text: str = "Customer asked about Ubuntu Rewards Platinum annual fee."

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "LiquidAI/LFM3-Omni-2B"}]},
            )

        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        with self.lock:
            self.requests.append((request.url.path, body))

        if request.url.path == "/v1/chat/completions":
            # Critical protocol assertion: never send raw client-only extra_body over httpx;
            # per-stage sampling overrides must use top-level sampling_params_list.
            assert "extra_body" not in body, f"Forbidden raw extra_body in payload: {body.keys()}"
            sp_list = body.get("sampling_params_list")
            assert isinstance(sp_list, list) and len(sp_list) == 1
            assert isinstance(sp_list[0].get("extra_args"), dict)

            messages = body.get("messages") or []
            sys_content = str((messages[0] if messages else {}).get("content") or "")

            if sys_content == "Perform live ASR.":
                with self.lock:
                    if self.asr_idx < len(self.asr_sequence):
                        text = self.asr_sequence[self.asr_idx]
                        self.asr_idx += 1
                    else:
                        text = self.asr_sequence[-1] if self.asr_sequence else ""
                return httpx.Response(
                    200,
                    json={"choices": [{"message": {"role": "assistant", "content": text}}]},
                )

            if "Output strictly" in sys_content:
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"message": {"role": "assistant", "content": self.router_decision}}
                        ]
                    },
                )

            if "Output ONLY one concise standalone question" in sys_content:
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"message": {"role": "assistant", "content": self.query_extract_text}}
                        ]
                    },
                )

            if "compact factual call memory summary" in sys_content:
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"message": {"role": "assistant", "content": self.summary_text}}
                        ]
                    },
                )

            # Streaming omni chat turn (text + optional interleaved audio PCM chunks)
            sys_lower = sys_content.lower()
            is_filler = (
                "already spoke the holding phrase" not in sys_lower
                and ("holding phrase" in sys_lower or "filler" in sys_lower)
            )
            text_out = self.filler_text if is_filler else self.answer_text

            chunks: list[bytes] = []
            with self.lock:
                pending_tool = None if is_filler else self.tool_call_on_next_answer
                if pending_tool is not None:
                    self.tool_call_on_next_answer = None
                omit_audio = self.omit_chat_audio

            if pending_tool is not None:
                t_name, t_args = pending_tool
                chunks.append(_sse_tool_chunk(t_name, json.dumps(t_args)))
                chunks.append(b"data: [DONE]\n\n")
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=b"".join(chunks),
                )

            mid = max(1, len(text_out) // 2)
            chunks.append(_sse_chunk(text_out[:mid], modality="text"))
            chunks.append(_sse_chunk(text_out[mid:], modality="text"))
            if not omit_audio:
                pcm1 = _pcm_frame_bytes(frames=2, amplitude=350)
                # Prefix first chunk with WAV header to verify strip_wav_header_if_present()
                wav_prefixed = pcm16_to_wav_bytes(pcm1, sample_rate=24000)
                pcm2 = _pcm_frame_bytes(frames=2, amplitude=500)
                chunks.append(
                    _sse_chunk(base64.b64encode(wav_prefixed).decode("ascii"), modality="audio")
                )
                chunks.append(
                    _sse_chunk(base64.b64encode(pcm2).decode("ascii"), modality="audio")
                )
            chunks.append(b"data: [DONE]\n\n")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"".join(chunks),
            )

        if request.url.path == "/v1/audio/speech":
            extra_params = body.get("extra_params") or {}
            assert isinstance(extra_params.get("audio_temperature"), float)
            assert isinstance(extra_params.get("audio_top_k"), int)
            assert isinstance(extra_params.get("audio_repetition_penalty"), float)
            assert body.get("response_format") == "pcm"
            assert body.get("stream") is True

            pcm = _pcm_frame_bytes(frames=2, amplitude=420)
            wav_prefixed = pcm16_to_wav_bytes(pcm, sample_rate=24000)
            if self.tts_mode == "sse":
                b64 = base64.b64encode(wav_prefixed).decode("ascii")
                sse_body = (
                    f'data: {json.dumps({"audio": b64})}\n\n'.encode("utf-8")
                    + b"data: [DONE]\n\n"
                )
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=sse_body,
                )
            # Raw binary PCM stream wrapped in WAV header to verify header stripping
            return httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=wav_prefixed,
            )

        return httpx.Response(404, json={"detail": f"Unhandled mock path {request.url.path}"})


def test_vllm_backend_imports_without_mlx() -> None:
    """Importing the app and building a `--backend vllm` config/engine must never import `mlx`."""
    script = """
import sys
import telesales_agent
from voice_agent.backend_factory import build_engine
import voice_agent.server
import voice_agent.session
import voice_agent.vllm_backend

cfg = telesales_agent.build_config(backend="vllm", vllm_base_url="http://127.0.0.1:8091")
assert cfg.engine.backend == "vllm"
assert cfg.engine.model_id == "LiquidAI/LFM3-Omni-2B"
eng = build_engine(cfg.engine)
assert eng.__class__.__name__ == "VllmOmniEngine"
loaded_mlx = [m for m in sys.modules if m == "mlx" or m.startswith("mlx.") or m == "mlx_vlm"]
assert not loaded_mlx, f"Unexpected MLX modules imported on vllm backend: {loaded_mlx}"
print("OK_NO_MLX")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "OK_NO_MLX" in proc.stdout


def test_vllm_engine_load_and_config_endpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("FLAMETREE_AGENT_TOKEN", raising=False)
    router = MockVllmOmniRouter()
    transport = httpx.MockTransport(router.handler)

    cfg = build_config(
        backend="vllm",
        vllm_base_url="http://mock-vllm:8091",
        vllm_api_key="super-secret-vllm-key",
        calls_dir=str(tmp_path / "calls"),
    )
    engine = VllmOmniEngine(cfg.engine, transport=transport)
    engine.load()
    assert engine.is_loaded
    assert engine.available_models == ["LiquidAI/LFM3-Omni-2B"]

    app = create_app(cfg, engine=engine, load_on_startup=False)
    try:
        with TestClient(app) as client:
            health = client.get("/api/health").json()
            assert health["status"] == "ok"
            assert health["backend"] == "vllm"
            assert health["model_loaded"] is True

            pub_cfg = client.get("/api/config").json()
            assert pub_cfg["backend"] == "vllm"
            assert pub_cfg["batch_parallel_mode"] == "vllm_continuous_batch"
            assert pub_cfg["vllm_base_url"] == "http://mock-vllm:8091"
            assert "super-secret-vllm-key" not in json.dumps(pub_cfg)
    finally:
        engine.shutdown()


def test_vllm_live_s2s_listener_diff_echo_endpoint_and_barge_in() -> None:
    router = MockVllmOmniRouter()
    transport = httpx.MockTransport(router.handler)
    settings = EngineSettings(
        backend="vllm",
        model_id="LiquidAI/LFM3-Omni-2B",
        vllm_base_url="http://mock-vllm:8091",
    )
    engine = VllmOmniEngine(settings, transport=transport)
    engine.load()
    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation("System prompt.", tools=None, language="en")
        block = b"\x01\x00" * 10240  # 20,480 bytes = 640 ms @ 16 kHz mono PCM16

        # 1. Echo while agent speaking must be suppressed without opening a user turn
        router.asr_sequence = ["Naledi from Karoo Crest"]
        evs_echo = sess.ingest_pcm_and_listen(
            block,
            agent_speaking=True,
            recent_agent_text="Hello, I am Naledi from Karoo Crest Bank.",
            patience="balanced",
        )
        assert evs_echo == []
        assert not sess.user_speaking

        # 2. Genuine barge-in speech while agent speaking triggers barge_in_detected + user_speech_start
        router.asr_idx = 0
        router.asr_sequence = [
            "Naledi from Karoo Crest Wait tell",
            "Naledi from Karoo Crest Wait tell me the fee",
            "Naledi from Karoo Crest Wait tell me the fee",
        ]
        evs_bi = sess.ingest_pcm_and_listen(
            block,
            agent_speaking=True,
            recent_agent_text="Hello, I am Naledi from Karoo Crest Bank.",
            patience="snappy",
        )
        assert [e["type"] for e in evs_bi] == ["barge_in_detected", "user_speech_start"]
        assert sess.user_speaking

        # 3. Incremental delta followed by silent empty chunks commits the user turn
        evs_delta = sess.ingest_pcm_and_listen(block, patience="snappy")
        assert evs_delta and evs_delta[0]["type"] == "user_speech_delta"

        committed: list[dict[str, Any]] = []
        for _ in range(3):
            committed.extend(sess.ingest_pcm_and_listen(block, patience="snappy"))
        assert any(e["type"] == "user_turn_committed" for e in committed)
        assert not sess.user_speaking
        assert "fee" in sess.latest_user_text().lower()

        # 4. Listener reset works cleanly
        sess.listener.reset_window()
        assert sess.listener.window_seconds == 0.0
    finally:
        engine.shutdown()


def test_vllm_session_greeting_always_delegate_selective_and_compaction() -> None:
    router = MockVllmOmniRouter()
    transport = httpx.MockTransport(router.handler)
    settings = EngineSettings(
        backend="vllm",
        model_id="LiquidAI/LFM3-Omni-2B",
        vllm_base_url="http://mock-vllm:8091",
        compaction_trigger_tokens=1,
    )
    engine = VllmOmniEngine(settings, transport=transport)
    engine.load()
    persona = next(p for p in build_personas() if p.id == "naledi")
    tools = build_tools()
    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation(persona.system_prompt, tools=tools, language="en")

        # Greeting + finalize_greeting_turn
        sess.append_text_user_turn("Start call now.")
        events: list[dict[str, Any]] = []
        cancel_ev = threading.Event()
        out = sess.generate_assistant_turn(
            persona=persona,
            tools=None,
            force_interleaved=True,
            cancel_event=cancel_ev,
            on_event=events.append,
        )
        assert out.kind == "response"
        assert not out.used_tts_fallback
        assert out.audio_frames == 4
        assert out.audio_chunks_sent == 2
        assert any(e["type"] == "text_delta" for e in events)
        assert any(e["type"] == "audio_chunk" for e in events)
        sess.finalize_greeting_turn()
        assert sess.history[0].text == "Hello?"

        # always_delegate: filler + query extract in parallel
        sess.append_text_user_turn("Tell me Platinum FX fees and lounge visits.")
        ad_events: list[dict[str, Any]] = []
        ad_plan = sess.run_filler_and_query_batch(
            cancel_event=threading.Event(),
            on_event=ad_events.append,
        )
        assert ad_plan.decision == "delegate"
        assert "Platinum" in ad_plan.delegation_query
        assert ad_plan.filler_audio_frames == 4
        assert any(e["type"] == "delegation_query_ready" for e in ad_events)
        assert any(e["type"] == "audio_chunk" for e in ad_events)

        # selective_tool DIRECT route
        router.router_decision = "DIRECT"
        dir_events: list[dict[str, Any]] = []
        dir_plan = sess.run_speculative_selective_batch(
            persona=persona,
            tools=tools,
            dynamic_filler_enabled=True,
            cancel_event=threading.Event(),
            on_event=dir_events.append,
        )
        assert dir_plan.decision == "direct"
        assert dir_plan.direct_outcome is not None
        assert dir_plan.direct_outcome.audio_frames == 4
        assert any(e["type"] == "audio_chunk" for e in dir_events)

        # selective_tool DELEGATE route
        router.router_decision = "DELEGATE: What are the Ubuntu Platinum airport lounge rules?"
        del_events: list[dict[str, Any]] = []
        del_plan = sess.run_speculative_selective_batch(
            persona=persona,
            tools=tools,
            dynamic_filler_enabled=True,
            cancel_event=threading.Event(),
            on_event=del_events.append,
        )
        assert del_plan.decision == "delegate"
        assert "airport lounge" in del_plan.delegation_query
        assert del_plan.filler_audio_frames == 4
        assert any(e["type"] == "delegation_query_ready" for e in del_events)

        # Complete delegated turn before next turn
        sess.append_delegated_knowledge_context(
            del_plan.delegation_query,
            "Protea Platinum includes 4 domestic airport lounge visits per year.",
            "flametree_live",
        )
        delegated_out = sess.generate_assistant_turn(
            persona=persona,
            tools=None,
            force_interleaved=True,
            cancel_event=threading.Event(),
            on_event=lambda _e: None,
        )
        assert delegated_out.kind == "response"
        assert sess._pending_filler_text == ""

        # selective_tool CRM tool call route
        router.router_decision = "DIRECT"
        router.tool_call_on_next_answer = (
            "check_pre_qualification",
            {"monthly_income_zar": 35000},
        )
        tc_plan = sess.run_speculative_selective_batch(
            persona=persona,
            tools=tools,
            dynamic_filler_enabled=True,
            cancel_event=threading.Event(),
            on_event=lambda _e: None,
        )
        assert tc_plan.decision == "tool_calls"
        assert tc_plan.direct_outcome is not None
        assert tc_plan.direct_outcome.tool_calls[0]["function"]["name"] == "check_pre_qualification"

        # Context compaction (> 6 turns + compaction_trigger_tokens=1)
        for i in range(4):
            sess.append_text_user_turn(f"Follow-up question {i}")
        assert len(sess.history) > 6
        compacted = sess.maybe_compact_context()
        assert compacted is True
        assert sess.compaction_count == 1
        assert len(sess.history) == 6
    finally:
        engine.shutdown()


@pytest.mark.parametrize("tts_mode", ["raw_pcm", "sse"])
def test_vllm_tts_fallback_raw_pcm_and_sse(tts_mode: str) -> None:
    router = MockVllmOmniRouter()
    router.omit_chat_audio = True
    router.tts_mode = tts_mode
    transport = httpx.MockTransport(router.handler)
    settings = EngineSettings(
        backend="vllm",
        model_id="LiquidAI/LFM3-Omni-2B",
        vllm_base_url="http://mock-vllm:8091",
    )
    engine = VllmOmniEngine(settings, transport=transport)
    engine.load()
    persona = next(p for p in build_personas() if p.id == "naledi")
    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation(persona.system_prompt, tools=None, language="en")
        sess.append_text_user_turn("Hello!")
        events: list[dict[str, Any]] = []
        out = sess.generate_assistant_turn(
            persona=persona,
            tools=None,
            force_interleaved=False,
            cancel_event=threading.Event(),
            on_event=events.append,
        )
        assert out.used_tts_fallback is True
        assert out.audio_frames == 2
        assert out.audio_chunks_sent >= 1
        pcm_chunks = [e["pcm16"] for e in events if e["type"] == "audio_chunk"]
        assert len(pcm_chunks) >= 1
        joined_pcm = b"".join(pcm_chunks)
        # WAV header must have been stripped cleanly -> exact 2 frames * 1920 samples * 2 bytes
        assert len(joined_pcm) == 2 * 1920 * 2
        assert not joined_pcm.startswith(b"RIFF")
    finally:
        engine.shutdown()


def test_vllm_mid_gen_barge_in_rollback_and_session_no_shape_crash() -> None:
    router = MockVllmOmniRouter()
    transport = httpx.MockTransport(router.handler)
    cfg = build_config(backend="vllm", vllm_base_url="http://mock-vllm:8091", calls_dir="")
    engine = VllmOmniEngine(cfg.engine, transport=transport)
    engine.load()

    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent_json: list[dict[str, Any]] = []
            self.sent_bytes: list[bytes] = []

        async def send_json(self, data: dict[str, Any]) -> None:
            self.sent_json.append(data)

        async def send_text(self, data: str) -> None:
            self.sent_json.append(json.loads(data))

        async def send_bytes(self, data: bytes) -> None:
            self.sent_bytes.append(data)

    async def _run() -> None:
        ws = _FakeWebSocket()
        call_session = CallSession(ws, config=cfg, engine=engine)
        call_session.loop = asyncio.get_running_loop()
        pipeline = call_session._ensure_pipeline()
        pipeline.initialize_conversation("Sys prompt", tools=None, language="en")
        pipeline.append_text_user_turn("Tell me about Karoo Crest rewards.")

        interrupted = TurnRecord(
            role="assistant",
            text="We offer great lounge access cashback rewards and travel cover.",
            raw_audio_codes=[None] * 10,  # Python list (vllm_backend representation, no .shape)
        )
        call_session._played_samples_24k = 4 * 1920
        stream_state = {
            "turn_started_ui": True,
            "first_audio_ts": 100.2,
            "first_token_ts": 100.05,
            "chunks_sent": 2,
            "first_chunk_samples": 1920,
            "filler_prefix": "",
            "display_text": interrupted.text,
        }
        turn_ctx = {
            "t_turn_start": 100.0,
            "ttft_ms": 50.0,
            "route_decision": "direct",
            "after_barge_in": True,
        }

        await call_session._handle_mid_gen_barge_in(
            turn_id="ast-mid-1",
            stream_state=stream_state,
            interrupted_turn=interrupted,
            mid_gen_mic_events=[{"type": "barge_in_detected", "heard_text": "Wait stop"}],
            filler_frames=0,
            turn_ctx=turn_ctx,
        )

        # Verify rollback kept a truncated prefix and no AttributeError was raised
        assert pipeline.latest_assistant_text().endswith("...")
        trunc_msgs = [m for m in ws.sent_json if m.get("type") == "assistant_turn_truncated"]
        assert len(trunc_msgs) == 1
        assert isinstance(trunc_msgs[0].get("metrics"), dict)
        assert trunc_msgs[0]["metrics"]["ttft_ms"] == 50.0
        assert call_session.stats.summary()["turn_count"] == 1

    try:
        asyncio.run(_run())
    finally:
        engine.shutdown()


def test_vllm_websocket_full_call_metrics_and_recorder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("FLAMETREE_AGENT_TOKEN", raising=False)
    router = MockVllmOmniRouter()
    router.router_decision = "DIRECT"
    transport = httpx.MockTransport(router.handler)
    calls_dir = tmp_path / "calls"

    cfg = build_config(
        backend="vllm",
        vllm_base_url="http://mock-vllm:8091",
        language="en",
        delegation_mode="selective_tool",
        calls_dir=str(calls_dir),
    )
    engine = VllmOmniEngine(cfg.engine, transport=transport)
    engine.load()

    try:
        app = create_app(cfg, engine=engine, load_on_startup=False)
        with TestClient(app) as client:
            with client.websocket_connect("/ws/call") as ws:
                assert ws.receive_json() == {"type": "call_state", "state": "idle"}
                ws.send_json({"type": "start_call", "language": "en"})

                call_id = ""
                live_ttfts: list[dict[str, Any]] = []
                live_ttfas: list[dict[str, Any]] = []
                turn_ends: list[dict[str, Any]] = []
                audio_bytes_total = 0

                for _ in range(200):
                    msg = ws.receive()
                    if msg.get("bytes") is not None:
                        audio_bytes_total += len(msg["bytes"])
                    elif msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        t = ev.get("type")
                        if t == "call_started":
                            call_id = ev["call_id"]
                        elif t == "live_ttft":
                            live_ttfts.append(ev)
                        elif t == "live_ttfa":
                            live_ttfas.append(ev)
                        elif t == "assistant_turn_end":
                            turn_ends.append(ev)
                        elif t == "delivery_stats" and ev.get("stats", {}).get("turn_count", 0) >= 1:
                            break

                assert call_id.startswith("call-")
                assert audio_bytes_total > 0

                ws.send_json({"type": "user_text", "text": "Hi Naledi, what are the fees?"})
                for _ in range(200):
                    msg = ws.receive()
                    if msg.get("bytes") is not None:
                        audio_bytes_total += len(msg["bytes"])
                    elif msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        t = ev.get("type")
                        if t == "live_ttft":
                            live_ttfts.append(ev)
                        elif t == "live_ttfa":
                            live_ttfas.append(ev)
                        elif t == "assistant_turn_end":
                            turn_ends.append(ev)
                        elif t == "delivery_stats" and ev.get("stats", {}).get("turn_count", 0) >= 2:
                            break

                ws.send_json({"type": "hang_up"})
                for _ in range(40):
                    msg = ws.receive()
                    if msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        if ev.get("type") == "call_state" and ev.get("state") == "idle":
                            break

            assert len(live_ttfts) >= 2
            assert len(live_ttfas) >= 2
            assert len(turn_ends) >= 2
            for tev in turn_ends:
                assert isinstance(tev.get("metrics"), dict)
                assert tev["metrics"]["ttft_ms"] is not None
                assert tev["metrics"]["ttfa_server_ms"] is not None

            detail = client.get(f"/api/calls/{call_id}").json()
            assert detail["metadata"]["status"] == "completed"
            assert detail["metrics"]["turn_count"] >= 2
            ast_entries = [
                e for e in detail["transcript"]["entries"] if e.get("role") == "assistant"
            ]
            assert len(ast_entries) >= 2
            for ae in ast_entries:
                assert ae["metrics"]["ttft_ms"] is not None
                assert ae["metrics"]["ttfa_server_ms"] is not None

            call_dir = calls_dir / call_id
            with wave.open(str(call_dir / "audio/agent_24k.wav"), "rb") as w_out:
                assert w_out.getframerate() == 24000
                assert w_out.getnframes() > 0
            txt = (call_dir / "transcript.txt").read_text(encoding="utf-8")
            assert "TTFT:" in txt
            assert "TTFA:" in txt
    finally:
        engine.shutdown()
