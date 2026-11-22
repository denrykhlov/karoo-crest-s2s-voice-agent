"""Tests for per-call call_id generation, artifact persistence, WAV recording, and /api/calls."""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from telesales_agent import build_config
from voice_agent.call_recorder import (
    CallRecorder,
    generate_call_id,
    list_recorded_calls,
    load_recorded_call,
)
from voice_agent.config import EngineSettings
from voice_agent.engine import Lfm3OmniEngine
from voice_agent.metrics import DeliveryStatsTracker, TurnDeliveryMetrics
from voice_agent.server import create_app


def test_call_recorder_unit_artifacts_wav_latches_and_security(tmp_path: Path) -> None:
    cid1 = generate_call_id()
    cid2 = generate_call_id()
    assert cid1 != cid2
    assert re.fullmatch(r"call-\d{8}-\d{6}-[0-9a-f]{8}", cid1)

    secret_token = "super-secret-flametree-token-12345"
    base_dir = tmp_path / "calls"
    rec = CallRecorder(
        call_id=cid1,
        base_dir=base_dir,
        initial_metadata={
            "app_title": "Test Studio",
            "company_name": "Karoo Crest Bank",
            "model_id": "LiquidAI/LFM3-Omni-2B-MLX-8bit",
            "initial_settings": {
                "persona_id": "naledi",
                "persona_name": "Naledi Mokoena",
                "language": "en",
                "delegation_mode": "selective_tool",
            },
            "client_info": {
                "host": "127.0.0.1",
                "authorization": f"Bearer {secret_token}",
            },
            "flametree": {
                "enabled": True,
                "live": True,
                "tenant_name": "Denis Demo",
                "api_token": secret_token,
            },
        },
        record_audio=True,
    )

    rec.record_event("call_started", {"language": "en", "api_token": secret_token})
    rec.record_config_change(
        {"language": "es-MX"},
        {
            "persona_id": "naledi",
            "persona_name": "Naledi Mokoena",
            "language": "es-MX",
            "delegation_mode": "always_delegate",
        },
        rebuild_needed=True,
    )

    # Latch Flametree session_id, then ensure clearing on hangup does not erase it.
    rec.update_flametree_status(
        live=True,
        session_id="ft-sess-abc999",
        session_url="https://portal.flametree.ai/sessions/ft-sess-abc999?tenant_id=t1",
        last_error=None,
    )
    rec.update_flametree_status(
        live=True,
        session_id=None,
        session_url="",
        last_error=None,
    )

    # Write caller (16kHz, 0.5s = 8000 samples) & agent (24kHz, 0.25s = 6000 samples) PCM16 audio
    caller_pcm = (np.zeros(8000, dtype=np.int16)).tobytes()
    agent_pcm = (np.ones(6000, dtype=np.int16) * 123).tobytes()
    rec.record_caller_audio(caller_pcm)
    rec.record_agent_audio(agent_pcm)

    # Upsert transcript entries & simulate barge-in truncation update
    rec.record_transcript_entry(
        turn_id="usr-1",
        role="user",
        speaker="Caller",
        text="Hola, quiero info de la tarjeta Platinum",
        language="es-MX",
    )
    rec.record_transcript_entry(
        turn_id="ast-1",
        role="assistant",
        speaker="Naledi Mokoena",
        text="Claro, te cuento todo sobre Protea Platinum y sus beneficios completos.",
        language="es-MX",
        route_decision="delegate",
        metrics={"ttft_ms": 42.3, "ttfa_server_ms": 145.0},
    )
    rec.record_transcript_entry(
        turn_id="ast-1",
        role="assistant",
        speaker="Naledi Mokoena",
        text="Claro, te cuento...",
        language="es-MX",
        truncated=True,
        interrupted=True,
    )
    rec.update_transcript_metrics("ast-1", {"ttfa_perceived_ms": 180.2})

    stats = DeliveryStatsTracker()
    stats.mark_barge_in(cancel_ms=11.4)
    stats.record_turn(
        TurnDeliveryMetrics(
            turn_id="ast-1",
            turn_index=1,
            language="es-MX",
            delegation_mode="always_delegate",
            route_decision="delegate",
            interrupted=True,
            ttfa_server_ms=145.0,
            ttfa_perceived_ms=180.2,
            audio_chunks_sent=3,
            first_chunk_samples=1440,
        )
    )
    crm_state = {
        "qualification": None,
        "repayment": None,
        "application": None,
        "callback": None,
        "tool_log": [{"tool": "check_pre_qualification", "result": {"ok": True}}],
    }
    delegation_log = [{"turn_id": "ast-1", "query": "Protea Platinum fee", "source": "flametree_live"}]
    ctx_stats = {"tokens": 512, "compactions": 1, "listener_resets": 2}

    final_meta = rec.finalize(
        end_reason="hang_up",
        metrics_report=stats.summary(include_all_turns=True),
        crm_state=crm_state,
        delegation_log=delegation_log,
        context_stats=ctx_stats,
    )

    call_dir = base_dir / cid1
    assert call_dir.is_dir()
    for rel in (
        "metadata.json",
        "trace.jsonl",
        "transcript.json",
        "transcript.txt",
        "metrics.json",
        "crm_and_delegation.json",
        "audio/caller_16k.wav",
        "audio/agent_24k.wav",
    ):
        assert (call_dir / rel).is_file(), f"Missing artifact: {rel}"

    assert final_meta["status"] == "completed"
    assert final_meta["end_reason"] == "hang_up"
    assert final_meta["flametree"]["session_id"] == "ft-sess-abc999"
    assert final_meta["counts"]["user_turns"] == 1
    assert final_meta["counts"]["assistant_turns"] == 1
    assert final_meta["counts"]["interrupted_turns"] == 1
    assert final_meta["counts"]["barge_in_count"] == 1
    assert final_meta["counts"]["tool_call_count"] == 1
    assert final_meta["counts"]["delegation_count"] == 1
    assert final_meta["counts"]["compaction_count"] == 1
    assert final_meta["counts"]["listener_resets"] == 2
    assert len(final_meta["config_changes"]) == 1

    # Verify WAV headers and exact sample counts
    with wave.open(str(call_dir / "audio/caller_16k.wav"), "rb") as w_in:
        assert w_in.getnchannels() == 1
        assert w_in.getsampwidth() == 2
        assert w_in.getframerate() == 16000
        assert w_in.getnframes() == 8000
    with wave.open(str(call_dir / "audio/agent_24k.wav"), "rb") as w_out:
        assert w_out.getnchannels() == 1
        assert w_out.getsampwidth() == 2
        assert w_out.getframerate() == 24000
        assert w_out.getnframes() == 6000

    # Verify transcript upsert replaced full text with interrupted truncated text
    # and rendered per-message TTFT/TTFA/Perceived metrics inline
    txt_content = (call_dir / "transcript.txt").read_text(encoding="utf-8")
    assert "Claro, te cuento..." in txt_content
    assert "[interrupted/truncated]" in txt_content
    assert "(TTFT: 42 ms | TTFA: 145 ms | Perceived: 180 ms)" in txt_content
    assert "sus beneficios completos" not in txt_content

    # Secret must never appear in any persisted artifact
    for artifact in call_dir.rglob("*"):
        if artifact.is_file() and artifact.suffix in (".json", ".jsonl", ".txt"):
            assert secret_token not in artifact.read_text(encoding="utf-8")

    # Verify inspection helpers and path traversal protection
    summaries = list_recorded_calls(base_dir)
    assert len(summaries) == 1
    assert summaries[0]["call_id"] == cid1

    bundle = load_recorded_call(base_dir, cid1)
    assert bundle is not None
    assert bundle["metrics"]["turn_count"] == 1
    assert len(bundle["metrics"]["turns"]) == 1
    ast_transcript = [
        e for e in bundle["transcript"]["entries"] if e["turn_id"] == "ast-1"
    ][0]
    assert ast_transcript["metrics"]["ttft_ms"] == 42.3
    assert ast_transcript["metrics"]["ttfa_server_ms"] == 145.0
    assert ast_transcript["metrics"]["ttfa_perceived_ms"] == 180.2
    assert bundle["trace_count"] >= 2
    assert bundle["trace_events"][-1]["event"] == "call_ended"

    with pytest.raises(ValueError):
        load_recorded_call(base_dir, "../etc/passwd")
    with pytest.raises(ValueError):
        load_recorded_call(base_dir, "a/b")
    with pytest.raises(ValueError):
        load_recorded_call(base_dir, "..")
    assert load_recorded_call(base_dir, "call-does-not-exist") is None


def test_websocket_call_persists_folder_and_api_calls_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("FLAMETREE_AGENT_TOKEN", raising=False)
    calls_dir = tmp_path / "calls"
    cfg = build_config(
        language="en",
        delegation_mode="selective_tool",
        calls_dir=str(calls_dir),
    )
    engine = Lfm3OmniEngine(EngineSettings())
    engine.load()
    try:
        app = create_app(cfg, engine=engine, load_on_startup=False)
        with TestClient(app) as client:
            with client.websocket_connect("/ws/call") as ws:
                assert ws.receive_json() == {"type": "call_state", "state": "idle"}
                ws.send_json({"type": "start_call", "language": "en"})

                call_id = ""
                artifacts_dir = ""
                greeting_done = False
                live_ttfts: list[dict] = []
                turn_end_events: list[dict] = []
                for _ in range(400):
                    msg = ws.receive()
                    if msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        if ev.get("type") == "call_started":
                            call_id = ev["call_id"]
                            artifacts_dir = ev["artifacts_dir"]
                        elif ev.get("type") == "live_ttft":
                            live_ttfts.append(ev)
                        elif ev.get("type") in ("assistant_turn_end", "assistant_turn_truncated"):
                            turn_end_events.append(ev)
                        elif ev.get("type") == "delivery_stats" and ev.get("stats", {}).get("turn_count", 0) >= 1:
                            greeting_done = True
                            break

                assert call_id.startswith("call-")
                assert Path(artifacts_dir) == calls_dir.resolve() / call_id
                assert greeting_done

                ws.send_json({"type": "user_text", "text": "What is the Ubuntu Gold fee?"})
                for _ in range(400):
                    msg = ws.receive()
                    if msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        if ev.get("type") == "live_ttft":
                            live_ttfts.append(ev)
                        elif ev.get("type") in ("assistant_turn_end", "assistant_turn_truncated"):
                            turn_end_events.append(ev)
                        elif ev.get("type") == "delivery_stats" and ev.get("stats", {}).get("turn_count", 0) >= 2:
                            break

                assert len(live_ttfts) >= 2
                assert all(ev.get("ttft_ms", -1) >= 0 for ev in live_ttfts)
                assert len(turn_end_events) >= 2
                for tev in turn_end_events:
                    assert isinstance(tev.get("metrics"), dict)
                    assert "ttft_ms" in tev["metrics"]
                    assert "ttfa_server_ms" in tev["metrics"]

                ws.send_json({"type": "hang_up"})
                for _ in range(40):
                    msg = ws.receive()
                    if msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        if ev.get("type") == "call_state" and ev.get("state") == "idle":
                            break

            call_dir = calls_dir / call_id
            assert call_dir.is_dir()
            for rel in (
                "metadata.json",
                "trace.jsonl",
                "transcript.json",
                "transcript.txt",
                "metrics.json",
                "crm_and_delegation.json",
                "audio/agent_24k.wav",
            ):
                assert (call_dir / rel).is_file(), f"Missing {rel}"

            # Check WAV readable
            with wave.open(str(call_dir / "audio/agent_24k.wav"), "rb") as w_out:
                assert w_out.getframerate() == 24000
                assert w_out.getnframes() > 24000

            # Check HTTP inspection routes
            res_list = client.get("/api/calls")
            assert res_list.status_code == 200
            list_data = res_list.json()
            assert any(item["call_id"] == call_id for item in list_data["calls"])

            res_detail = client.get(f"/api/calls/{call_id}")
            assert res_detail.status_code == 200
            detail = res_detail.json()
            assert detail["call_id"] == call_id
            assert detail["metadata"]["status"] == "completed"
            assert detail["metadata"]["end_reason"] == "hang_up"
            assert detail["metrics"]["turn_count"] >= 2
            assert len(detail["metrics"]["turns"]) >= 2
            assert len(detail["transcript"]["entries"]) >= 3
            ast_entries = [
                e for e in detail["transcript"]["entries"] if e.get("role") == "assistant"
            ]
            assert len(ast_entries) >= 2
            for ast_e in ast_entries:
                assert isinstance(ast_e.get("metrics"), dict)
                assert ast_e["metrics"].get("ttft_ms") is not None
                assert ast_e["metrics"].get("ttfa_server_ms") is not None
            txt_persisted = (call_dir / "transcript.txt").read_text(encoding="utf-8")
            assert "TTFT:" in txt_persisted
            assert "TTFA:" in txt_persisted
            assert detail["trace_count"] >= 5
            assert detail["trace_events"][-1]["event"] == "call_ended"

            res_bad = client.get("/api/calls/..%2Fsecret")
            assert res_bad.status_code in (400, 404)
            res_bad2 = client.get("/api/calls/bad/path")
            assert res_bad2.status_code == 400
            res_missing = client.get("/api/calls/call-20990101-000000-00000000")
            assert res_missing.status_code == 404
    finally:
        engine.shutdown()
