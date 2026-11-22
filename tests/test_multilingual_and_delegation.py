"""Tests for multilingual es-MX/es/en profiles, Flametree HTTP delegation, stats & WebSocket."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from telesales_agent import build_config
from voice_agent.config import EngineSettings, FlametreeSettings
from voice_agent.delegation import FlametreeHttpDelegate
from voice_agent.engine import Lfm3OmniEngine
from voice_agent.languages import get_language_profile, normalize_language_code
from voice_agent.live_pipeline import (
    has_real_speech,
    is_likely_agent_echo,
    is_semantically_complete,
)
from voice_agent.metrics import DeliveryStatsTracker, TurnDeliveryMetrics
from voice_agent.server import create_app


@pytest.fixture(scope="module")
def shared_engine() -> Lfm3OmniEngine:
    engine = Lfm3OmniEngine(EngineSettings())
    engine.load()
    yield engine
    engine.shutdown()


def test_language_profiles_and_spanish_unicode_helpers() -> None:
    assert normalize_language_code("es-mx") == "es-MX"
    assert normalize_language_code("es_MX") == "es-MX"
    assert normalize_language_code("es-ES") == "es"
    assert normalize_language_code("en-ZA") == "en"

    p_mx = get_language_profile("es-MX")
    p_es = get_language_profile("es")
    p_en = get_language_profile("en")
    assert p_mx.tts_accent == "Spanish"
    assert p_es.tts_accent == "Spanish"
    assert p_en.tts_accent == "British English"

    assert has_real_speech("¿Sí, qué cuota mensual?")
    assert not has_real_speech("... ¿?")
    assert is_semantically_complete("¿Cuál es la tasa de interés anual?", "es-MX")
    assert not is_semantically_complete("Quiero saber porque", "es-MX")
    assert not is_likely_agent_echo("espera", "espera un segundo te digo", "es-MX")
    assert is_likely_agent_echo("cuota mensual", "La cuota mensual es noventa y cinco", "es-MX")


def test_flametree_http_delegate_contract_and_fallback() -> None:
    observed: list[tuple[str, str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        content_type = request.headers.get("content-type", "")
        auth = request.headers.get("authorization", "")
        observed.append((request.method, str(request.url), content_type, auth))
        if request.url.path.endswith("/session") and request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "session": {
                        "session_id": "06b02c36-ff54-7e22-8000-a93a3ae7f016",
                        "session_type": "text_tool_calling",
                    }
                },
            )
        if request.url.path.endswith("/run_v2") and request.method == "POST":
            assert "multipart/form-data" in content_type
            return httpx.Response(
                200,
                json={"response": "La cuota mensual es de 145 rand con cero por ciento en divisas."},
            )
        if (
            request.url.path.endswith("/06b02c36-ff54-7e22-8000-a93a3ae7f016")
            and request.method == "DELETE"
        ):
            return httpx.Response(200, json={"result": True})
        return httpx.Response(500, json={"error": "unexpected"})

    settings = FlametreeSettings(
        enabled=True,
        base_url="https://portal.flametree.ai",
        agent_id="1717ee86-0823-4700-aa70-d2b2b1c165bb",
        tenant_id="8470d927-5847-48fc-9a55-d72dd4117ac1",
    )

    async def scenario() -> None:
        client = FlametreeHttpDelegate(
            settings=settings,
            api_token="Bearer test-token-value",
            transport=httpx.MockTransport(handler),
        )
        sid = await client.start_session("call-001", "es-MX")
        assert sid == "06b02c36-ff54-7e22-8000-a93a3ae7f016"
        assert client.current_session_id == "06b02c36-ff54-7e22-8000-a93a3ae7f016"
        res = await client.query("¿Cuánto cuesta Protea Platinum?", "es-MX")
        assert res.source == "flametree_live"
        assert "145 rand" in res.answer
        assert res.session_id == "06b02c36-ff54-7e22-8000-a93a3ae7f016"
        assert client.last_error is None
        await client.close_session()
        assert client.current_session_id is None

    asyncio.run(scenario())

    assert [m for m, *_ in observed] == ["POST", "POST", "DELETE"]
    assert all(auth == "Bearer test-token-value" for *_, auth in observed)
    assert "multipart/form-data" in observed[1][2]

    # Verify fallback on HTTP 503 records last_error for diagnostics
    def failing_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "down"})

    async def fallback_scenario() -> None:
        client = FlametreeHttpDelegate(
            settings=settings,
            api_token="test-token-value",
            transport=httpx.MockTransport(failing_handler),
        )
        res = await client.query("¿Cuál es el cargo en el extranjero de Protea Platinum?", "es-MX")
        assert res.source == "local_kb_fallback"
        assert "cero por ciento" in res.answer.lower()
        assert res.error is not None
        assert client.last_error is not None
        await client.close_session()

    asyncio.run(fallback_scenario())


def test_delivery_stats_tracker_percentiles_and_barge_in() -> None:
    tracker = DeliveryStatsTracker()
    tracker.record_turn(
        TurnDeliveryMetrics(
            turn_id="t1",
            turn_index=1,
            language="es-MX",
            delegation_mode="selective_tool",
            route_decision="direct",
            endpoint_ms=480.0,
            prefill_ms=35.0,
            ttft_ms=55.0,
            ttfa_server_ms=190.0,
            audio_chunks_sent=6,
            first_chunk_samples=1440,
            codec_chunk_avg_ms=7.2,
            audio_duration_sec=2.4,
            gen_rtf=0.42,
            slots_per_sec=68.0,
        )
    )
    assert tracker.update_perceived_ttfa("t1", 238.4)
    tracker.mark_barge_in(cancel_ms=14.2)
    immediate_summary = tracker.summary()
    assert immediate_summary["aggregates"]["barge_in_cancel_ms"]["last"] == 14.2
    after_bi, cancel_ms = tracker.consume_pending_barge_in()
    assert after_bi is True
    assert cancel_ms == 14.2

    tracker.record_turn(
        TurnDeliveryMetrics(
            turn_id="t2",
            turn_index=2,
            language="es-MX",
            delegation_mode="always_delegate",
            route_decision="delegate",
            after_barge_in=after_bi,
            barge_in_cancel_ms=cancel_ms,
            endpoint_ms=510.0,
            prefill_ms=72.0,
            ttft_ms=110.0,
            ttfa_server_ms=260.0,
            ttfa_filler_ms=260.0,
            delegation_ms=580.0,
            delegation_source="flametree_live",
            audio_chunks_sent=12,
            first_chunk_samples=1440,
            codec_chunk_avg_ms=6.9,
            audio_duration_sec=3.6,
            gen_rtf=0.51,
            slots_per_sec=64.0,
        )
    )
    tracker.update_perceived_ttfa("t2", 304.6)

    summary = tracker.summary()
    assert summary["turn_count"] == 2
    assert summary["barge_in_count"] == 1
    agg = summary["aggregates"]
    assert agg["ttfa_server_ms"]["p50"] == 225.0
    assert agg["ttfa_perceived_ms"]["last"] == 304.6
    assert agg["ttfa_after_barge_in_ms"]["last"] == 304.6
    assert agg["barge_in_cancel_ms"]["last"] == 14.2
    assert agg["delegation_ms"]["last"] == 580.0

    # Interrupted mid-gen turn still counts in delivery stats
    tracker.record_turn(
        TurnDeliveryMetrics(
            turn_id="t3",
            turn_index=3,
            language="es-MX",
            delegation_mode="selective_tool",
            route_decision="direct",
            interrupted=True,
            ttfa_server_ms=215.0,
            audio_chunks_sent=3,
            first_chunk_samples=1440,
        )
    )
    int_summary = tracker.summary()
    assert int_summary["turn_count"] == 3
    assert int_summary["recent_turns"][-1]["interrupted"] is True


def test_websocket_streaming_multilingual_switching_and_delegation(
    shared_engine: Lfm3OmniEngine,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Ensure offline fallback works deterministically even if shell exports a token.
    monkeypatch.delenv("FLAMETREE_AGENT_TOKEN", raising=False)
    cfg = build_config(
        language="es-MX",
        delegation_mode="always_delegate",
        calls_dir=str(tmp_path / "calls"),
    )
    app = create_app(cfg, engine=shared_engine, load_on_startup=False)
    client = TestClient(app)

    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "no-store" in res_root.headers.get("cache-control", "")

    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
    assert "no-store" in res_js.headers.get("cache-control", "")

    res_cfg = client.get("/api/config")
    assert res_cfg.status_code == 200
    pub = res_cfg.json()
    assert pub["supported_languages"] == ["es-MX", "es", "en"]
    assert pub["first_chunk_frames"] == 1
    assert pub["default_language"] == "es-MX"
    assert pub["flametree"]["tenant_name"] == "Denis Demo"

    with client.websocket_connect("/ws/call") as ws:
        # Consume initial handshake frames (call_state, delegation_status, delivery_stats)
        init_msgs = [ws.receive_json() for _ in range(3)]
        assert any(m["type"] == "delegation_status" for m in init_msgs)

        # Switch dynamically to es-MX + always_delegate + user_text
        ws.send_json(
            {
                "type": "set_config",
                "language": "es-MX",
                "delegation_mode": "always_delegate",
                "dynamic_filler": True,
            }
        )
        ws.send_json(
            {
                "type": "user_text",
                "text": "¿Cuál es la comisión en el extranjero de la tarjeta Protea Platinum?",
            }
        )

        json_events: list[dict] = []
        binary_chunks_before_audio_end: list[int] = []
        saw_audio_end = False

        for _ in range(600):
            msg = ws.receive()
            if msg.get("bytes") is not None:
                assert not saw_audio_end, "Binary audio must stream BEFORE audio_end!"
                binary_chunks_before_audio_end.append(len(msg["bytes"]))
                if len(binary_chunks_before_audio_end) == 1:
                    turn_start = next(
                        (e for e in json_events if e.get("type") == "assistant_turn_start"),
                        None,
                    )
                    if turn_start:
                        ws.send_json(
                            {"type": "playback_started", "turn_id": turn_start["turn_id"]}
                        )
            elif msg.get("text") is not None:
                ev = json.loads(msg["text"])
                json_events.append(ev)
                if ev.get("type") == "audio_end":
                    saw_audio_end = True
                if ev.get("type") == "delivery_stats" and saw_audio_end:
                    # Wait for perceived_ms update if playback_started is still in flight
                    turns = ev.get("stats", {}).get("recent_turns", [])
                    if turns and turns[-1].get("ttfa_perceived_ms") is not None:
                        break
                    if turns and turns[-1].get("ttfa_perceived_ms") is None:
                        # Trigger at most one more read for the playback_started push
                        nxt = ws.receive()
                        if nxt.get("text"):
                            nev = json.loads(nxt["text"])
                            json_events.append(nev)
                        break

        assert saw_audio_end
        assert len(binary_chunks_before_audio_end) >= 2, "Expected streamed multi-chunk PCM"
        # Chunk 1 must be Frame 1 (1440 samples * 2 bytes = 2880 bytes) or filler Frame 1!
        assert binary_chunks_before_audio_end[0] == 2880

        del_events = [e for e in json_events if e.get("type") == "delegation_event"]
        assert len(del_events) == 1
        assert del_events[0]["source"] == "local_kb_fallback"
        assert del_events[0]["language"] == "es-MX"

        final_stats = [e for e in json_events if e.get("type") == "delivery_stats"][-1]["stats"]
        assert any(e.get("type") == "live_ttfa" for e in json_events)
        assert final_stats["turn_count"] == 1
        turn = final_stats["recent_turns"][-1]
        assert turn["route_decision"] == "delegate"
        assert turn["first_chunk_samples"] == 1440
        assert turn["audio_chunks_sent"] >= 2
        assert turn["ttfa_perceived_ms"] is not None

        # Live switch to English + selective_tool mid-call
        ws.send_json(
            {
                "type": "set_config",
                "language": "en",
                "delegation_mode": "selective_tool",
            }
        )
        ack = None
        for _ in range(8):
            nxt = ws.receive_json()
            if nxt.get("type") == "delegation_status":
                ack = nxt
                break
        assert ack is not None
        assert ack["language"] == "en"
        assert ack["mode"] == "selective_tool"
