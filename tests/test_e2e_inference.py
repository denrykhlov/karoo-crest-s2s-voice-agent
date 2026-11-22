"""End-to-end WebSocket & HTTP smoke tests for the Pure-S2S voice agent."""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from telesales_agent import build_config
from voice_agent.engine import Lfm3OmniEngine
from voice_agent.server import create_app


def _synth_16k(engine: Lfm3OmniEngine, text: str) -> np.ndarray:
    tok = engine.tokenizer
    model = engine.model
    codec = engine.codec
    cfg = model.config
    prompt = tok.apply_chat_template(
        [
            {
                "role": "system",
                "content": "Perform TTS in British English using a male voice.",
            },
            {"role": "user", "content": text},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = mx.array([tok.encode(prompt, add_special_tokens=False)], dtype=mx.int32)
    frames = []
    for _, c in model.stream_slots(
        ids,
        max_tokens=200,
        generation_mode="audio",
        audio_temperature=0.4,
        audio_top_k=128,
        audio_repetition_penalty=1.2,
    ):
        if c is not None:
            cn = np.asarray(c, dtype=np.int32)
            if int(cn[0]) == cfg.audio_eos_code:
                break
            frames.append(cn)
    wav24 = np.asarray(codec.decode(mx.array(np.stack(frames, axis=-1))), dtype=np.float32)
    n16 = int(round(wav24.shape[0] * 16000.0 / 24000.0))
    return np.interp(
        np.linspace(0.0, 1.0, n16, endpoint=False),
        np.linspace(0.0, 1.0, wav24.shape[0], endpoint=False),
        wav24,
    ).astype(np.float32)


@pytest.fixture(scope="module")
def server_env(tmp_path_factory: pytest.TempPathFactory):
    cfg = build_config(calls_dir=str(tmp_path_factory.mktemp("e2e_calls")))
    engine = Lfm3OmniEngine(cfg.engine)
    engine.load()
    yield cfg, engine
    engine.shutdown()


def test_fastapi_endpoints_and_websocket_pure_s2s_audio_stream(server_env) -> None:
    cfg, engine = server_env
    app = create_app(cfg, engine=engine, load_on_startup=False)

    # Synthesize a spoken user question BEFORE opening WS so we can stream real 16kHz PCM
    q_wav = _synth_16k(engine, "What is the monthly fee on the Ubuntu Gold card?")
    rng = np.random.default_rng(3)
    sil_pre = rng.normal(0.0, 0.0003, int(1.0 * 16000)).astype(np.float32)
    sil_post = rng.normal(0.0, 0.0003, int(2.2 * 16000)).astype(np.float32)
    full_stream = np.concatenate([sil_pre, q_wav, sil_post])
    pcm_all = (np.clip(full_stream, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    with TestClient(app) as client:
        r_idx = client.get("/")
        assert r_idx.status_code == 200
        assert "Karoo Crest Bank" in r_idx.text

        r_js1 = client.get("/static/capture-worklet.js")
        assert r_js1.status_code == 200
        assert "javascript" in r_js1.headers.get("content-type", "")

        r_js2 = client.get("/static/playback-worklet.js")
        assert r_js2.status_code == 200
        assert "javascript" in r_js2.headers.get("content-type", "")

        r_health = client.get("/api/health")
        assert r_health.status_code == 200
        assert r_health.json()["model_loaded"] is True

        r_cfg = client.get("/api/config")
        assert r_cfg.status_code == 200
        jcfg = r_cfg.json()
        assert jcfg["pipeline_mode"] == "pure_s2s_streaming"
        assert jcfg["target_call_duration_sec"] == 300
        assert len(jcfg["personas"]) == 2

        with client.websocket_connect("/ws/call") as ws:
            first_msg = ws.receive_json()
            assert first_msg == {"type": "call_state", "state": "idle"}

            # 1. Test start_call greeting
            ws.send_json(
                {
                    "type": "start_call",
                    "persona_id": "naledi",
                    "tools_enabled": True,
                    "turn_patience": "balanced",
                    "input_mode": "live",
                }
            )
            greeting_bytes = 0
            greeting_text = ""
            for _ in range(300):
                m = ws.receive()
                if m.get("bytes") is not None:
                    greeting_bytes += len(m["bytes"])
                elif m.get("text") is not None:
                    p = json.loads(m["text"])
                    if p["type"] == "assistant_turn_end":
                        greeting_text = p["text"]
                    elif p["type"] == "metrics":
                        break
            assert greeting_bytes > 24000
            assert len(greeting_text.strip()) > 10

            ws.send_json({"type": "playback_done", "played_samples": greeting_bytes // 2})

            # 2. Stream continuous 16 kHz mic PCM bytes (silence -> speech -> silence)
            # No external VAD: pure S2S detects speech, streams live caption, and replies!
            chunk_bytes = 10240  # 320 ms chunks
            for off in range(0, len(pcm_all), chunk_bytes):
                ws.send_bytes(pcm_all[off : off + chunk_bytes])

            user_caption = ""
            reply_text = ""
            reply_bytes = 0
            saw_context_stats = False
            for _ in range(400):
                m = ws.receive()
                if m.get("bytes") is not None:
                    reply_bytes += len(m["bytes"])
                elif m.get("text") is not None:
                    p = json.loads(m["text"])
                    if p["type"] in ("user_turn", "user_transcript_update"):
                        user_caption = p["text"]
                    elif p["type"] == "context_stats":
                        saw_context_stats = True
                    elif p["type"] == "assistant_turn_end":
                        reply_text = p["text"]
                    elif p["type"] == "metrics":
                        break

            print(f"\n[WS Pure-S2S] user_caption={user_caption!r} reply={reply_text!r} bytes={reply_bytes}")
            assert saw_context_stats
            assert "gold" in user_caption.lower() or "fee" in user_caption.lower()
            assert reply_bytes > 24000
            assert "95" in reply_text or "gold" in reply_text.lower()
