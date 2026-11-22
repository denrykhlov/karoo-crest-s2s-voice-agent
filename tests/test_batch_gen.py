"""Tests for MLX B=2 heterogeneous parallel batch generation, cache cloning & Frame-1 streaming."""

from __future__ import annotations

import threading
import time

import mlx.core as mx
import numpy as np
import pytest

from telesales_agent import build_config, build_personas, build_tools
from voice_agent.audio_stream import (
    clone_lfm3_cache,
    extract_lfm3_cache,
    merge_lfm3_caches,
)
from voice_agent.config import EngineSettings
from voice_agent.engine import Lfm3OmniEngine
from voice_agent.languages import (
    build_localized_greeting_trigger,
    build_localized_system_prompt,
)
from voice_agent.live_pipeline import StreamingConversationSession


@pytest.fixture(scope="module")
def shared_engine() -> Lfm3OmniEngine:
    engine = Lfm3OmniEngine(EngineSettings())
    engine.load()
    yield engine
    engine.shutdown()


def test_clone_merge_extract_past_rotation_exactness(
    shared_engine: Lfm3OmniEngine,
) -> None:
    model = shared_engine.model
    tok = shared_engine.tokenizer
    bb = model.language_model.model

    # Prefill >270 tokens so RotatingKVCache(max_size=256) rotates cleanly.
    long_text = " ".join(["Karoo Crest Bank Ubuntu Rewards"] * 45)
    base_ids = tok.encode(long_text, add_special_tokens=False)
    assert len(base_ids) > 270

    base = model.make_cache()
    bb(None, base, bb.embed_tokens(mx.array([base_ids], dtype=mx.int32)))

    c0 = clone_lfm3_cache(base)
    c1 = clone_lfm3_cache(base)
    ids0 = tok.encode("Summarize benefits briefly.", add_special_tokens=False)
    ids1 = tok.encode(
        "Output ONLY a single concise standalone question in Mexican Spanish.",
        add_special_tokens=False,
    )
    h0 = bb(None, c0, bb.embed_tokens(mx.array([ids0], dtype=mx.int32)))[:, -1]
    h1 = bb(None, c1, bb.embed_tokens(mx.array([ids1], dtype=mx.int32)))[:, -1]
    mx.eval(h0, h1)

    # Independent B=1 reference rollout for row 0
    ref0 = clone_lfm3_cache(c0)
    ref_toks: list[int] = []
    hr = h0
    for _ in range(8):
        tid = int(mx.argmax(bb.embed_tokens.as_linear(hr).astype(mx.float32), axis=-1).item())
        ref_toks.append(tid)
        hr = bb(None, ref0, bb.embed_tokens(mx.array([[tid]], dtype=mx.int32)))[:, -1]

    # Batched B=2 unequal rollout + mid-stream extract(0)
    bc = merge_lfm3_caches([c0, c1])
    hb = mx.concatenate([h0, h1], axis=0)
    b_toks: list[int] = []
    for step in range(4):
        t0 = int(mx.argmax(bb.embed_tokens.as_linear(hb[0:1]).astype(mx.float32), axis=-1).item())
        t1 = int(mx.argmax(bb.embed_tokens.as_linear(hb[1:2]).astype(mx.float32), axis=-1).item())
        b_toks.append(t0)
        hb = bb(None, bc, bb.embed_tokens(mx.array([[t0], [t1]], dtype=mx.int32)))[:, -1]

    ec0 = extract_lfm3_cache(bc, 0)
    he0 = hb[0:1]
    for _ in range(4):
        t0 = int(mx.argmax(bb.embed_tokens.as_linear(he0).astype(mx.float32), axis=-1).item())
        b_toks.append(t0)
        he0 = bb(None, ec0, bb.embed_tokens(mx.array([[t0]], dtype=mx.int32)))[:, -1]

    assert ref_toks == b_toks


@pytest.mark.parametrize(
    "lang,question",
    [
        ("es-MX", "¿Cuál es la cuota mensual y comisión en el extranjero de Protea Platinum?"),
        ("en", "What is the foreign exchange fee and airport lounge access on Protea Platinum?"),
    ],
)
def test_batch_filler_and_query_parallel_execution(
    shared_engine: Lfm3OmniEngine,
    lang: str,
    question: str,
) -> None:
    session = StreamingConversationSession(
        shared_engine.model,
        shared_engine.tokenizer,
        shared_engine.codec,
        EngineSettings(),
    )
    persona = build_personas()[0]
    prompt = build_localized_system_prompt(
        persona.system_prompt,
        lang_code=lang,
        persona_name=persona.name,
        voice_gender=persona.voice_gender,
        delegation_mode="always_delegate",
    )
    session.initialize_conversation(prompt, tools=None, language=lang)
    session.append_text_user_turn(question)
    toks_before = session.conv_tokens

    events: list[tuple[float, dict]] = []
    t0 = time.perf_counter()

    def on_ev(ev: dict) -> None:
        events.append(((time.perf_counter() - t0) * 1000.0, ev))

    plan = session.run_filler_and_query_batch(threading.Event(), on_ev)
    total_ms = (time.perf_counter() - t0) * 1000.0

    print(
        f"\n[Batch B=2 {lang}] query={plan.delegation_query!r} | filler={plan.filler_text!r} "
        f"| chunks={plan.audio_chunks_sent} first_samples={plan.first_chunk_samples} "
        f"| ttfa_filler={plan.ttfa_filler_ms:.1f}ms total={total_ms:.1f}ms"
    )
    assert plan.decision == "delegate"
    assert len(plan.delegation_query.strip()) >= 8
    q_low = plan.delegation_query.lower()
    assert "not specified" not in q_low
    assert "$200" not in plan.delegation_query
    assert "protea" in q_low or "platinum" in q_low
    assert plan.filler_audio_frames >= 1
    assert plan.audio_chunks_sent >= 1
    assert plan.first_chunk_samples == 1440

    query_events = [(ts, ev) for ts, ev in events if ev.get("type") == "delegation_query_ready"]
    assert len(query_events) == 1
    assert query_events[0][0] <= total_ms + 5.0
    # Base conv_cache must stay untouched (unpolluted by filler prompt).
    assert session.conv_tokens == toks_before


def test_speculative_selective_routing_direct_and_delegate(
    shared_engine: Lfm3OmniEngine,
) -> None:
    cfg = build_config()
    session = StreamingConversationSession(
        shared_engine.model,
        shared_engine.tokenizer,
        shared_engine.codec,
        cfg.engine,
    )
    persona = cfg.get_persona("naledi")
    tools = build_tools()

    # 1. Simple greeting in Spanish -> DIRECT with immediate Frame-1 speech streaming
    session.initialize_conversation(
        build_localized_system_prompt(
            persona.system_prompt, "es-MX", persona.name, persona.voice_gender, "selective_tool"
        ),
        tools=tools,
        language="es-MX",
    )
    session.append_text_user_turn("Hola Naledi, buenos días, sí tengo un minuto.")
    events_direct: list[dict] = []
    plan_direct = session.run_speculative_selective_batch(
        persona=persona,
        tools=tools,
        dynamic_filler_enabled=True,
        cancel_event=threading.Event(),
        on_event=events_direct.append,
    )
    print(f"\n[Speculative DIRECT es-MX] decision={plan_direct.decision} chunks={plan_direct.audio_chunks_sent} text={plan_direct.direct_outcome.text if plan_direct.direct_outcome else ''!r}")
    assert plan_direct.decision == "direct"
    assert plan_direct.direct_outcome is not None
    assert plan_direct.direct_outcome.audio_frames > 5
    assert plan_direct.audio_chunks_sent >= 2

    # 2. Complex policy/fee question -> DELEGATE with B=2 filler + query
    session.append_text_user_turn(
        "¿Cuál es la comisión por compras en el extranjero y los accesos a salas VIP de Protea Platinum?"
    )
    events_del: list[dict] = []
    plan_del = session.run_speculative_selective_batch(
        persona=persona,
        tools=tools,
        dynamic_filler_enabled=True,
        cancel_event=threading.Event(),
        on_event=events_del.append,
    )
    print(f"[Speculative DELEGATE es-MX] decision={plan_del.decision} query={plan_del.delegation_query!r} filler={plan_del.filler_text!r} chunks={plan_del.audio_chunks_sent}")
    assert plan_del.decision == "delegate"
    assert len(plan_del.delegation_query.strip()) >= 8
    assert plan_del.audio_chunks_sent >= 1
    assert any(ev.get("type") == "delegation_query_ready" for ev in events_del)


def test_selective_router_delegates_spoken_spanish_audio_question(
    shared_engine: Lfm3OmniEngine,
) -> None:
    """Verify real Spanish speech audio routes to DELEGATE even after greeting history."""
    cfg = build_config(language="es-MX", delegation_mode="selective_tool")
    persona = cfg.get_persona("naledi")
    tools = build_tools()

    prompt = shared_engine.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Perform TTS in Spanish using a male voice."},
            {
                "role": "user",
                "content": "¿Cuál es la cuota mensual y los requisitos de FICA y POPIA de la tarjeta Protea Platinum?",
            },
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = mx.array(
        [shared_engine.tokenizer.encode(prompt, add_special_tokens=False)],
        dtype=mx.int32,
    )
    frames = []
    for _t, c in shared_engine.model.stream_slots(
        ids, max_tokens=200, generation_mode="audio", audio_temperature=0.3
    ):
        if c is not None:
            arr = np.asarray(c, dtype=np.int32)
            if int(arr[0]) == shared_engine.model.config.audio_eos_code:
                break
            frames.append(arr)
    codes = np.stack(frames, axis=-1)
    wav24 = np.asarray(shared_engine.codec.decode(mx.array(codes)), dtype=np.float32).reshape(-1)
    n16 = int(round(len(wav24) * 16000 / 24000))
    wav16 = np.interp(np.linspace(0, len(wav24) - 1, n16), np.arange(len(wav24)), wav24)
    pcm_user = (np.clip(wav16, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    sil_trail = np.zeros(int(16000 * 3.5), dtype="<i2").tobytes()

    session = StreamingConversationSession(
        shared_engine.model,
        shared_engine.tokenizer,
        shared_engine.codec,
        cfg.engine,
    )
    session.initialize_conversation(
        build_localized_system_prompt(
            persona.system_prompt, "es-MX", persona.name, persona.voice_gender, "selective_tool"
        ),
        tools=tools,
        language="es-MX",
    )
    session.append_text_user_turn(build_localized_greeting_trigger("es-MX", persona.name))
    session.generate_assistant_turn(persona, None, True, threading.Event(), lambda _e: None)

    evs = session.ingest_pcm_and_listen(pcm_user + sil_trail, agent_speaking=False, patience="balanced")
    assert any(e.get("type") == "user_turn_committed" for e in evs)

    plan_events: list[dict] = []
    plan = session.run_speculative_selective_batch(
        persona=persona,
        tools=tools,
        dynamic_filler_enabled=True,
        cancel_event=threading.Event(),
        on_event=plan_events.append,
    )
    print(
        f"\n[Spoken es-MX Audio Router] decision={plan.decision} "
        f"query={plan.delegation_query!r} first_samples={plan.first_chunk_samples}"
    )
    assert plan.decision == "delegate"
    assert len(plan.delegation_query.strip()) >= 8
    assert plan.first_chunk_samples == 1440
    assert any(e.get("type") == "delegation_query_ready" for e in plan_events)
