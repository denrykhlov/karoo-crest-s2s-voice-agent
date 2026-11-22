"""Integration tests for Pure-S2S always-streaming pipeline, barge-in rollback, and 5-min call lifecycle."""

from __future__ import annotations

import json
import threading
import time

import mlx.core as mx
import numpy as np
import pytest

from telesales_agent import build_config
from voice_agent.config import EngineSettings
from voice_agent.engine import Lfm3OmniEngine
from voice_agent.languages import (
    build_localized_greeting_trigger,
    build_localized_system_prompt,
)
from voice_agent.live_pipeline import (
    StreamingConversationSession,
    is_likely_agent_echo,
    is_semantically_complete,
    required_empty_chunks,
)


def _float32_to_pcm16(wav: np.ndarray) -> bytes:
    clipped = np.clip(wav, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _synth_16k(engine: Lfm3OmniEngine, text: str, gender: str = "male") -> np.ndarray:
    tok = engine.tokenizer
    model = engine.model
    codec = engine.codec
    cfg = model.config
    prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": f"Perform TTS in British English using a {gender} voice."},
            {"role": "user", "content": text},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = mx.array([tok.encode(prompt, add_special_tokens=False)], dtype=mx.int32)
    frames = []
    for _, c in model.stream_slots(
        ids,
        max_tokens=220,
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
def s2s_env():
    cfg = build_config()
    engine = Lfm3OmniEngine(cfg.engine)
    engine.load()
    yield cfg, engine
    engine.shutdown()


def test_semantic_helpers_and_echo_gate() -> None:
    assert is_semantically_complete("What is the monthly fee?")
    assert is_semantically_complete("Yes, go ahead.")
    assert not is_semantically_complete("Hello, I wanted to...")
    assert not is_semantically_complete("I earn twenty thousand and")
    assert required_empty_chunks("Yes, go ahead.", "balanced") == 1
    assert required_empty_chunks("I wanted to...", "balanced") == 2
    assert required_empty_chunks("I wanted to...", "patient") == 3

    agent_line = "You are eligible for the Ubuntu Gold card at 95 Rands per month."
    assert is_likely_agent_echo("Ubuntu Gold card", agent_line)
    assert not is_likely_agent_echo("Wait, hold on a second", agent_line)
    assert not is_likely_agent_echo("What about Platinum?", agent_line)


def test_pure_s2s_silence_and_noise_immunity(s2s_env) -> None:
    cfg, engine = s2s_env
    persona = cfg.get_persona("naledi")
    session = StreamingConversationSession(
        engine.model, engine.tokenizer, engine.codec, cfg.engine
    )
    session.initialize_conversation(persona.system_prompt, cfg.tools)

    rng = np.random.default_rng(9)
    silence_4s = rng.normal(0.0, 0.0005, 4 * 16000).astype(np.float32)
    noise_4s = rng.normal(0.0, 0.02, 4 * 16000).astype(np.float32)

    all_events = []
    for wav in (silence_4s, noise_4s):
        for off in range(0, len(wav), 5120):
            evs = session.ingest_pcm_and_listen(
                _float32_to_pcm16(wav[off : off + 5120]),
                agent_speaking=False,
                patience="balanced",
            )
            all_events.extend(evs)

    assert not all_events, f"Unexpected speech events on silence/noise: {all_events}"
    assert not session.user_speaking


def test_live_stream_speech_tool_call_and_barge_in_rollback(s2s_env) -> None:
    cfg, engine = s2s_env
    persona = cfg.get_persona("naledi")
    session = StreamingConversationSession(
        engine.model, engine.tokenizer, engine.codec, cfg.engine
    )
    session.initialize_conversation(persona.system_prompt, cfg.tools)

    # 1. Opening greeting
    cancel = threading.Event()
    session.append_text_user_turn(persona.greeting_trigger)
    g_out = session.generate_assistant_turn(
        persona=persona,
        tools=None,
        force_interleaved=True,
        cancel_event=cancel,
        on_event=lambda _: None,
    )
    assert g_out.kind == "response" and g_out.audio_frames > 10

    # 2. Stream spoken question + 2.5s trailing silence in 320 ms chunks
    rng = np.random.default_rng(11)
    q_wav = _synth_16k(
        engine,
        "Can I pre-qualify if I earn twenty eight thousand Rand a month and have a South African ID?",
    )
    sil_trail = rng.normal(0.0, 0.0004, int(3.2 * 16000)).astype(np.float32)
    stream_wav = np.concatenate([q_wav, sil_trail])

    events = []
    for off in range(0, len(stream_wav), 5120):
        events.extend(
            session.ingest_pcm_and_listen(
                _float32_to_pcm16(stream_wav[off : off + 5120]),
                agent_speaking=False,
                patience="balanced",
            )
        )

    event_types = [e["type"] for e in events]
    print(f"\n[Live Stream Turn 2 Events] {event_types}")
    assert "user_speech_start" in event_types
    assert "user_turn_committed" in event_types
    committed = [e for e in events if e["type"] == "user_turn_committed"][0]
    print(f"[Live Stream Caption] {committed['text']!r}")
    assert len(committed["text"]) > 10

    # Generate response (should invoke check_pre_qualification via incremental conv_cache!)
    out2 = session.generate_assistant_turn(
        persona=persona,
        tools=cfg.tools,
        force_interleaved=False,
        cancel_event=cancel,
        on_event=lambda _: None,
    )
    print(f"[Incremental Prefill Turn 2] kind={out2.kind} prefill_ms={out2.prefill_ms:.1f} tools={out2.tool_calls}")
    assert out2.prefill_ms < 250.0
    if out2.kind == "tool_calls":
        tool_map = {t.name: t for t in cfg.tools}
        results = [
            {"tool": c["function"]["name"], "result": tool_map[c["function"]["name"]].handler(**c["function"]["arguments"])}
            for c in out2.tool_calls
        ]
        session.append_tool_exchange(out2.tool_calls, json.dumps(results))
        out2 = session.generate_assistant_turn(
            persona=persona,
            tools=None,
            force_interleaved=True,
            cancel_event=cancel,
            on_event=lambda _: None,
        )
    assert out2.kind == "response" and out2.audio_frames > 10

    # 3. Simulate barge-in over Turn 2's completed reply and verify coherent Turn 3!
    barge_wav = _synth_16k(engine, "Hold on, what is the Classic card fee?")
    barge_stream = np.concatenate([barge_wav, sil_trail])
    barge_events = []
    rolled_back = False
    for off in range(0, len(barge_stream), 5120):
        evs = session.ingest_pcm_and_listen(
            _float32_to_pcm16(barge_stream[off : off + 5120]),
            agent_speaking=not rolled_back,
            recent_agent_text=out2.text,
            patience="balanced",
        )
        for ev in evs:
            if ev["type"] == "barge_in_detected":
                rolled_back = True
                trunc = session.rollback_interrupted_assistant_turn(
                    played_frames=12,
                    interrupted_turn=out2.turn_record,
                    replace_last_history=True,
                )
                assert trunc is not None and trunc.text.endswith("...")
        barge_events.extend(evs)

    assert rolled_back
    assert any(e["type"] == "user_turn_committed" for e in barge_events)

    out3 = session.generate_assistant_turn(
        persona=persona,
        tools=cfg.tools,
        force_interleaved=False,
        cancel_event=cancel,
        on_event=lambda _: None,
    )
    print(f"[Post-Barge-In Turn 3] kind={out3.kind} text={out3.text!r} frames={out3.audio_frames}")
    assert out3.kind == "response" and out3.audio_frames > 5


def test_five_minute_rollover_and_compaction_recall(s2s_env) -> None:
    cfg, engine = s2s_env
    persona = cfg.get_persona("naledi")
    # Use a low compaction threshold (650 tokens) to exercise multiple compactions
    # across a rich multi-turn call, AND stream 312 seconds (> 5 minutes) through
    # StreamingAudioEncoder + LiveS2SListener to prove O(1) rollover stability!
    settings = EngineSettings(
        compaction_trigger_tokens=650,
        keep_recent_user_audio_turns=2,
        keep_recent_assistant_audio_turns=2,
        max_slots=350,
    )
    session = StreamingConversationSession(
        engine.model, engine.tokenizer, engine.codec, settings
    )
    session.initialize_conversation(persona.system_prompt, None)

    # Part A: 312 seconds (> 5 minutes) of continuous streaming through causal encoder + LiveS2SListener
    rng = np.random.default_rng(42)
    sil_chunk = rng.normal(0.0, 0.0003, int(3.0 * 16000)).astype(np.float32)
    pcm_3s = _float32_to_pcm16(sil_chunk)
    t0 = time.perf_counter()
    for _ in range(104):  # 104 * 3s = 312 seconds (5m 12s)
        evs = session.ingest_pcm_and_listen(pcm_3s, agent_speaking=False)
        assert not evs
    elapsed_312s = time.perf_counter() - t0
    print(
        f"\n[312s Continuous Stream Benchmark] processed 5m12s in {elapsed_312s:.2f}s "
        f"| listener_resets={session.listener.reset_count} | window_toks={session.listener.audio_tokens_in_window}"
    )
    assert session.listener.reset_count >= 10
    assert session.listener.audio_tokens_in_window <= session.listener.hard_reset_tokens

    # Part B: Multi-turn conversation triggering multiple compactions + fact recall
    cancel = threading.Event()
    turns_input = [
        "My name is Sipho Ndlovu and my monthly salary is twenty eight thousand Rand.",
        "I mostly buy groceries at Checkers and Woolworths.",
        "Does the card include fifty five days interest free on purchases?",
        "Great, can I also get free virtual cards on the app?",
        "And what is my name and monthly salary that I told you at the start of this call?",
    ]
    last_out = None
    for idx, msg in enumerate(turns_input):
        if idx < 4:
            wav = _synth_16k(engine, msg)
            sil = rng.normal(0.0, 0.0003, int(2.0 * 16000)).astype(np.float32)
            full_wav = np.concatenate([wav, sil])
            for off in range(0, len(full_wav), 5120):
                session.ingest_pcm_and_listen(
                    _float32_to_pcm16(full_wav[off : off + 5120]),
                    agent_speaking=False,
                    patience="snappy",
                )
            if session.user_speaking:
                session._commit_current_user_turn([])
        else:
            session.append_text_user_turn(msg)

        last_out = session.generate_assistant_turn(
            persona=persona,
            tools=None,
            force_interleaved=True,
            cancel_event=cancel,
            on_event=lambda _: None,
        )
        compacted = session.maybe_compact_context()
        print(
            f"[Turn {idx+1}] tokens={session.conv_tokens} compacted={compacted} total_compactions={session.compaction_count}"
        )

    assert session.compaction_count >= 1
    assert last_out is not None and last_out.kind == "response"
    print(f"[Post-Compaction Recall Reply] {last_out.text!r}")
    lower_reply = last_out.text.lower()
    assert "sipho" in lower_reply or "28" in lower_reply or "twenty" in lower_reply


def test_conversational_progression_no_greeting_loop_and_late_barge_in(s2s_env) -> None:
    """Verify finalize_greeting_turn, safe barge-in after rebuild, and forward progression."""
    cfg, engine = s2s_env
    persona = cfg.get_persona("naledi")
    session = StreamingConversationSession(
        engine.model, engine.tokenizer, engine.codec, cfg.engine
    )
    sys_prompt = build_localized_system_prompt(
        persona.system_prompt,
        "en",
        persona.name,
        persona.voice_gender,
        "selective_tool",
    )
    session.initialize_conversation(sys_prompt, cfg.tools, language="en")

    # 1. Greeting + cleanup via finalize_greeting_turn
    cancel = threading.Event()
    session.append_text_user_turn(build_localized_greeting_trigger("en", persona.name))
    g_out = session.generate_assistant_turn(
        persona=persona,
        tools=None,
        force_interleaved=True,
        cancel_event=cancel,
        on_event=lambda _: None,
    )
    assert g_out.kind == "response"
    session.finalize_greeting_turn()
    assert session.history[0].text == "Hello?"
    assert session.pre_turn_snapshot is None

    # 2. Spoken "Yeah, please continue." while agent_speaking=True triggers barge-in
    rng = np.random.default_rng(17)
    sil_trail = rng.normal(0.0, 0.0003, int(3.2 * 16000)).astype(np.float32)
    u1_wav = _synth_16k(engine, "Yeah, please continue.")
    u1_stream = np.concatenate([u1_wav, sil_trail])
    rolled = False
    evs1 = []
    for off in range(0, len(u1_stream), 5120):
        evs = session.ingest_pcm_and_listen(
            _float32_to_pcm16(u1_stream[off : off + 5120]),
            agent_speaking=not rolled,
            recent_agent_text=g_out.text,
            patience="balanced",
        )
        for ev in evs:
            if ev["type"] == "barge_in_detected":
                rolled = True
                session.rollback_interrupted_assistant_turn(
                    played_frames=0,
                    interrupted_turn=g_out.turn_record,
                    replace_last_history=True,
                )
        evs1.extend(evs)

    assert rolled
    assert session.pending_barge_in_blocks is None
    assert session.history[1].role == "assistant"
    assert any(e.get("type") == "user_turn_committed" for e in evs1)

    plan1 = session.run_speculative_selective_batch(
        persona=persona,
        tools=cfg.tools,
        dynamic_filler_enabled=True,
        cancel_event=cancel,
        on_event=lambda _: None,
    )
    assert plan1.direct_outcome is not None
    t1_text = plan1.direct_outcome.text
    print(f"\n[Progression Turn 1] dec={plan1.decision} text={t1_text!r}")
    assert "do you have a quick minute" not in t1_text.lower()
    assert "my gross monthly income is" not in t1_text.lower()

    # 3. Second affirmative turn must also keep advancing, not repeat introduction
    session.append_text_user_turn("Yes, yes, please continue.")
    plan2 = session.run_speculative_selective_batch(
        persona=persona,
        tools=cfg.tools,
        dynamic_filler_enabled=True,
        cancel_event=cancel,
        on_event=lambda _: None,
    )
    assert plan2.direct_outcome is not None
    t2_text = plan2.direct_outcome.text
    print(f"[Progression Turn 2] dec={plan2.decision} text={t2_text!r}")
    assert "do you have a quick minute" not in t2_text.lower()
    assert "my gross monthly income is" not in t2_text.lower()
