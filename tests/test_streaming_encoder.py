"""Tests verifying causal StreamingLogMel and StreamingAudioEncoder parity and 35s streaming."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx_vlm.models.lfm3_audio.audio import (
    extract_features,
    num_encoder_tokens,
    num_frames,
)

from voice_agent.audio_stream import StreamingAudioEncoder, StreamingLogMel
from voice_agent.config import EngineSettings
from voice_agent.engine import Lfm3OmniEngine


@pytest.fixture(scope="module")
def loaded_engine() -> Lfm3OmniEngine:
    engine = Lfm3OmniEngine(EngineSettings())
    engine.load()
    yield engine
    engine.shutdown()


def test_streaming_log_mel_and_audio_encoder_parity(loaded_engine: Lfm3OmniEngine) -> None:
    model = loaded_engine.model
    tok = loaded_engine.tokenizer
    codec = loaded_engine.codec
    cfg = model.config

    # Synthesize real 16 kHz speech via LFM3-Omni TTS
    prompt = tok.apply_chat_template(
        [
            {
                "role": "system",
                "content": "Perform TTS in British English using a female voice.",
            },
            {
                "role": "user",
                "content": "Welcome to Karoo Crest Bank in South Africa. How can I help today?",
            },
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
    wav16 = np.interp(
        np.linspace(0.0, 1.0, n16, endpoint=False),
        np.linspace(0.0, 1.0, wav24.shape[0], endpoint=False),
        wav24,
    ).astype(np.float32)

    feats, mask = extract_features([wav16])
    batch_emb = model.audio_projector(
        model.audio_tower(mx.array(feats), mx.array(mask).sum(-1))
    )
    mx.eval(batch_emb)

    mel_stream = StreamingLogMel()
    enc_stream = StreamingAudioEncoder(model)
    mel_parts: list[np.ndarray] = []
    emb_parts: list[mx.array] = []
    for i in range(0, len(wav16), 512):
        m = mel_stream.push(wav16[i : i + 512], is_final=False)
        if m.shape[1]:
            mel_parts.append(m)
            e = enc_stream.push(m, is_final=False)
            if e.shape[0]:
                emb_parts.append(e)
    m_fin = mel_stream.push(np.zeros((0,), dtype=np.float32), is_final=True)
    if m_fin.shape[1]:
        mel_parts.append(m_fin)
    e_fin = enc_stream.push(m_fin, is_final=True)
    if e_fin.shape[0]:
        emb_parts.append(e_fin)

    stream_mel = np.concatenate(mel_parts, axis=1)
    stream_emb = mx.concatenate(emb_parts, axis=0)
    mx.eval(stream_emb)

    assert stream_mel.shape == feats[0].shape
    mel_diff = float(np.max(np.abs(stream_mel - feats[0])))
    assert mel_diff < 1e-6

    assert stream_emb.shape == batch_emb.shape
    diff = np.abs(
        np.asarray(stream_emb.astype(mx.float32))
        - np.asarray(batch_emb.astype(mx.float32))
    )
    rmse = float(np.sqrt(np.mean(diff * diff)))
    print(f"\n[StreamingAudioEncoder] tokens={stream_emb.shape[0]} mel_diff={mel_diff:.2e} emb_rmse={rmse:.2e}")
    assert rmse < 5e-3


def test_streaming_audio_encoder_35_seconds_continuous(loaded_engine: Lfm3OmniEngine) -> None:
    model = loaded_engine.model
    mel_stream = StreamingLogMel()
    enc_stream = StreamingAudioEncoder(model)

    # 35 seconds (> 13x the 2.56s sliding attention window) in 512-sample chunks
    rng = np.random.default_rng(1)
    total_samples = 35 * 16000
    chunk_1s = (0.05 * np.sin(2 * np.pi * 500 * np.arange(16000) / 16000) + rng.normal(0, 0.005, 16000)).astype(np.float32)

    total_tokens = 0
    for sec in range(35):
        for off in range(0, 16000, 512):
            m = mel_stream.push(chunk_1s[off : off + 512], is_final=False)
            if m.shape[1]:
                e = enc_stream.push(m, is_final=False)
                total_tokens += int(e.shape[0])
    m_fin = mel_stream.push(np.zeros((0,), dtype=np.float32), is_final=True)
    e_fin = enc_stream.push(m_fin, is_final=True)
    total_tokens += int(e_fin.shape[0])

    expected = int(num_encoder_tokens(num_frames(total_samples)))
    assert total_tokens == expected
    # Encoder KVCache remains bounded at sliding_window=32
    for c in enc_stream.cache:
        if hasattr(c, "max_size"):
            assert c.size() <= 32
