"""Integration test verifying StreamingLiquidCodecDecoder matches batch LiquidCodec.decode."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx_vlm.models.lfm3_audio.codec import load_codec

from voice_agent.codec_stream import (
    StreamingAudioFlusher,
    StreamingLiquidCodecDecoder,
)
from voice_agent.config import EngineSettings
from voice_agent.engine import Lfm3OmniEngine


@pytest.fixture(scope="module")
def loaded_engine() -> Lfm3OmniEngine:
    engine = Lfm3OmniEngine(EngineSettings())
    engine.load()
    yield engine
    engine.shutdown()


def _stream_decode_all(codec, codes_all: np.ndarray) -> np.ndarray:
    decoder = StreamingLiquidCodecDecoder(codec)
    stream_parts: list[np.ndarray] = []
    idx = 0
    chunk_size = 2
    while idx < codes_all.shape[1]:
        nxt = min(idx + chunk_size, codes_all.shape[1])
        part = decoder.decode_chunk(codes_all[:, idx:nxt], finalize=False)
        if part.size > 0:
            stream_parts.append(part)
        idx = nxt
        chunk_size = 4
    tail = decoder.decode_chunk(np.zeros((12, 0), dtype=np.int32), finalize=True)
    if tail.size > 0:
        stream_parts.append(tail)
    return np.concatenate(stream_parts) if stream_parts else np.zeros((0,), dtype=np.float32)


def test_streaming_codec_matches_batch_decode_on_real_speech_codes(
    loaded_engine: Lfm3OmniEngine,
) -> None:
    model = loaded_engine.model
    codec_bf16 = loaded_engine.codec
    tokenizer = loaded_engine.tokenizer
    cfg = model.config

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "Perform TTS in British English using a female voice.",
            },
            {
                "role": "user",
                "content": "Good afternoon from Karoo Crest Bank in South Africa.",
            },
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = mx.array([tokenizer.encode(prompt, add_special_tokens=False)], dtype=mx.int32)

    frames = []
    for _token, codes in model.stream_slots(
        input_ids,
        max_tokens=200,
        generation_mode="audio",
        audio_temperature=0.4,
        audio_top_k=128,
        audio_repetition_penalty=1.2,
    ):
        if codes is not None:
            c_np = np.asarray(codes, dtype=np.int32)
            if int(c_np[0]) == cfg.audio_eos_code:
                break
            frames.append(c_np)

    assert len(frames) >= 12, f"Expected >= 12 speech frames, got {len(frames)}"
    codes_all = np.stack(frames, axis=-1)  # [12, T]

    # 1. Exact algorithmic equivalence in float32 (rules out cache/mask/ISTFT bugs).
    codec_fp32 = load_codec(model.model_path, dtype=mx.float32)
    batch_fp32 = np.asarray(codec_fp32.decode(mx.array(codes_all)), dtype=np.float32)
    stream_fp32 = _stream_decode_all(codec_fp32, codes_all)
    assert stream_fp32.shape == batch_fp32.shape == (codes_all.shape[1] * 1920,)
    fp32_max = float(np.max(np.abs(stream_fp32 - batch_fp32)))
    fp32_rmse = float(np.sqrt(np.mean((stream_fp32 - batch_fp32) ** 2)))
    print(f"\n[Codec Stream fp32] frames={codes_all.shape[1]}, max_abs_diff={fp32_max:.2e}, rmse={fp32_rmse:.2e}")
    assert fp32_max < 1e-4, f"float32 streaming diverged: {fp32_max}"

    # 2. Runtime bfloat16 equivalence within BF16 ULP tolerance.
    batch_bf16 = np.asarray(codec_bf16.decode(mx.array(codes_all)), dtype=np.float32)
    stream_bf16 = _stream_decode_all(codec_bf16, codes_all)
    assert stream_bf16.shape == batch_bf16.shape == (codes_all.shape[1] * 1920,)
    bf16_max = float(np.max(np.abs(stream_bf16 - batch_bf16)))
    bf16_rmse = float(np.sqrt(np.mean((stream_bf16 - batch_bf16) ** 2)))
    print(f"[Codec Stream bf16] frames={codes_all.shape[1]}, max_abs_diff={bf16_max:.2e}, rmse={bf16_rmse:.2e}")
    assert bf16_rmse < 5e-3, f"bfloat16 streaming RMSE too high: {bf16_rmse}"
    assert bf16_max < 1e-1, f"bfloat16 streaming peak diff too high: {bf16_max}"


def test_frame1_streaming_flusher_emits_1440_samples_immediately(
    loaded_engine: Lfm3OmniEngine,
) -> None:
    codec = loaded_engine.codec
    rng = np.random.default_rng(7)
    codes = rng.integers(0, codec.config.codebook_size, size=(12, 9), dtype=np.int32)
    emitted: list[dict] = []

    flusher = StreamingAudioFlusher(
        codec=codec,
        first_chunk_frames=1,
        stream_chunk_frames=2,
        on_event=emitted.append,
        segment_name="answer",
    )
    for idx in range(codes.shape[1]):
        flusher.push_frame(codes[:, idx])
    flusher.flush(finalize=True)

    # 9 frames = 1 (1440) + 2 (3840) + 2 (3840) + 2 (3840) + 2 (3840) + final tail (480) = 6 chunks
    assert len(emitted) == 6, f"Expected 6 streamed chunks, got {len(emitted)}"
    assert emitted[0]["chunk_index"] == 1
    assert emitted[0]["frames"] == 1
    assert emitted[0]["samples"] == 1440  # 60 ms @ 24 kHz (> 960-sample prebuffer)
    assert flusher.first_chunk_samples == 1440
    total_pcm_samples = sum(len(ev["pcm16"]) // 2 for ev in emitted)
    assert total_pcm_samples == codes.shape[1] * 1920
