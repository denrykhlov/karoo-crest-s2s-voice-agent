"""Tests for the direct PyTorch/Transformers backend (`voice_agent/transformers_backend.py`)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from telesales_agent import build_config, build_personas, build_tools
from voice_agent.backend_factory import build_engine
from voice_agent.config import EngineSettings
from voice_agent.server import create_app
from voice_agent.transformers_backend import (
    TransformersLiveS2SListener,
    TransformersOmniEngine,
)

AUDIO_OUT_ID = 125032
AUDIO_EOS_CODE = 2016
MIXED_START_ID = 125015
TEXT_END_ID = 125014
IM_END_ID = 124900
PAD_ID = 124893


def _pcm16_16k_block(blocks: int = 1, freq: float = 300.0) -> bytes:
    """Create non-silent 16 kHz mono Int16 PCM bytes (30,720 bytes = 960 ms per block)."""
    n_samples = blocks * 15360
    t = np.arange(n_samples, dtype=np.float32)
    return (np.sin(2.0 * np.pi * freq * t / 16000.0) * 1200).astype("<i2").tobytes()


class FakeDynamicCache:
    """Mutable tensor-backed cache tracking exact seq_len and conv state for rollback tests."""

    def __init__(self) -> None:
        self.seq_len: int = 0
        self.events: list[tuple[str, int]] = []
        self.conv_state: np.ndarray = np.zeros((1, 4), dtype=np.float32)

    def get_seq_length(self) -> int:
        return self.seq_len

    def advance(self, kind: str, n: int) -> None:
        if n > 0:
            self.seq_len += n
            self.events.append((kind, n))
            self.conv_state += float(n)


@dataclass
class FakeFeatureCache:
    num_samples: int = 0
    num_frames: int = 0
    finished: bool = False


class FakeFeatureExtractor:
    """Causal sample->mel streamer matching `Lfm3OmniFeatureExtractor.stream` semantics."""

    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    def stream(
        self,
        raw_speech: np.ndarray,
        cache: FakeFeatureCache | None = None,
        is_final: bool = False,
        sampling_rate: int = 16000,
    ) -> tuple[dict[str, Any], FakeFeatureCache]:
        assert sampling_rate == 16000
        if cache is None:
            cache = FakeFeatureCache()
        if cache.finished:
            raise ValueError("Feature stream already finalized")
        arr = np.asarray(raw_speech, dtype=np.float32).reshape(-1)
        new_cache = copy.deepcopy(cache)
        new_cache.num_samples += int(arr.size)
        # 160 samples per mel frame (10 ms); flush >=1 frame when final
        target_frames = new_cache.num_samples // 160
        if is_final:
            target_frames = max(target_frames, new_cache.num_frames + 1)
        count = max(0, target_frames - new_cache.num_frames)
        new_cache.num_frames += count
        new_cache.finished = bool(is_final)
        self.stream_calls.append(
            {"samples": int(arr.size), "mel_frames": count, "is_final": is_final}
        )
        return {"input_features": np.zeros((1, 128, count), dtype=np.float32)}, new_cache


@dataclass
class FakeCodecCache:
    started: bool = False
    finished: bool = False


class FakeLiquidCodec:
    """Streaming 24 kHz LiquidCodec emitting 1440 samples on frame 1 and 480 tail on finalize."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(n_q=12, codebook_size=2017)
        self.decode_calls: list[dict[str, Any]] = []

    def make_streaming_cache(self) -> FakeCodecCache:
        return FakeCodecCache()

    def decode(
        self,
        audio_codes: np.ndarray,
        streaming_cache: FakeCodecCache | None = None,
        finalize: bool = True,
        return_dict: bool = True,
    ) -> Any:
        codes = np.asarray(audio_codes)
        assert codes.ndim == 3 and codes.shape[1] == 12
        n_frames = int(codes.shape[-1])
        self.decode_calls.append({"frames": n_frames, "finalize": finalize})

        samples = 0
        if streaming_cache is not None:
            if n_frames > 0:
                if not streaming_cache.started:
                    streaming_cache.started = True
                    samples = 1440 + max(0, n_frames - 1) * 1920
                else:
                    samples = n_frames * 1920
            if finalize:
                if streaming_cache.started and not streaming_cache.finished:
                    samples += 480
                streaming_cache.finished = True
        else:
            samples = n_frames * 1920

        t = np.arange(samples, dtype=np.float32)
        wav = (np.sin(2.0 * np.pi * 440.0 * t / 24000.0) * 0.08).astype(np.float32)
        return SimpleNamespace(audio_values=wav[None, :])


class FakeTokenizer:
    """Deterministic sub-vocab + UTF-8 fallback tokenizer compatible with LFM3 tags."""

    SPECIAL_TO_ID = {
        "<|startoftext|>": 124894,
        "<|endoftext|>": 124895,
        "<|im_start|>": 124899,
        "<|im_end|>": IM_END_ID,
        "<|vision_start|>": 125009,
        "<|vision_end|>": 125010,
        "<|audio|>": 125031,
        "<|audio_start|>": 125012,
        "<|audio_out|>": AUDIO_OUT_ID,
        "<|mixed_start|>": MIXED_START_ID,
        "<|mixed_end|>": 125016,
        "<|text_end|>": TEXT_END_ID,
        "<think>": 124901,
        "</think>": 124902,
        "<|pad|>": PAD_ID,
    }
    ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TO_ID.items()}

    def convert_tokens_to_ids(self, tok: str) -> int | None:
        return self.SPECIAL_TO_ID.get(tok)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        if add_special_tokens:
            raise AssertionError("Pass add_special_tokens=False to avoid double BOS")
        ids: list[int] = []
        i = 0
        tags = sorted(self.SPECIAL_TO_ID.keys(), key=len, reverse=True)
        while i < len(text):
            matched = False
            for tag in tags:
                if text.startswith(tag, i):
                    ids.append(self.SPECIAL_TO_ID[tag])
                    i += len(tag)
                    matched = True
                    break
            if not matched:
                for b in text[i].encode("utf-8"):
                    ids.append(b)
                i += 1
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        out = bytearray()
        parts: list[str] = []
        for tid in token_ids:
            tid = int(tid)
            if tid in self.ID_TO_SPECIAL:
                if out:
                    parts.append(out.decode("utf-8", errors="replace"))
                    out.clear()
                if not skip_special_tokens:
                    parts.append(self.ID_TO_SPECIAL[tid])
            elif 0 <= tid < 256:
                out.append(tid)
        if out:
            parts.append(out.decode("utf-8", errors="replace"))
        return "".join(parts)


@dataclass
class FakeEncoderState:
    num_tokens: int = 0
    finished: bool = False


@dataclass
class FakeLiveState:
    encoder_cache: FakeEncoderState = field(default_factory=FakeEncoderState)
    context_restarts: int = 0
    finished: bool = False


class FakeTransformersLfm3Model:
    """In-process fake implementing `model`, `prefill_audio`, `transcribe_live`, `generate`."""

    def __init__(self, tokenizer: FakeTokenizer) -> None:
        self.tok = tokenizer
        self.config = SimpleNamespace(
            audio_out_token_id=AUDIO_OUT_ID,
            audio_eos_code=AUDIO_EOS_CODE,
            num_codebooks=12,
            codebook_size=2017,
            mixed_start_token_id=MIXED_START_ID,
            text_end_token_id=TEXT_END_ID,
            interleaved_n_text=6,
            interleaved_n_audio=12,
        )
        self.prefill_audio_calls: list[dict[str, Any]] = []
        self.live_asr_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.asr_sequence: list[str] = []
        self._asr_idx: int = 0
        self.router_decision: str = "DIRECT"
        self.answer_text: str = "Hello! Naledi from Karoo Crest Bank here."
        self.filler_text: str = "One moment while I check."
        self.summary_text: str = "Customer inquired about Ubuntu fees."
        self.tool_call_on_next_answer: tuple[str, dict[str, Any]] | None = None
        self.interleaved_frames_per_turn: int = 3
        self.slot_delay_s: float = 0.0

    def make_conv_cache(self) -> FakeDynamicCache:
        return FakeDynamicCache()

    def model(
        self,
        input_ids: Any = None,
        inputs_embeds: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: FakeDynamicCache | None = None,
        use_cache: bool = True,
    ) -> Any:
        if past_key_values is None:
            past_key_values = FakeDynamicCache()
        inp = input_ids if input_ids is not None else inputs_embeds
        n_tok = int(inp.shape[1])
        if attention_mask is not None:
            assert attention_mask.shape == (1, past_key_values.get_seq_length() + n_tok)
        past_key_values.advance("text", n_tok)
        return SimpleNamespace(past_key_values=past_key_values)

    def prefill_audio(
        self,
        input_features: np.ndarray,
        past_key_values: FakeDynamicCache | None = None,
        encoder_cache: FakeEncoderState | None = None,
        attention_mask: Any = None,
        is_final: bool = False,
    ) -> Any:
        if past_key_values is None:
            past_key_values = FakeDynamicCache()
        past_len = past_key_values.get_seq_length()
        if attention_mask is None:
            assert past_len == 0
            attention_mask = np.ones((1, 0), dtype=np.int64)
        assert attention_mask.shape == (1, past_len), (
            f"attention_mask {attention_mask.shape} != past_len {past_len}"
        )
        if encoder_cache is None:
            encoder_cache = FakeEncoderState()
        mel_frames = int(input_features.shape[-1])
        audio_tokens = max(0, mel_frames // 8)
        if is_final and mel_frames > 0 and audio_tokens == 0:
            audio_tokens = 1
        elif is_final and encoder_cache.num_tokens == 0:
            audio_tokens = max(audio_tokens, 1)
        encoder_cache = copy.deepcopy(encoder_cache)
        encoder_cache.num_tokens += audio_tokens
        encoder_cache.finished = bool(is_final)
        past_key_values.advance("audio", audio_tokens)
        if audio_tokens:
            attention_mask = np.concatenate(
                (attention_mask, np.ones((1, audio_tokens), dtype=np.int64)), axis=1
            )
        self.prefill_audio_calls.append(
            {
                "mel_frames": mel_frames,
                "audio_tokens": audio_tokens,
                "is_final": is_final,
                "seq_len_after": past_key_values.get_seq_length(),
            }
        )
        return SimpleNamespace(
            past_key_values=past_key_values,
            encoder_cache=encoder_cache,
            attention_mask=attention_mask,
        )

    def transcribe_live(
        self,
        input_features: np.ndarray,
        state: FakeLiveState | None = None,
        input_ids: Any = None,
        is_final: bool = False,
        max_new_tokens: int = 48,
        suppress_token_ids: list[int] | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> Any:
        if state is None:
            state = FakeLiveState()
        else:
            state = copy.deepcopy(state)
        if input_ids is not None:
            prompt_str = self.tok.decode(np.asarray(input_ids[0]).tolist(), skip_special_tokens=False)
            assert "Perform live ASR." in prompt_str
            state.context_restarts += 1
        mel_frames = int(input_features.shape[-1])
        num_audio_tokens = max(1, mel_frames // 8) if mel_frames else 0
        state.encoder_cache.num_tokens += num_audio_tokens
        if is_final:
            state.finished = True
        idx = self._asr_idx
        self._asr_idx += 1
        text = self.asr_sequence[idx] if idx < len(self.asr_sequence) else ""
        self.live_asr_calls.append(
            {
                "mel_frames": mel_frames,
                "num_audio_tokens": num_audio_tokens,
                "restarts": state.context_restarts,
                "text": text,
            }
        )
        seq = np.asarray([self.tok.encode(text, add_special_tokens=False)], dtype=np.int64)
        return SimpleNamespace(
            sequences=seq,
            num_audio_tokens=num_audio_tokens,
            closed=True,
            state=state,
        )

    def generate(
        self,
        input_ids: Any,
        attention_mask: Any = None,
        generation_config: Any = None,
        generation_mode: str | None = None,
        streamer: Any = None,
        audio_streamer: Any = None,
        past_key_values: FakeDynamicCache | None = None,
        return_dict_in_generate: bool = True,
        **_kwargs: Any,
    ) -> Any:
        prompt_ids = np.asarray(input_ids[0]).tolist()
        rendered = self.tok.decode(prompt_ids, skip_special_tokens=False)
        used_cache = past_key_values is not None
        if past_key_values is None:
            past_key_values = FakeDynamicCache()
        if attention_mask is not None:
            assert attention_mask.shape == (1, past_key_values.get_seq_length() + len(prompt_ids))

        self.generate_calls.append(
            {
                "rendered": rendered,
                "mode": generation_mode,
                "used_cache": used_cache,
                "cache_seq_before": past_key_values.get_seq_length(),
            }
        )
        past_key_values.advance("prompt", len(prompt_ids))

        if generation_mode == "text" or (
            "<|mixed_start|>" not in rendered and self.tool_call_on_next_answer is None and "Output strictly" in rendered
        ):
            if "Output strictly" in rendered:
                reply = self.router_decision
            elif "Produce a compact factual call memory summary" in rendered or "compact" in rendered:
                reply = self.summary_text
            elif "standalone banking knowledge question" in rendered:
                reply = "What are the Platinum fees?"
            else:
                reply = "Ok"
            gen_ids = self.tok.encode(reply, add_special_tokens=False) + [IM_END_ID]
            past_key_values.advance("gen", len(gen_ids))
            return SimpleNamespace(sequences=np.asarray([prompt_ids + gen_ids], dtype=np.int64))

        # Streamed generation (with or without tools)
        streams = [s for s in (streamer, audio_streamer) if s is not None]
        gen_out_ids: list[int] = []
        try:
            if streamer is not None:
                streamer.put(np.asarray([prompt_ids], dtype=np.int64))

            if self.tool_call_on_next_answer is not None and generation_mode is None:
                name, args = self.tool_call_on_next_answer
                self.tool_call_on_next_answer = None
                tc_str = (
                    f"<tool_call>\n<function={name}>\n"
                    + "\n".join(
                        f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>"
                        for k, v in args.items()
                    )
                    + "\n</function>\n</tool_call>"
                )
                toks = self.tok.encode(tc_str, add_special_tokens=False) + [IM_END_ID]
                for tid in toks:
                    gen_out_ids.append(tid)
                    past_key_values.advance("gen", 1)
                    if streamer is not None:
                        streamer.put(np.asarray([[tid]], dtype=np.int64))
                return SimpleNamespace(
                    sequences=np.asarray([prompt_ids + gen_out_ids], dtype=np.int64)
                )

            if "holding phrase" in rendered.lower():
                text_str = self.filler_text
            else:
                text_str = self.answer_text

            text_toks = self.tok.encode(text_str, add_special_tokens=False)
            # Emit interleaved: chunk text tokens + audio frames, ending with text_end, audio, im_end
            split = max(1, len(text_toks) // 2)
            for part_idx, chunk in enumerate((text_toks[:split], text_toks[split:])):
                for tid in chunk:
                    if self.slot_delay_s > 0:
                        time.sleep(self.slot_delay_s)
                    gen_out_ids.append(tid)
                    past_key_values.advance("gen", 1)
                    if streamer is not None:
                        streamer.put(np.asarray([[tid]], dtype=np.int64))
                if part_idx == 1:
                    gen_out_ids.append(TEXT_END_ID)
                    past_key_values.advance("gen", 1)
                    if streamer is not None:
                        streamer.put(np.asarray([[TEXT_END_ID]], dtype=np.int64))
                for f_idx in range(self.interleaved_frames_per_turn):
                    if self.slot_delay_s > 0:
                        time.sleep(self.slot_delay_s)
                    frame = np.full((12,), 10 + part_idx * 10 + f_idx, dtype=np.int64)
                    gen_out_ids.append(AUDIO_OUT_ID)
                    past_key_values.advance("gen", 1)
                    if audio_streamer is not None:
                        audio_streamer.put(frame)

            # Audio EOS slot (advances cache, not sent to audio_streamer)
            past_key_values.advance("gen", 1)
            gen_out_ids.append(IM_END_ID)
            past_key_values.advance("gen", 1)
            if streamer is not None:
                streamer.put(np.asarray([[IM_END_ID]], dtype=np.int64))
        except BaseException:
            for s in streams:
                try:
                    s.end()
                except Exception:
                    pass
            raise

        for s in streams:
            s.end()
        return SimpleNamespace(sequences=np.asarray([prompt_ids + gen_out_ids], dtype=np.int64))


def _make_fake_processor(
    tok: FakeTokenizer,
    feat: FakeFeatureExtractor,
    codec: FakeLiquidCodec,
) -> Any:
    return SimpleNamespace(
        tokenizer=tok,
        feature_extractor=feat,
        audio_tokenizer=codec,
        vision_start_token="<|vision_start|>",
        vision_end_token="<|vision_end|>",
    )


def test_zero_mlx_imports_and_config_wiring() -> None:
    """`transformers` backend must import cleanly with zero `mlx` imports and proper defaults."""
    code = (
        "import sys; "
        "from voice_agent.config import EngineSettings; "
        "from voice_agent.backend_factory import build_engine; "
        "eng = build_engine(EngineSettings(backend='transformers', model_id='LiquidAI/LFM3-Omni-2B')); "
        "assert 'mlx' not in sys.modules and 'mlx_vlm' not in sys.modules; "
        "assert eng.__class__.__name__ == 'TransformersOmniEngine'"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0

    cfg = build_config(backend="transformers", calls_dir="")
    assert cfg.engine.backend == "transformers"
    assert cfg.engine.model_id == "LiquidAI/LFM3-Omni-2B"
    pub = cfg.to_public_config()
    assert pub["backend"] == "transformers"
    assert pub["batch_parallel_mode"] == "transformers_direct_cuda"


def test_transformers_incremental_prefill_audio_and_live_asr_stateful() -> None:
    """Audio chunks stream into `prefill_audio` during speech and commit with `is_final=True`."""
    tok = FakeTokenizer()
    feat = FakeFeatureExtractor()
    codec = FakeLiquidCodec()
    model = FakeTransformersLfm3Model(tok)
    model.asr_sequence = [
        "Hello,",  # block 1 (speech)
        "how are you.",  # block 2 (speech, semantically complete)
        "",  # block 3 (silence -> commit)
    ]
    proc = _make_fake_processor(tok, feat, codec)
    settings = EngineSettings(backend="transformers", model_id="LiquidAI/LFM3-Omni-2B")
    engine = TransformersOmniEngine(settings, model=model, processor=proc, codec=codec)
    engine.load()
    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation("You are Naledi.", tools=None, language="en")
        seq_after_init = sess.conv_tokens
        assert seq_after_init > 0

        block = _pcm16_16k_block(blocks=1)
        evs1 = sess.ingest_pcm_and_listen(block, patience="balanced")
        assert evs1[0]["type"] == "user_speech_start"
        assert evs1[0]["text"] == "Hello,"
        # Incremental `prefill_audio(..., is_final=False)` already ran on block 1!
        assert any(not c["is_final"] for c in model.prefill_audio_calls)
        seq_after_b1 = sess.conv_tokens
        assert seq_after_b1 > seq_after_init
        prefill_count_b1 = len(model.prefill_audio_calls)

        evs2 = sess.ingest_pcm_and_listen(block, patience="balanced")
        assert evs2[0]["type"] == "user_speech_delta"
        assert sess.conv_tokens > seq_after_b1
        assert len(model.prefill_audio_calls) > prefill_count_b1

        # Silence block -> commits turn using `is_final=True` flush (zero full-audio recompute)
        resets_before = sess.listener.reset_count
        evs3 = sess.ingest_pcm_and_listen(block, patience="balanced")
        committed = [e for e in evs3 if e["type"] == "user_turn_committed"]
        assert len(committed) == 1
        assert committed[0]["text"] == "Hello, how are you."
        assert isinstance(committed[0]["last_speech_ts"], float)
        assert model.prefill_audio_calls[-1]["is_final"] is True
        assert model.prefill_audio_calls[-1]["mel_frames"] <= 10
        assert sess.listener.reset_count == resets_before + 1

        # Half-stride 480ms probe triggers snappy early commit after complete sentence
        model._asr_idx = 0
        model.asr_sequence = ["Yes, please.", ""]  # 960ms speech + 480ms silent probe
        evs_h1 = sess.ingest_pcm_and_listen(block, patience="balanced")
        assert evs_h1[0]["type"] == "user_speech_start"
        evs_h2 = sess.ingest_pcm_and_listen(block[:15360], patience="balanced")
        assert any(e["type"] == "user_turn_committed" for e in evs_h2)

        # Rollover at hard limit preserves encoder cache while restarting LLM context
        sess.listener.hard_reset_sec = 1.5
        model.asr_sequence = ["One.", "Two."]
        model._asr_idx = 0
        sess.listener.reset_window()
        sess.listener.step_block(block)
        restarts_before = model.live_asr_calls[-1]["restarts"]
        sess.listener.step_block(block)
        assert sess.listener.maybe_reset(is_idle=False) is True
        sess.listener.step_block(block)
        assert model.live_asr_calls[-1]["restarts"] == restarts_before + 1
    finally:
        engine.shutdown()


def test_transformers_interleaved_stream_frame1_and_cache_growth() -> None:
    """Interleaved `generate` emits frame-1 24 kHz audio, clean text, and grows `_conv_cache`."""
    tok = FakeTokenizer()
    feat = FakeFeatureExtractor()
    codec = FakeLiquidCodec()
    model = FakeTransformersLfm3Model(tok)
    model.interleaved_frames_per_turn = 2  # 4 total frames
    proc = _make_fake_processor(tok, feat, codec)
    persona = next(p for p in build_personas() if p.id == "naledi")
    settings = EngineSettings(backend="transformers", model_id="LiquidAI/LFM3-Omni-2B")
    engine = TransformersOmniEngine(settings, model=model, processor=proc, codec=codec)
    engine.load()
    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation(persona.system_prompt, tools=None, language="en")
        sess.append_text_user_turn("Hello Naledi!")
        seq_before = sess.conv_tokens

        events: list[dict[str, Any]] = []
        out = sess.generate_assistant_turn(
            persona=persona,
            tools=None,
            force_interleaved=True,
            cancel_event=threading.Event(),
            on_event=events.append,
        )
        assert out.kind == "response"
        assert not out.cancelled
        assert out.text == model.answer_text
        assert out.audio_frames == 4
        # Frame 1 decoded immediately (1440 samples = 60 ms)
        assert out.first_chunk_samples == 1440
        assert codec.decode_calls[0] == {"frames": 1, "finalize": False}
        assert codec.decode_calls[-1]["finalize"] is True
        # Prompt echo `<|im_start|>assistant` must NEVER appear in text_delta events
        joined_deltas = "".join(e["delta"] for e in events if e["type"] == "text_delta")
        assert joined_deltas == model.answer_text
        assert sess.conv_tokens > seq_before
        assert model.generate_calls[-1]["used_cache"] is True
    finally:
        engine.shutdown()


def test_transformers_coop_barge_in_and_rollback() -> None:
    """Cooperative slot polling aborts `generate` immediately and restores exact `_conv_cache`."""
    tok = FakeTokenizer()
    feat = FakeFeatureExtractor()
    codec = FakeLiquidCodec()
    model = FakeTransformersLfm3Model(tok)
    model.asr_sequence = ["Stop right there."]
    proc = _make_fake_processor(tok, feat, codec)
    persona = next(p for p in build_personas() if p.id == "naledi")
    settings = EngineSettings(
        backend="transformers",
        model_id="LiquidAI/LFM3-Omni-2B",
        coop_listen_slot_interval=4,
    )
    engine = TransformersOmniEngine(settings, model=model, processor=proc, codec=codec)
    engine.load()
    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation(persona.system_prompt, tools=None, language="en")
        sess.append_text_user_turn("Tell me everything.")
        seq_pre_answer = sess.conv_tokens
        conv_sum_pre = float(sess._conv_cache.conv_state.sum())

        cancel_ev = threading.Event()
        mic_block = _pcm16_16k_block(blocks=1)
        mic_fed = False

        def poll_mic() -> list[dict[str, Any]]:
            nonlocal mic_fed
            if mic_fed:
                return []
            mic_fed = True
            return sess.ingest_pcm_and_listen(
                mic_block,
                agent_speaking=True,
                recent_agent_text="",
                patience="balanced",
            )

        events: list[dict[str, Any]] = []
        out = sess.generate_assistant_turn(
            persona=persona,
            tools=None,
            force_interleaved=True,
            cancel_event=cancel_ev,
            on_event=events.append,
            poll_mic_fn=poll_mic,
        )
        assert out.cancelled is True
        assert cancel_ev.is_set()
        assert sess.pending_barge_in_pcm is not None
        # During `generate` cancellation, cache must already be restored to pre-turn state
        assert sess.conv_tokens == seq_pre_answer
        assert float(sess._conv_cache.conv_state.sum()) == conv_sum_pre

        # Rollback with 0 played frames drops unplayed assistant turn & prefills buffered barge-in PCM
        sess.rollback_interrupted_assistant_turn(
            played_frames=0,
            interrupted_turn=out.turn_record,
            replace_last_history=False,
        )
        assert sess.pending_barge_in_pcm is None
        assert sess.user_turn_open is True
        assert sess.conv_tokens > seq_pre_answer
    finally:
        engine.shutdown()


def test_transformers_router_tools_and_uncached_filler() -> None:
    """3-way router runs DIRECT, TOOL (with rollback + tool replay), and DELEGATE (uncached filler)."""
    tok = FakeTokenizer()
    feat = FakeFeatureExtractor()
    codec = FakeLiquidCodec()
    model = FakeTransformersLfm3Model(tok)
    proc = _make_fake_processor(tok, feat, codec)
    persona = next(p for p in build_personas() if p.id == "naledi")
    tools = build_tools()
    settings = EngineSettings(backend="transformers", model_id="LiquidAI/LFM3-Omni-2B")
    engine = TransformersOmniEngine(settings, model=model, processor=proc, codec=codec)
    engine.load()
    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation(persona.system_prompt, tools=tools, language="en")

        # 1. DIRECT
        model.router_decision = "DIRECT"
        sess.append_text_user_turn("Hi!")
        dir_plan = sess.run_speculative_selective_batch(
            persona=persona,
            tools=tools,
            dynamic_filler_enabled=True,
            cancel_event=threading.Event(),
            on_event=lambda _e: None,
        )
        assert dir_plan.decision == "direct"
        assert dir_plan.direct_outcome is not None
        assert dir_plan.direct_outcome.audio_frames > 0

        # 2. TOOL
        model.router_decision = "TOOL"
        model.tool_call_on_next_answer = (
            "check_pre_qualification",
            {"monthly_income_zar": 35000},
        )
        sess.append_text_user_turn("I earn R35,000.")
        seq_before_tool = sess.conv_tokens
        tool_plan = sess.run_speculative_selective_batch(
            persona=persona,
            tools=tools,
            dynamic_filler_enabled=True,
            cancel_event=threading.Event(),
            on_event=lambda _e: None,
        )
        assert tool_plan.decision == "tool_calls"
        assert tool_plan.direct_outcome is not None
        assert tool_plan.direct_outcome.tool_calls[0]["function"]["name"] == "check_pre_qualification"
        # Tool call rolled back so history has clean tool exchange when caller invokes append_tool_exchange
        assert sess.conv_tokens == seq_before_tool
        sess.append_tool_exchange(
            tool_plan.direct_outcome.tool_calls,
            json.dumps({"eligible_tier": "Ubuntu Gold"}),
        )
        assert sess.conv_tokens > seq_before_tool

        # 3. DELEGATE (filler runs uncached on clone -> `_conv_cache` not polluted by filler)
        model.router_decision = "DELEGATE: What are the Platinum lounge fees?"
        sess.append_text_user_turn("What are the Platinum lounge fees?")
        seq_before_del = sess.conv_tokens
        del_events: list[dict[str, Any]] = []
        del_plan = sess.run_speculative_selective_batch(
            persona=persona,
            tools=tools,
            dynamic_filler_enabled=True,
            cancel_event=threading.Event(),
            on_event=del_events.append,
        )
        assert del_plan.decision == "delegate"
        assert del_plan.filler_audio_frames > 0
        assert any(e["type"] == "delegation_query_ready" for e in del_events)
        assert sess.conv_tokens == seq_before_del
    finally:
        engine.shutdown()


def test_transformers_e2e_websocket_call_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Full `/ws/call` E2E on `transformers` backend records `e2e_speech_to_audio_ms` cleanly."""
    monkeypatch.delenv("FLAMETREE_AGENT_TOKEN", raising=False)
    tok = FakeTokenizer()
    feat = FakeFeatureExtractor()
    codec = FakeLiquidCodec()
    model = FakeTransformersLfm3Model(tok)
    model.router_decision = "DIRECT"
    model.asr_sequence = ["What is the monthly fee?", "", ""]
    proc = _make_fake_processor(tok, feat, codec)
    calls_dir = tmp_path / "calls"

    cfg = build_config(
        backend="transformers",
        language="en",
        delegation_mode="selective_tool",
        calls_dir=str(calls_dir),
    )
    engine = TransformersOmniEngine(cfg.engine, model=model, processor=proc, codec=codec)
    engine.load()
    try:
        app = create_app(cfg, engine=engine, load_on_startup=False)
        with TestClient(app) as client:
            cfg_resp = client.get("/api/config").json()
            assert cfg_resp["backend"] == "transformers"
            assert cfg_resp["batch_parallel_mode"] == "transformers_direct_cuda"

            with client.websocket_connect("/ws/call") as ws:
                assert ws.receive_json() == {"type": "call_state", "state": "idle"}
                ws.send_json({"type": "start_call", "language": "en"})

                call_id = ""
                t1_turn_id = ""
                t1_perc_ev: dict[str, Any] | None = None
                for _ in range(200):
                    msg = ws.receive()
                    if msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        t = ev.get("type")
                        if t == "call_started":
                            call_id = ev["call_id"]
                        elif t == "assistant_turn_start" and not t1_turn_id:
                            t1_turn_id = ev["turn_id"]
                        elif t == "assistant_turn_end" and ev.get("turn_id") == t1_turn_id:
                            ws.send_json(
                                {
                                    "type": "playback_started",
                                    "turn_id": t1_turn_id,
                                    "buffered_samples": 4800,
                                }
                            )
                        elif (
                            t == "live_ttfa"
                            and "ttfa_perceived_ms" in ev
                            and ev.get("turn_id") == t1_turn_id
                        ):
                            t1_perc_ev = ev
                            break

                assert call_id
                assert t1_perc_ev is not None
                assert t1_perc_ev.get("e2e_speech_to_audio_ms") is None
                ws.send_json({"type": "playback_done", "played_samples": 4800})

                # Stream 960 ms speech block + 40 ms pause + two silent blocks to commit
                ws.send_bytes(_pcm16_16k_block(blocks=1))
                time.sleep(0.04)
                ws.send_bytes(b"\x00" * 30720)
                ws.send_bytes(b"\x00" * 30720)

                t2_turn_id = ""
                t2_perc_ev: dict[str, Any] | None = None
                for _ in range(200):
                    msg = ws.receive()
                    if msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        t = ev.get("type")
                        if t == "assistant_turn_start" and ev.get("turn_id") != t1_turn_id:
                            t2_turn_id = ev["turn_id"]
                        elif t == "assistant_turn_end" and ev.get("turn_id") == t2_turn_id:
                            ws.send_json(
                                {
                                    "type": "playback_started",
                                    "turn_id": t2_turn_id,
                                    "buffered_samples": 4800,
                                }
                            )
                        elif (
                            t == "live_ttfa"
                            and "ttfa_perceived_ms" in ev
                            and ev.get("turn_id") == t2_turn_id
                        ):
                            t2_perc_ev = ev
                            break

                assert t2_perc_ev is not None
                assert t2_perc_ev["ttfa_perceived_ms"] is not None
                assert t2_perc_ev.get("e2e_speech_to_audio_ms") is not None
                assert t2_perc_ev["e2e_speech_to_audio_ms"] >= t2_perc_ev["ttfa_perceived_ms"]

                ws.send_json({"type": "hang_up"})
                for _ in range(40):
                    msg = ws.receive()
                    if msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        if ev.get("type") == "call_state" and ev.get("state") == "idle":
                            break

            detail = client.get(f"/api/calls/{call_id}").json()
            assert detail["metadata"]["counts"]["error_count"] == 0
            e2e_agg = detail["metrics"]["aggregates"]["e2e_speech_to_audio_ms"]
            assert e2e_agg["count"] == 1
            assert e2e_agg["last"] == t2_perc_ev["e2e_speech_to_audio_ms"]
            txt = (calls_dir / call_id / "transcript.txt").read_text(encoding="utf-8")
            assert "E2E:" in txt
    finally:
        engine.shutdown()
