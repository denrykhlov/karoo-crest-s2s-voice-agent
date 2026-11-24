"""Tests for the Linux/CUDA native `vllm_omni.AsyncOmni` backend (`voice_agent/vllm_backend.py`)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
import wave
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from telesales_agent import build_config, build_personas, build_tools
from voice_agent.backend_factory import build_engine
from voice_agent.config import EngineSettings
from voice_agent.protocol import TurnRecord
from voice_agent.server import create_app
from voice_agent.session import CallSession
from voice_agent.vllm_backend import VllmOmniEngine


def _pcm16_24k_sine(frames: int = 2, amplitude: float = 0.08) -> np.ndarray:
    """Create 24 kHz mono float32 waveform (1920 samples per LFM3 codec frame)."""
    samples = frames * 1920
    t = np.arange(samples, dtype=np.float32)
    return (np.sin(2.0 * np.pi * 440.0 * t / 24000.0) * amplitude).astype(np.float32)


def _pcm16_16k_speech_block(blocks: int = 1) -> bytes:
    """Create non-silent 16 kHz mono Int16 PCM bytes (20,480 bytes per 640 ms block)."""
    n_samples = blocks * 10240
    t = np.arange(n_samples, dtype=np.float32)
    wave_i16 = (np.sin(2.0 * np.pi * 300.0 * t / 16000.0) * 1200).astype("<i2")
    return wave_i16.tobytes()


class _FakeTensor:
    """Minimal torch-Tensor-like wrapper exercising `.detach().cpu().float().numpy()`."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = np.asarray(arr, dtype=np.float32)

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def float(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeMultimodalPayload(Mapping):
    """Non-dict Mapping mirroring `vllm_omni.outputs.mm_outputs.MultimodalPayload`."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


assert not isinstance(_FakeMultimodalPayload({}), dict)


@dataclass
class FakeCompletionOutput:
    text: str = ""
    multimodal_output: Any = field(default_factory=dict)


@dataclass
class FakeOmniRequestOutput:
    request_id: str
    stage_id: int
    final_output_type: str
    outputs: list[FakeCompletionOutput]
    finished: bool = False
    error: str | None = None
    multimodal_output: Any = field(default_factory=dict)


class FakeTokenizer:
    """Reproduces key LFM3 chat_template contracts without requiring local HF weights."""

    def __init__(self) -> None:
        self.template_calls: list[dict[str, Any]] = []
        self.encode_calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
    ) -> str:
        assert tokenize is False
        self.template_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "add_generation_prompt": add_generation_prompt,
                "continue_final_message": continue_final_message,
            }
        )
        parts: list[str] = ["<|startoftext|>"]
        for idx, msg in enumerate(messages):
            role = msg["role"]
            content = msg.get("content")
            parts.append(f"<|im_start|>{role}\n")
            if role == "system" and tools:
                parts.append("<|tool_list_start|>")
                parts.append(json.dumps(tools))
                parts.append("<|tool_list_end|>")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "audio":
                        parts.append("<|vision_start|><|audio|><|vision_end|>")
                    elif isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
            elif content is not None:
                parts.append(str(content))

            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc["function"]
                    args = fn.get("arguments")
                    # LFM3 chat_template.jinja raises if tool_call arguments is a JSON string.
                    if not isinstance(args, dict):
                        raise ValueError("Tool call arguments must be a mapping")
                    parts.append(f"<tool_call><function={fn['name']}>")
                    for k, v in args.items():
                        parts.append(f"<parameter={k}>{json.dumps(v)}</parameter>")
                    parts.append("</function></tool_call>")

            is_last = idx == len(messages) - 1
            if not (continue_final_message and is_last):
                parts.append("<|im_end|>\n")

        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n<think></think>")
        return "".join(parts)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        self.encode_calls.append({"length": len(text), "add_special_tokens": add_special_tokens})
        if add_special_tokens:
            raise AssertionError("Must pass add_special_tokens=False to avoid double BOS")
        return list(text.encode("utf-8"))

    @staticmethod
    def decode_prompt_ids(token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8")


class FakeAsyncOmni:
    """In-process fake implementing `vllm_omni.AsyncOmni` streaming & abort semantics."""

    num_stages = 2

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []
        self.aborted_ids: list[str] = []
        self.shutdown_called: bool = False
        self.asr_sequence: list[str] = []
        self.asr_idx: int = 0
        self.router_decision: str = "DIRECT"
        self.query_extract_text: str = "What is the annual fee and lounge benefit on Ubuntu Platinum?"
        self.tool_call_on_next_answer: tuple[str, dict[str, Any]] | None = None
        self.omit_chat_audio: bool = False
        self.answer_text: str = "Hello! I am Naledi from Karoo Crest Bank. How can I help today?"
        self.filler_text: str = "Let me check our policy details for you right away."
        self.summary_text: str = "Customer asked about Ubuntu Rewards Platinum annual fee."
        self.stream_delay_s: float = 0.0
        self.tts_stream_delay_s: float = 0.0
        self.stream_chunks: int = 2
        self.audio_on_all_steps: bool = False

    async def abort(self, request_id: str | list[str]) -> None:
        ids = [request_id] if isinstance(request_id, str) else list(request_id)
        with self.lock:
            self.aborted_ids.extend(ids)

    def shutdown(self) -> None:
        self.shutdown_called = True

    def _is_aborted(self, request_id: str) -> bool:
        with self.lock:
            return request_id in self.aborted_ids

    async def generate(
        self,
        prompt: dict[str, Any],
        request_id: str = "",
        *,
        sampling_params_list: list[Any] | None = None,
        output_modalities: list[str] | None = None,
        **_kwargs: Any,
    ) -> AsyncGenerator[FakeOmniRequestOutput, None]:
        assert isinstance(prompt, dict) and "prompt_token_ids" in prompt
        assert sampling_params_list is not None
        assert len(sampling_params_list) == self.num_stages, (
            f"Expected {self.num_stages} stage SamplingParams, got {len(sampling_params_list)}"
        )
        rendered = FakeTokenizer.decode_prompt_ids(prompt["prompt_token_ids"])
        # Never duplicate BOS token.
        assert rendered.count("<|startoftext|>") == 1

        with self.lock:
            self.calls.append(
                {
                    "request_id": request_id,
                    "rendered": rendered,
                    "multi_modal_data": prompt.get("multi_modal_data"),
                    "mm_processor_kwargs": prompt.get("mm_processor_kwargs"),
                    "sampling_params_list": sampling_params_list,
                    "output_modalities": list(output_modalities or []),
                }
            )

        # 1. LiveASR assistant-turn block
        if "<|im_start|>system\nPerform live ASR.<|im_end|>" in rendered:
            mm = prompt.get("multi_modal_data") or {}
            audio_val = mm.get("audio")
            assert isinstance(audio_val, tuple) and audio_val[1] == 16000
            assert isinstance(audio_val[0], np.ndarray) and audio_val[0].dtype == np.float32
            assert "<think>" not in rendered
            assert "<|vision_start|>" not in rendered and "<|vision_end|>" not in rendered
            assert rendered.endswith("<|im_start|>assistant\n<|audio|>")
            assert prompt.get("mm_processor_kwargs") == {"audio_markers": False}
            with self.lock:
                if self.asr_idx < len(self.asr_sequence):
                    text = self.asr_sequence[self.asr_idx]
                    self.asr_idx += 1
                else:
                    text = self.asr_sequence[-1] if self.asr_sequence else ""
            yield FakeOmniRequestOutput(
                request_id=request_id,
                stage_id=0,
                final_output_type="text",
                outputs=[FakeCompletionOutput(text=text)],
                finished=True,
            )
            return

        # 2. PTT turn ASR
        if "<|im_start|>system\nPerform ASR in " in rendered:
            mm = prompt.get("multi_modal_data") or {}
            audio_val = mm.get("audio")
            assert isinstance(audio_val, tuple) and audio_val[1] == 16000
            yield FakeOmniRequestOutput(
                request_id=request_id,
                stage_id=0,
                final_output_type="text",
                outputs=[FakeCompletionOutput(text="Hello Ubuntu Gold.")],
                finished=True,
            )
            return

        # 3. Text-only router / query extract / compaction
        if "Output strictly" in rendered:
            if self.stream_delay_s > 0:
                await asyncio.sleep(self.stream_delay_s * 0.4)
            yield FakeOmniRequestOutput(
                request_id=request_id,
                stage_id=0,
                final_output_type="text",
                outputs=[FakeCompletionOutput(text=self.router_decision)],
                finished=True,
            )
            return

        if "Output ONLY one concise standalone question" in rendered:
            yield FakeOmniRequestOutput(
                request_id=request_id,
                stage_id=0,
                final_output_type="text",
                outputs=[FakeCompletionOutput(text=self.query_extract_text)],
                finished=True,
            )
            return

        if "compact factual call memory summary" in rendered:
            yield FakeOmniRequestOutput(
                request_id=request_id,
                stage_id=0,
                final_output_type="text",
                outputs=[FakeCompletionOutput(text=self.summary_text)],
                finished=True,
            )
            return

        # 4. Audio-only TTS fallback (`Perform TTS in ...`)
        if "<|im_start|>system\nPerform TTS in " in rendered:
            assert rendered.endswith("<|audio_start|>")
            assert output_modalities == ["audio"]
            tts_delay = (
                self.tts_stream_delay_s if self.tts_stream_delay_s > 0 else self.stream_delay_s
            )
            if tts_delay > 0:
                total_tts_steps = max(2, self.stream_chunks)
                for step_idx in range(total_tts_steps):
                    await asyncio.sleep(tts_delay)
                    if self._is_aborted(request_id):
                        return
                    yield FakeOmniRequestOutput(
                        request_id=request_id,
                        stage_id=1,
                        final_output_type="audio",
                        outputs=[
                            FakeCompletionOutput(
                                text="",
                                multimodal_output=_FakeMultimodalPayload(
                                    {"model_outputs": [_FakeTensor(_pcm16_24k_sine(frames=2))]}
                                ),
                            )
                        ],
                        finished=(step_idx == total_tts_steps - 1),
                    )
                return
            # Emit metadata-only empty chunks first, then `model_outputs` via non-dict Mapping
            yield FakeOmniRequestOutput(
                request_id=request_id,
                stage_id=1,
                final_output_type="audio",
                outputs=[FakeCompletionOutput(text="", multimodal_output=_FakeMultimodalPayload({"model_outputs": None}))],
                finished=False,
            )
            yield FakeOmniRequestOutput(
                request_id=request_id,
                stage_id=1,
                final_output_type="audio",
                outputs=[
                    FakeCompletionOutput(
                        text="",
                        multimodal_output=_FakeMultimodalPayload({"model_outputs": np.zeros((0,), dtype=np.float32)}),
                    )
                ],
                finished=False,
            )
            yield FakeOmniRequestOutput(
                request_id=request_id,
                stage_id=1,
                final_output_type="audio",
                outputs=[
                    FakeCompletionOutput(
                        text="",
                        multimodal_output=_FakeMultimodalPayload(
                            {"model_outputs": [_FakeTensor(_pcm16_24k_sine(frames=2))]}
                        ),
                    )
                ],
                finished=True,
            )
            return

        # 5. Streaming Omni chat turn
        lower = rendered.lower()
        is_filler = (
            "already spoke the holding phrase" not in lower
            and ("holding phrase" in lower or "filler" in lower)
        )
        with self.lock:
            has_tools_in_prompt = "<|tool_list_start|>" in rendered
            pending_tool = None if (is_filler or not has_tools_in_prompt) else self.tool_call_on_next_answer
            if pending_tool is not None and not self.stream_delay_s:
                self.tool_call_on_next_answer = None
            omit_audio = self.omit_chat_audio

        if pending_tool is not None:
            t_name, t_args = pending_tool
            xml = f"<tool_call>\n<function={t_name}>\n"
            for k, v in t_args.items():
                xml += f"<parameter={k}>\n{json.dumps(v)}\n</parameter>\n"
            xml += "</function>\n</tool_call>"
            half = len(xml) // 2
            yield FakeOmniRequestOutput(
                request_id=request_id,
                stage_id=0,
                final_output_type="text",
                outputs=[FakeCompletionOutput(text=xml[:half])],
                finished=False,
            )
            yield FakeOmniRequestOutput(
                request_id=request_id,
                stage_id=0,
                final_output_type="text",
                outputs=[FakeCompletionOutput(text=xml[half:])],
                finished=True,
            )
            return

        text_out = self.filler_text if is_filler else self.answer_text
        mid = max(1, len(text_out) // 2)
        pieces = [text_out[:mid], text_out[mid:]]
        n_steps = max(len(pieces), self.stream_chunks)
        for step_idx in range(n_steps):
            if self.stream_delay_s > 0:
                await asyncio.sleep(self.stream_delay_s)
            if self._is_aborted(request_id):
                return
            if step_idx < len(pieces) and pieces[step_idx]:
                yield FakeOmniRequestOutput(
                    request_id=request_id,
                    stage_id=0,
                    final_output_type="text",
                    outputs=[FakeCompletionOutput(text=pieces[step_idx])],
                    finished=False,
                )
            if not omit_audio:
                # Include one empty metadata chunk before second audio chunk to verify filtering.
                if step_idx == 1:
                    yield FakeOmniRequestOutput(
                        request_id=request_id,
                        stage_id=1,
                        final_output_type="audio",
                        outputs=[
                            FakeCompletionOutput(
                                text="",
                                multimodal_output=_FakeMultimodalPayload(
                                    {"audio": np.zeros((0,), dtype=np.float32)}
                                ),
                            )
                        ],
                        finished=False,
                    )
                if step_idx < 2 or self.audio_on_all_steps:
                    wav = _pcm16_24k_sine(frames=2, amplitude=0.05 + 0.02 * step_idx)
                    yield FakeOmniRequestOutput(
                        request_id=request_id,
                        stage_id=1,
                        final_output_type="audio",
                        outputs=[
                            FakeCompletionOutput(
                                text="",
                                multimodal_output=_FakeMultimodalPayload({"audio": _FakeTensor(wav)}),
                            )
                        ],
                        finished=(step_idx == n_steps - 1),
                    )
        yield FakeOmniRequestOutput(
            request_id=request_id,
            stage_id=0,
            final_output_type="text",
            outputs=[FakeCompletionOutput(text="")],
            finished=True,
        )


def test_vllm_backend_imports_without_mlx() -> None:
    """Importing the app and building a `--backend vllm` config/engine must never import `mlx`."""
    script = """
import sys
import telesales_agent
from voice_agent.backend_factory import build_engine
import voice_agent.server
import voice_agent.session
import voice_agent.vllm_backend

cfg = telesales_agent.build_config(backend="vllm")
assert cfg.engine.backend == "vllm"
assert cfg.engine.model_id == "LiquidAI/LFM3-Omni-2B"
assert cfg.engine.vllm_deploy_config == "deploy/lfm3_omni_runpod.yaml"
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


def test_vllm_engine_load_and_config_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("FLAMETREE_AGENT_TOKEN", raising=False)
    omni = FakeAsyncOmni()
    tok = FakeTokenizer()

    cfg = build_config(
        backend="vllm",
        vllm_api_key="super-secret-vllm-key",
        calls_dir=str(tmp_path / "calls"),
    )
    engine = VllmOmniEngine(cfg.engine, omni_client=omni, tokenizer=tok)
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
            assert pub_cfg["vllm_api_mode"] == "native_async_omni"
            assert pub_cfg["vllm_deploy_config"] == "deploy/lfm3_omni_runpod.yaml"
            assert pub_cfg["batch_parallel_mode"] == "vllm_continuous_batch"
            assert "super-secret-vllm-key" not in json.dumps(pub_cfg)
    finally:
        engine.shutdown()
        assert omni.shutdown_called is True


def test_vllm_live_s2s_listener_diff_echo_endpoint_and_barge_in() -> None:
    omni = FakeAsyncOmni()
    tok = FakeTokenizer()
    settings = EngineSettings(backend="vllm", model_id="LiquidAI/LFM3-Omni-2B")
    engine = VllmOmniEngine(settings, omni_client=omni, tokenizer=tok)
    engine.load()
    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation("System prompt.", tools=None, language="en")
        block = _pcm16_16k_speech_block(blocks=1)

        # 0. Silence returns empty without starting a user turn
        omni.asr_sequence = ["<|text_end|>"]
        assert sess.ingest_pcm_and_listen(block, patience="balanced") == []
        assert not sess.user_speaking

        # 1. Echo while agent speaking is suppressed
        omni.asr_idx = 0
        omni.asr_sequence = ["Naledi from Karoo Crest"]
        evs_echo = sess.ingest_pcm_and_listen(
            block,
            agent_speaking=True,
            recent_agent_text="Hello, I am Naledi from Karoo Crest Bank.",
            patience="balanced",
        )
        assert evs_echo == []
        assert not sess.user_speaking

        # 2. Genuine barge-in + overlap dedup + commit window reset
        omni.asr_idx = 0
        omni.asr_sequence = [
            "Good",
            "Good morning.",
            "morning.",  # trailing overlap only -> empty delta -> endpoints
            "",
        ]
        evs_bi = sess.ingest_pcm_and_listen(
            block,
            agent_speaking=True,
            recent_agent_text="Hello, I am Naledi from Karoo Crest Bank.",
            patience="snappy",
        )
        assert [e["type"] for e in evs_bi] == ["barge_in_detected", "user_speech_start"]
        assert sess.user_speaking

        evs_delta = sess.ingest_pcm_and_listen(block, patience="snappy")
        assert evs_delta and evs_delta[0]["type"] == "user_speech_delta"
        assert evs_delta[0]["text"] == "Good morning."

        resets_before_commit = sess.listener.reset_count
        evs_commit = sess.ingest_pcm_and_listen(block, patience="snappy")
        assert any(e["type"] == "user_turn_committed" for e in evs_commit)
        assert not sess.user_speaking
        assert sess.latest_user_text() == "Good morning."
        assert sess.listener.reset_count == resets_before_commit + 1
        assert sess.listener._prev_tail_pcm == b""
        assert sess.listener._prev_chunk_text == ""
        assert all(call["add_special_tokens"] is False for call in tok.encode_calls)
    finally:
        engine.shutdown()


def test_vllm_ptt_turn_asr() -> None:
    omni = FakeAsyncOmni()
    tok = FakeTokenizer()
    settings = EngineSettings(backend="vllm", model_id="LiquidAI/LFM3-Omni-2B")
    engine = VllmOmniEngine(settings, omni_client=omni, tokenizer=tok)
    engine.load()
    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation("System prompt.", tools=None, language="en")
        block = _pcm16_16k_speech_block(blocks=1)
        ptt_evs = sess.flush_ptt_audio(block)
        assert [e["type"] for e in ptt_evs] == ["user_speech_start", "user_turn_committed"]
        assert ptt_evs[1]["text"] == "Hello Ubuntu Gold."
        assert sess.listener.window_seconds == 0.0
    finally:
        engine.shutdown()


def test_vllm_tool_call_and_history_rendering() -> None:
    omni = FakeAsyncOmni()
    tok = FakeTokenizer()
    settings = EngineSettings(backend="vllm", model_id="LiquidAI/LFM3-Omni-2B")
    engine = VllmOmniEngine(settings, omni_client=omni, tokenizer=tok)
    engine.load()
    persona = next(p for p in build_personas() if p.id == "naledi")
    tools = build_tools()
    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation(persona.system_prompt, tools=tools, language="en")
        sess.append_text_user_turn("I earn R35,000 a month. Which card do I qualify for?")

        # Pass raw JSON-string args once too, verifying template normalizes to dict
        omni.tool_call_on_next_answer = (
            "check_pre_qualification",
            {"monthly_income_zar": 35000},
        )
        events: list[dict[str, Any]] = []
        tc_out = sess.generate_assistant_turn(
            persona=persona,
            tools=tools,
            force_interleaved=False,
            cancel_event=threading.Event(),
            on_event=events.append,
        )
        assert tc_out.kind == "tool_calls"
        assert tc_out.tool_calls[0]["function"]["name"] == "check_pre_qualification"
        assert tc_out.tool_calls[0]["function"]["arguments"] == {"monthly_income_zar": 35000}
        # Tool call markup must be suppressed from UI text_delta events
        assert not [e for e in events if e["type"] == "text_delta"]
        assert any(call["tools"] for call in tok.template_calls)

        # Feed tool exchange back with a string-encoded arguments copy to test normalization
        string_arg_calls = [
            {
                "type": "function",
                "function": {
                    "name": "check_pre_qualification",
                    "arguments": '{"monthly_income_zar": 35000}',
                },
            }
        ]
        sess.append_tool_exchange(
            string_arg_calls,
            json.dumps({"eligible_tier": "Ubuntu Gold", "indicative_limit": 55000}),
        )
        follow_events: list[dict[str, Any]] = []
        reply_out = sess.generate_assistant_turn(
            persona=persona,
            tools=None,
            force_interleaved=True,
            cancel_event=threading.Event(),
            on_event=follow_events.append,
        )
        assert reply_out.kind == "response"
        assert reply_out.audio_frames == 4
        assert any(e["type"] == "audio_chunk" for e in follow_events)
    finally:
        engine.shutdown()


def test_vllm_speculative_selective_direct_and_delegate_abort() -> None:
    omni = FakeAsyncOmni()
    tok = FakeTokenizer()
    settings = EngineSettings(
        backend="vllm",
        model_id="LiquidAI/LFM3-Omni-2B",
        compaction_trigger_tokens=1,
    )
    engine = VllmOmniEngine(settings, omni_client=omni, tokenizer=tok)
    engine.load()
    persona = next(p for p in build_personas() if p.id == "naledi")
    tools = build_tools()
    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation(persona.system_prompt, tools=tools, language="en")
        sess.append_text_user_turn("Start call now.")
        sess.generate_assistant_turn(
            persona=persona,
            tools=None,
            force_interleaved=True,
            cancel_event=threading.Event(),
            on_event=lambda _e: None,
        )
        sess.finalize_greeting_turn()
        assert sess.history[0].text == "Hello?"

        # 1. DIRECT route
        omni.router_decision = "DIRECT"
        sess.append_text_user_turn("Hi Naledi!")
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

        # 2. DELEGATE route aborts speculative generation and speaks dynamic filler
        omni.stream_delay_s = 0.08
        omni.stream_chunks = 6
        omni.router_decision = "DELEGATE: What are the Ubuntu Platinum airport lounge rules?"
        sess.append_text_user_turn("Tell me Platinum airport lounge rules.")
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
        assert len(omni.aborted_ids) >= 1
        assert del_plan.filler_audio_frames > 0
        assert any(e["type"] == "delegation_query_ready" for e in del_events)

        # Compaction test
        omni.stream_delay_s = 0.0
        omni.stream_chunks = 2
        for i in range(6):
            sess.append_text_user_turn(f"Follow-up question {i}")
        assert len(sess.history) > 6
        assert sess.maybe_compact_context() is True
        assert sess.compaction_count == 1
        assert len(sess.history) == 6
    finally:
        engine.shutdown()


def test_vllm_filler_and_query_batch() -> None:
    omni = FakeAsyncOmni()
    tok = FakeTokenizer()
    settings = EngineSettings(backend="vllm", model_id="LiquidAI/LFM3-Omni-2B")
    engine = VllmOmniEngine(settings, omni_client=omni, tokenizer=tok)
    engine.load()
    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation("Sys prompt", tools=build_tools(), language="en")
        sess.append_text_user_turn("Tell me Platinum FX fees and lounge visits.")
        events: list[dict[str, Any]] = []
        plan = sess.run_filler_and_query_batch(
            cancel_event=threading.Event(),
            on_event=events.append,
        )
        assert plan.decision == "delegate"
        assert "Platinum" in plan.delegation_query
        assert plan.filler_audio_frames == 4
        assert any(e["type"] == "delegation_query_ready" for e in events)
        assert any(e["type"] == "audio_chunk" for e in events)
    finally:
        engine.shutdown()


def test_vllm_mid_gen_barge_in_no_deadlock_and_abort() -> None:
    """Mid-gen LiveASR polling runs concurrently with streaming without deadlocking `_loop`."""
    omni = FakeAsyncOmni()
    omni.stream_delay_s = 0.12
    omni.stream_chunks = 10
    omni.asr_sequence = ["Hold on please."]
    tok = FakeTokenizer()
    cfg = build_config(backend="vllm", calls_dir="")
    engine = VllmOmniEngine(cfg.engine, omni_client=omni, tokenizer=tok)
    engine.load()
    persona = next(p for p in build_personas() if p.id == "naledi")

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

    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation(persona.system_prompt, tools=None, language="en")
        sess.append_text_user_turn("Explain the rewards.")
        cancel_ev = threading.Event()
        mic_block = _pcm16_16k_speech_block(blocks=1)
        poll_count = 0

        def poll_mic() -> list[dict[str, Any]]:
            nonlocal poll_count
            poll_count += 1
            return sess.ingest_pcm_and_listen(
                mic_block,
                agent_speaking=True,
                recent_agent_text="Hello! I am Naledi from Karoo Crest Bank.",
                patience="snappy",
            )

        out = sess.generate_assistant_turn(
            persona=persona,
            tools=None,
            force_interleaved=True,
            cancel_event=cancel_ev,
            on_event=lambda _e: None,
            poll_mic_fn=poll_mic,
        )
        assert poll_count >= 1
        assert cancel_ev.is_set()
        assert out.cancelled is True
        assert out.kind == "cancelled"
        assert len(omni.aborted_ids) >= 1

        # Verify CallSession mid-gen rollback works with vllm_backend list-based raw_audio_codes
        async def _verify_session_rollback() -> None:
            ws = _FakeWebSocket()
            call_session = CallSession(ws, config=cfg, engine=engine)
            call_session.loop = asyncio.get_running_loop()
            pipe = call_session._ensure_pipeline()
            pipe.initialize_conversation("Sys prompt", tools=None, language="en")
            pipe.append_text_user_turn("Tell me about Karoo Crest rewards.")

            interrupted = TurnRecord(
                role="assistant",
                text="We offer great lounge access cashback rewards and travel cover.",
                raw_audio_codes=[None] * 10,
            )
            call_session._played_samples_24k = 4 * 1920
            await call_session._handle_mid_gen_barge_in(
                turn_id="ast-mid-1",
                stream_state={
                    "turn_started_ui": True,
                    "first_audio_ts": 100.2,
                    "first_token_ts": 100.05,
                    "chunks_sent": 2,
                    "first_chunk_samples": 1920,
                    "filler_prefix": "",
                    "display_text": interrupted.text,
                },
                interrupted_turn=interrupted,
                mid_gen_mic_events=[{"type": "barge_in_detected", "heard_text": "Wait stop"}],
                filler_frames=0,
                turn_ctx={
                    "t_turn_start": 100.0,
                    "ttft_ms": 50.0,
                    "route_decision": "direct",
                    "after_barge_in": True,
                },
            )
            assert pipe.latest_assistant_text().endswith("...")
            trunc_msgs = [m for m in ws.sent_json if m.get("type") == "assistant_turn_truncated"]
            assert len(trunc_msgs) == 1
            assert trunc_msgs[0]["metrics"]["ttft_ms"] == 50.0

        asyncio.run(_verify_session_rollback())
    finally:
        engine.shutdown()


def test_vllm_tts_fallback_audio_only() -> None:
    omni = FakeAsyncOmni()
    omni.omit_chat_audio = True
    tok = FakeTokenizer()
    settings = EngineSettings(backend="vllm", model_id="LiquidAI/LFM3-Omni-2B")
    engine = VllmOmniEngine(settings, omni_client=omni, tokenizer=tok)
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
        assert out.audio_chunks_sent == 1
        pcm_chunks = [e["pcm16"] for e in events if e["type"] == "audio_chunk"]
        assert len(pcm_chunks) == 1
        assert len(pcm_chunks[0]) == 2 * 1920 * 2
    finally:
        engine.shutdown()


def test_vllm_call_session_websocket_e2e_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("FLAMETREE_AGENT_TOKEN", raising=False)
    omni = FakeAsyncOmni()
    omni.router_decision = "DIRECT"
    tok = FakeTokenizer()
    calls_dir = tmp_path / "calls"

    cfg = build_config(
        backend="vllm",
        language="en",
        delegation_mode="selective_tool",
        calls_dir=str(calls_dir),
    )
    engine = VllmOmniEngine(cfg.engine, omni_client=omni, tokenizer=tok)
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
                turn_event_order: dict[str, list[str]] = {}
                audio_bytes_total = 0

                for _ in range(200):
                    msg = ws.receive()
                    if msg.get("bytes") is not None:
                        audio_bytes_total += len(msg["bytes"])
                    elif msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        t = ev.get("type")
                        tid = ev.get("turn_id")
                        if tid and t in ("assistant_turn_start", "live_ttft", "live_ttfa", "assistant_turn_end"):
                            turn_event_order.setdefault(tid, []).append(t)
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
                        tid = ev.get("turn_id")
                        if tid and t in ("assistant_turn_start", "live_ttft", "live_ttfa", "assistant_turn_end"):
                            turn_event_order.setdefault(tid, []).append(t)
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
                tid = tev["turn_id"]
                order = turn_event_order[tid]
                assert "assistant_turn_start" in order and "live_ttft" in order
                assert order.index("assistant_turn_start") < order.index("live_ttft")
                assert isinstance(tev.get("metrics"), dict)
                assert tev["metrics"]["ttft_ms"] is not None
                assert tev["metrics"]["ttfa_server_ms"] is not None

            detail = client.get(f"/api/calls/{call_id}").json()
            assert detail["metadata"]["status"] == "completed"
            assert detail["metadata"]["counts"]["error_count"] == 0
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


def test_vllm_live_s2s_sliding_window_160s() -> None:
    """LiveASR maintains 960 ms overlap + 640 ms new block = 1.60 s O(1) sliding window."""
    omni = FakeAsyncOmni()
    omni.asr_sequence = ["", "", ""]
    tok = FakeTokenizer()
    engine = VllmOmniEngine(
        EngineSettings(backend="vllm", model_id="LiquidAI/LFM3-Omni-2B"),
        omni_client=omni,
        tokenizer=tok,
    )
    engine.load()
    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation("Sys prompt.", tools=None, language="en")
        block = _pcm16_16k_speech_block(blocks=1)  # 10,240 samples (640 ms)

        sess.ingest_pcm_and_listen(block, patience="balanced")
        sess.ingest_pcm_and_listen(block, patience="balanced")
        sess.ingest_pcm_and_listen(block, patience="balanced")

        asr_calls = [
            c for c in omni.calls if "<|im_start|>system\nPerform live ASR.<|im_end|>" in c["rendered"]
        ]
        assert len(asr_calls) == 3
        lengths = [len(c["multi_modal_data"]["audio"][0]) for c in asr_calls]
        assert lengths == [10240, 20480, 25600]
    finally:
        engine.shutdown()


def test_vllm_tts_fallback_mid_gen_barge_in() -> None:
    """Mid-gen voice barge-in aborts active `_stream_tts_fallback` streaming cleanly."""
    omni = FakeAsyncOmni()
    omni.omit_chat_audio = True
    omni.stream_delay_s = 0.0
    omni.tts_stream_delay_s = 0.08
    omni.stream_chunks = 10
    omni.asr_sequence = ["Hold on a second."]
    tok = FakeTokenizer()
    cfg = build_config(backend="vllm", calls_dir="")
    engine = VllmOmniEngine(cfg.engine, omni_client=omni, tokenizer=tok)
    engine.load()
    persona = next(p for p in build_personas() if p.id == "naledi")
    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation(persona.system_prompt, tools=None, language="en")
        sess.append_text_user_turn("Tell me about monthly fees.")
        hist_before = len(sess.history)

        cancel_ev = threading.Event()
        mic_block = _pcm16_16k_speech_block(blocks=1)
        poll_count = 0

        def poll_mic() -> list[dict[str, Any]]:
            nonlocal poll_count
            poll_count += 1
            return sess.ingest_pcm_and_listen(
                mic_block,
                agent_speaking=True,
                recent_agent_text=omni.answer_text,
                patience="snappy",
            )

        out = sess.generate_assistant_turn(
            persona=persona,
            tools=None,
            force_interleaved=False,
            cancel_event=cancel_ev,
            on_event=lambda _e: None,
            poll_mic_fn=poll_mic,
        )
        assert poll_count >= 1
        assert cancel_ev.is_set()
        assert out.cancelled is True
        assert out.kind == "cancelled"
        assert out.used_tts_fallback is True
        assert len(omni.aborted_ids) >= 1
        # Cancelled turn must not be appended verbatim to history; rollback inserts truncated portion.
        assert len(sess.history) == hist_before
    finally:
        engine.shutdown()


def test_vllm_websocket_mid_gen_voice_barge_in_immediate_and_rollback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Voice barge-in mid-generation emits `barge_in` immediately and rolls back direct turn."""
    monkeypatch.delenv("FLAMETREE_AGENT_TOKEN", raising=False)
    omni = FakeAsyncOmni()
    omni.router_decision = "DIRECT"
    omni.answer_text = (
        "We offer Ubuntu Classic Gold and Protea Platinum credit cards "
        "with cashback rewards and airport lounge visits."
    )
    tok = FakeTokenizer()
    calls_dir = tmp_path / "calls"

    cfg = build_config(
        backend="vllm",
        language="en",
        delegation_mode="selective_tool",
        calls_dir=str(calls_dir),
    )
    engine = VllmOmniEngine(cfg.engine, omni_client=omni, tokenizer=tok)
    engine.load()
    try:
        app = create_app(cfg, engine=engine, load_on_startup=False)
        with TestClient(app) as client:
            with client.websocket_connect("/ws/call") as ws:
                assert ws.receive_json() == {"type": "call_state", "state": "idle"}
                ws.send_json({"type": "start_call", "language": "en"})

                call_id = ""
                for _ in range(200):
                    msg = ws.receive()
                    if msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        if ev.get("type") == "call_started":
                            call_id = ev["call_id"]
                        elif ev.get("type") == "delivery_stats" and ev.get("stats", {}).get("turn_count", 0) >= 1:
                            break
                assert call_id
                ws.send_json({"type": "playback_done", "played_samples": 4 * 1920})

                # Slow Turn 2 streaming so caller voice audio catches Turn 2 mid-generation.
                omni.stream_delay_s = 0.08
                omni.stream_chunks = 14
                omni.audio_on_all_steps = True
                omni.asr_idx = 0
                omni.asr_sequence = ["Hold on please.", ""]

                saw_turn2_start = False
                sent_voice = False
                saw_barge_in = False
                saw_turn2_end = False
                saw_truncated = False
                turn3_done = False

                ws.send_json({"type": "user_text", "text": "Tell me everything about the cards."})

                for _ in range(400):
                    msg = ws.receive()
                    if msg.get("bytes") is not None:
                        if not saw_turn2_start:
                            continue
                        if not sent_voice:
                            sent_voice = True
                            ws.send_json({"type": "playback_progress", "played_samples": 3 * 1920})
                            ws.send_bytes(_pcm16_16k_speech_block(blocks=1))
                    elif msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        t = ev.get("type")
                        if t == "assistant_turn_start":
                            saw_turn2_start = True
                            continue
                        if not saw_turn2_start:
                            continue
                        if t == "barge_in":
                            saw_barge_in = True
                            assert not saw_turn2_end, "Turn 2 completed instead of being interrupted mid-gen"
                            # Finish utterance with silence so "Hold on please." commits cleanly for Turn 3.
                            omni.stream_delay_s = 0.0
                            omni.audio_on_all_steps = False
                            silence_block = b"\x00" * 20480
                            ws.send_bytes(silence_block)
                            ws.send_bytes(silence_block)
                        elif t == "assistant_turn_truncated":
                            saw_truncated = True
                            assert ev.get("text", "").endswith("...")
                        elif t == "assistant_turn_end":
                            if not saw_barge_in:
                                saw_turn2_end = True
                            else:
                                turn3_done = True
                        elif (
                            t == "delivery_stats"
                            and saw_barge_in
                            and turn3_done
                            and ev.get("stats", {}).get("turn_count", 0) >= 3
                        ):
                            break

                assert saw_barge_in is True
                assert saw_truncated is True
                assert turn3_done is True

                ws.send_json({"type": "hang_up"})
                for _ in range(40):
                    msg = ws.receive()
                    if msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        if ev.get("type") == "call_state" and ev.get("state") == "idle":
                            break

            detail = client.get(f"/api/calls/{call_id}").json()
            assert detail["metadata"]["counts"]["error_count"] == 0
            trace_events = detail["trace_events"]
            mid_bis = [t for t in trace_events if t.get("event") == "barge_in_mid_gen"]
            assert len(mid_bis) >= 1
            assert mid_bis[0]["data"]["played_frames"] == 3
            assert mid_bis[0]["data"]["truncated_text"]
    finally:
        engine.shutdown()


def test_vllm_barge_in_during_delegation_wait_cancels_delegation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caller speech while awaiting Flametree delegation cancels delegation and suppresses stale answer."""
    from voice_agent.delegation import DelegationResult

    class _SlowCancellingDelegate:
        def __init__(self) -> None:
            self.is_live = False
            self.last_error: str | None = None
            self.current_session_id: str | None = None
            self.call_num = 0
            self.started_first = threading.Event()
            self.cancelled_calls = 0

        async def start_session(self, call_id: str, language: str) -> str | None:
            return None

        async def close_session(self) -> None:
            return None

        async def query(self, question: str, language: str) -> DelegationResult:
            self.call_num += 1
            if self.call_num == 1:
                self.started_first.set()
                try:
                    await asyncio.sleep(5.0)
                except asyncio.CancelledError:
                    self.cancelled_calls += 1
                    raise
            return DelegationResult(
                query=question,
                answer="Ubuntu Gold costs R95 per month.",
                source="stub_kb",
                latency_ms=1.0,
            )

    delegate_stub = _SlowCancellingDelegate()
    monkeypatch.delenv("FLAMETREE_AGENT_TOKEN", raising=False)
    monkeypatch.setattr("voice_agent.session.build_delegate", lambda _cfg: delegate_stub)

    omni = FakeAsyncOmni()
    tok = FakeTokenizer()
    calls_dir = tmp_path / "calls"
    cfg = build_config(
        backend="vllm",
        language="en",
        delegation_mode="always_delegate",
        dynamic_filler=True,
        calls_dir=str(calls_dir),
    )
    engine = VllmOmniEngine(cfg.engine, omni_client=omni, tokenizer=tok)
    engine.load()
    try:
        app = create_app(cfg, engine=engine, load_on_startup=False)
        with TestClient(app) as client:
            with client.websocket_connect("/ws/call") as ws:
                assert ws.receive_json() == {"type": "call_state", "state": "idle"}
                ws.send_json({"type": "start_call", "language": "en"})

                call_id = ""
                for _ in range(200):
                    msg = ws.receive()
                    if msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        if ev.get("type") == "call_started":
                            call_id = ev["call_id"]
                        elif ev.get("type") == "delivery_stats" and ev.get("stats", {}).get("turn_count", 0) >= 1:
                            break
                assert call_id
                ws.send_json({"type": "playback_done", "played_samples": 4 * 1920})

                # Trigger Turn 2 delegation: filler finishes quickly, leaving session awaiting `delegation_task`.
                omni.asr_idx = 0
                omni.asr_sequence = ["Never mind, stop.", ""]
                ws.send_json({"type": "user_text", "text": "What are the Platinum lounge rules?"})

                # Wait for Turn 2 filler live_ttfa & delegate task entry while awaiting `delegation_task`.
                saw_turn2_start = False
                saw_turn2_filler_ttfa = False
                for _ in range(200):
                    msg = ws.receive()
                    if msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        t = ev.get("type")
                        if t == "assistant_turn_start":
                            saw_turn2_start = True
                        elif t == "live_ttfa" and saw_turn2_start:
                            saw_turn2_filler_ttfa = True
                            break
                assert saw_turn2_filler_ttfa
                assert delegate_stub.started_first.wait(timeout=5.0)
                time.sleep(0.05)

                # Interrupt while waiting on `delegation_task`.
                ws.send_bytes(_pcm16_16k_speech_block(blocks=1))

                delegation_events: list[dict[str, Any]] = []
                barge_ins = 0
                turn3_completed = False
                for _ in range(300):
                    msg = ws.receive()
                    if msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        t = ev.get("type")
                        if t == "barge_in":
                            barge_ins += 1
                            silence_block = b"\x00" * 20480
                            ws.send_bytes(silence_block)
                            ws.send_bytes(silence_block)
                        elif t == "delegation_event":
                            delegation_events.append(ev)
                        elif t == "assistant_turn_end" and barge_ins >= 1:
                            turn3_completed = True
                            break

                assert barge_ins >= 1
                assert turn3_completed is True
                assert delegate_stub.cancelled_calls == 1
                # Only Turn 3 ("Never mind, stop.") could emit a delegation_event; Turn 2 was cancelled.
                assert all("Platinum lounge" not in (d.get("query") or "") for d in delegation_events)

                ws.send_json({"type": "hang_up"})
                for _ in range(40):
                    msg = ws.receive()
                    if msg.get("text") is not None:
                        ev = json.loads(msg["text"])
                        if ev.get("type") == "call_state" and ev.get("state") == "idle":
                            break

            detail = client.get(f"/api/calls/{call_id}").json()
            assert detail["metadata"]["counts"]["error_count"] == 0
    finally:
        engine.shutdown()


def test_vllm_speculative_native_s2s_direct_tool_route_and_trailing_filler_prompt() -> None:
    """DIRECT uses native `<|mixed_start|>`, TOOL runs CRM tool schemas, DELEGATE puts filler last."""
    omni = FakeAsyncOmni()
    tok = FakeTokenizer()
    settings = EngineSettings(backend="vllm", model_id="LiquidAI/LFM3-Omni-2B")
    engine = VllmOmniEngine(settings, omni_client=omni, tokenizer=tok)
    engine.load()
    persona = next(p for p in build_personas() if p.id == "naledi")
    tools = build_tools()
    try:
        sess = engine.create_pipeline()
        sess.initialize_conversation(persona.system_prompt, tools=tools, language="en")

        # 1. DIRECT route even with `tools` provided uses native `<|mixed_start|>` without TTS fallback
        omni.router_decision = "DIRECT"
        sess.append_text_user_turn("Why did you repeat yourself?")
        omni.calls.clear()
        dir_plan = sess.run_speculative_selective_batch(
            persona=persona,
            tools=tools,
            dynamic_filler_enabled=True,
            cancel_event=threading.Event(),
            on_event=lambda _e: None,
        )
        assert dir_plan.decision == "direct"
        assert dir_plan.direct_outcome is not None
        assert dir_plan.direct_outcome.used_tts_fallback is False
        assert dir_plan.direct_outcome.audio_frames == 4
        chat_calls = [c for c in omni.calls if c["output_modalities"] == ["text", "audio"]]
        assert chat_calls
        assert chat_calls[-1]["rendered"].endswith("<|mixed_start|>")
        assert "<|tool_list_start|>" not in chat_calls[-1]["rendered"]

        # 2. TOOL route aborts speculative native stream and runs tool-enabled completion
        omni.router_decision = "TOOL"
        omni.tool_call_on_next_answer = (
            "check_pre_qualification",
            {"monthly_income_zar": 35000},
        )
        sess.append_text_user_turn("I earn R35,000 full-time with SA ID.")
        omni.calls.clear()
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
        tool_chat_calls = [c for c in omni.calls if "<|tool_list_start|>" in c["rendered"]]
        assert len(tool_chat_calls) == 1

        # 3. DELEGATE route places filler instruction in trailing user turn (never buried in system prompt)
        omni.router_decision = "DELEGATE: What are the Ubuntu Protea Platinum airport lounge fees?"
        sess.append_text_user_turn("What are the Protea Platinum airport lounge fees?")
        omni.calls.clear()
        del_plan = sess.run_speculative_selective_batch(
            persona=persona,
            tools=tools,
            dynamic_filler_enabled=True,
            cancel_event=threading.Event(),
            on_event=lambda _e: None,
        )
        assert del_plan.decision == "delegate"
        filler_calls = [
            c for c in omni.calls if "holding phrase" in c["rendered"].lower()
        ]
        assert len(filler_calls) == 1
        rendered_filler = filler_calls[0]["rendered"]
        sys_section = rendered_filler.split("<|im_end|>\n", 1)[0]
        trailing_section = rendered_filler.rsplit("<|im_start|>user\n", 1)[-1]
        assert "holding phrase" not in sys_section.lower()
        assert "holding phrase" in trailing_section.lower()
    finally:
        engine.shutdown()


def test_e2e_speech_to_audio_ms_metric_on_live_voice_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`e2e_speech_to_audio_ms` measures from last speech block to browser `playback_started` (None on greeting)."""
    monkeypatch.delenv("FLAMETREE_AGENT_TOKEN", raising=False)
    omni = FakeAsyncOmni()
    omni.router_decision = "DIRECT"
    omni.asr_sequence = ["What is the monthly fee?", "", ""]
    tok = FakeTokenizer()
    calls_dir = tmp_path / "calls"

    cfg = build_config(
        backend="vllm",
        language="en",
        delegation_mode="selective_tool",
        calls_dir=str(calls_dir),
    )
    engine = VllmOmniEngine(cfg.engine, omni_client=omni, tokenizer=tok)
    engine.load()
    try:
        app = create_app(cfg, engine=engine, load_on_startup=False)
        with TestClient(app) as client:
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
                        elif t == "live_ttfa" and "ttfa_perceived_ms" in ev and ev.get("turn_id") == t1_turn_id:
                            t1_perc_ev = ev
                            break

                assert call_id
                assert t1_perc_ev is not None
                assert t1_perc_ev.get("e2e_speech_to_audio_ms") is None
                ws.send_json({"type": "playback_done", "played_samples": 4800})

                # Stream speech block, pause 40 ms so endpoint_ms > 0, then silence to commit
                ws.send_bytes(_pcm16_16k_speech_block(blocks=1))
                time.sleep(0.04)
                ws.send_bytes(b"\x00" * 20480)
                ws.send_bytes(b"\x00" * 20480)

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
                        elif t == "live_ttfa" and "ttfa_perceived_ms" in ev and ev.get("turn_id") == t2_turn_id:
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

            ast_entries = [
                e for e in detail["transcript"]["entries"] if e.get("role") == "assistant"
            ]
            assert ast_entries[0]["metrics"].get("e2e_speech_to_audio_ms") is None
            assert ast_entries[1]["metrics"].get("e2e_speech_to_audio_ms") is not None
            txt = (calls_dir / call_id / "transcript.txt").read_text(encoding="utf-8")
            assert "E2E:" in txt
    finally:
        engine.shutdown()
