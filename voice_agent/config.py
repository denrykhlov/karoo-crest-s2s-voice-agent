"""Core configuration dataclasses and tool specifications for the pure-S2S LFM3-Omni voice agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class PersonaConfig:
    """Voice agent persona configuration."""

    id: str
    name: str
    role: str
    voice_gender: str
    voice_accent: str
    greeting_trigger: str
    system_prompt: str
    description: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize public-safe persona metadata for UI consumption."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "voice_gender": self.voice_gender,
            "voice_accent": self.voice_accent,
            "description": self.description,
        }


@dataclass(slots=True)
class ToolSpec:
    """Declarative specification for a voice-callable Python tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]

    def to_openai_schema(self) -> dict[str, Any]:
        """Return OpenAI-style function tool schema accepted by LFM3 chat template."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class FlametreeSettings:
    """Flametree Platform delegation endpoint configuration (secrets stay in env vars)."""

    enabled: bool = True
    base_url: str = "https://portal.flametree.ai"
    environment: str = "prod"
    tenant_id: str = "8470d927-5847-48fc-9a55-d72dd4117ac1"
    tenant_name: str = "Denis Demo"
    agent_id: str = ""
    api_token_env: str = "FLAMETREE_AGENT_TOKEN"
    timeout_sec: float = 12.0

    def has_token(self) -> bool:
        """Check whether the runtime bearer token env var is populated (never returns it)."""
        return bool(os.environ.get(self.api_token_env, "").strip())

    def to_public_dict(self) -> dict[str, Any]:
        portal_url = (
            f"{self.base_url.rstrip('/')}/agent-v2/{self.agent_id}?tenant_id={self.tenant_id}"
            if self.agent_id and self.tenant_id
            else ""
        )
        return {
            "enabled": self.enabled,
            "environment": self.environment,
            "tenant_name": self.tenant_name,
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "api_token_env": self.api_token_env,
            "live_token_present": self.has_token(),
            "portal_url": portal_url,
        }


@dataclass(slots=True)
class EngineSettings:
    """Inference, pure-S2S always-streaming, and 5-minute call management settings."""

    backend: str = "mlx"  # "mlx" | "vllm"
    model_id: str = "LiquidAI/LFM3-Omni-2B-MLX-8bit"
    vllm_base_url: str = "http://127.0.0.1:8091"
    vllm_api_key: str | None = None
    vllm_timeout_sec: float = 90.0
    audio_temperature: float = 0.4
    audio_top_k: int = 128
    audio_repetition_penalty: float = 1.2
    audio_repetition_window: int = 4096
    text_temperature: float = 0.2
    text_top_p: float = 0.9
    text_repetition_penalty: float = 1.08
    max_slots: int = 900

    # Immediate Frame-1 streaming output (Frame 1 = 1440 samples / 60 ms @ 24 kHz)
    first_chunk_frames: int = 1
    stream_chunk_frames: int = 2
    playback_prebuffer_ms: int = 40  # 960 samples < 1440 samples => zero chunk-1 stall

    # Pure-S2S always-streaming parameters (zero external VAD)
    live_chunk_tokens: int = 12  # 12 * 80 ms = 960 ms native LiveASR chunk
    half_stride_probe_tokens: int = 6  # 480 ms non-mutating speculative silence probe
    pre_speech_blocks: int = 2  # 1.92 s ring buffer covering LiveASR emission lag
    listener_idle_reset_sec: float = 15.0
    listener_hard_reset_sec: float = 30.0
    max_user_turn_sec: float = 25.0
    coop_listen_slot_interval: int = 32

    # Batched MLX parallel filler + delegation settings
    max_filler_slots: int = 110
    max_router_tokens: int = 56

    # 5-minute call context compaction settings
    compaction_trigger_tokens: int = 4096
    keep_recent_user_audio_turns: int = 4
    keep_recent_assistant_audio_turns: int = 3


@dataclass(slots=True)
class ProductCardTier:
    """Product tier metadata rendered in the UI reference sheet."""

    id: str
    name: str
    monthly_fee: str
    limit_range: str
    min_income: str
    interest_rate: str
    cashback: str
    highlights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "monthly_fee": self.monthly_fee,
            "limit_range": self.limit_range,
            "min_income": self.min_income,
            "interest_rate": self.interest_rate,
            "cashback": self.cashback,
            "highlights": list(self.highlights),
        }


@dataclass(slots=True)
class ProductSummary:
    """Structured product metadata passed to the browser UI via /api/config."""

    product_name: str
    tagline: str
    currency: str = "ZAR"
    interest_free_days: int = 55
    compliance_note: str = ""
    tiers: list[ProductCardTier] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_name": self.product_name,
            "tagline": self.tagline,
            "currency": self.currency,
            "interest_free_days": self.interest_free_days,
            "compliance_note": self.compliance_note,
            "tiers": [t.to_dict() for t in self.tiers],
        }


@dataclass(slots=True)
class VoiceAgentConfig:
    """Top-level configuration for the voice agent server and domain script."""

    app_title: str
    company_name: str
    subtitle: str
    badges: list[str]
    personas: list[PersonaConfig]
    default_persona_id: str
    tools: list[ToolSpec] = field(default_factory=list)
    product_summary: ProductSummary | None = None
    engine: EngineSettings = field(default_factory=EngineSettings)
    host: str = "127.0.0.1"
    port: int = 8000
    default_turn_patience: str = "balanced"  # "snappy" | "balanced" | "patient"
    target_call_duration_sec: int = 300
    default_language: str = "en"
    supported_languages: list[str] = field(
        default_factory=lambda: ["es-MX", "es", "en"]
    )
    default_delegation_mode: str = "selective_tool"  # "selective_tool" | "always_delegate"
    default_dynamic_filler: bool = True
    flametree: FlametreeSettings = field(default_factory=FlametreeSettings)
    calls_dir: Optional[str] = "calls"
    record_call_audio: bool = True

    def get_persona(self, persona_id: str | None = None) -> PersonaConfig:
        """Resolve a persona by ID, falling back to default_persona_id."""
        target = persona_id or self.default_persona_id
        for p in self.personas:
            if p.id == target:
                return p
        return self.personas[0]

    def to_public_config(self) -> dict[str, Any]:
        """Serialize configuration for GET /api/config without exposing secrets or raw prompts."""
        pub: dict[str, Any] = {
            "app_title": self.app_title,
            "company_name": self.company_name,
            "subtitle": self.subtitle,
            "badges": list(self.badges),
            "personas": [p.to_public_dict() for p in self.personas],
            "default_persona_id": self.default_persona_id,
            "tools": [
                {"name": t.name, "description": t.description} for t in self.tools
            ],
            "product_summary": (
                self.product_summary.to_dict() if self.product_summary else None
            ),
            "backend": self.engine.backend,
            "model_id": self.engine.model_id,
            "pipeline_mode": "pure_s2s_streaming",
            "batch_parallel_mode": (
                "vllm_continuous_batch"
                if self.engine.backend == "vllm"
                else "heterogeneous_b2"
            ),
            "default_turn_patience": self.default_turn_patience,
            "target_call_duration_sec": self.target_call_duration_sec,
            "compaction_trigger_tokens": self.engine.compaction_trigger_tokens,
            "first_chunk_frames": self.engine.first_chunk_frames,
            "stream_chunk_frames": self.engine.stream_chunk_frames,
            "default_language": self.default_language,
            "supported_languages": list(self.supported_languages),
            "default_delegation_mode": self.default_delegation_mode,
            "default_dynamic_filler": self.default_dynamic_filler,
            "flametree": self.flametree.to_public_dict(),
            "calls_dir": self.calls_dir,
            "record_call_audio": self.record_call_audio,
        }
        if self.engine.backend == "vllm":
            pub["vllm_base_url"] = self.engine.vllm_base_url
        return pub
