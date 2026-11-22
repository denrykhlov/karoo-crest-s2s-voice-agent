"""Local macOS (MLX) & CUDA (vLLM-Omni) speech-to-speech voice agent for LiquidAI/LFM3-Omni-2B."""

from .backend_factory import build_engine
from .call_recorder import (
    CallRecorder,
    generate_call_id,
    list_recorded_calls,
    load_recorded_call,
)
from .config import (
    EngineSettings,
    FlametreeSettings,
    PersonaConfig,
    ProductCardTier,
    ProductSummary,
    ToolSpec,
    VoiceAgentConfig,
)
from .languages import LanguageProfile, SUPPORTED_LANGUAGES
from .metrics import DeliveryStatsTracker, TurnDeliveryMetrics
from .protocol import BatchPlanOutcome, TurnOutcome, TurnRecord

__all__ = [
    "BatchPlanOutcome",
    "CallRecorder",
    "DeliveryStatsTracker",
    "EngineSettings",
    "FlametreeSettings",
    "LanguageProfile",
    "PersonaConfig",
    "ProductCardTier",
    "ProductSummary",
    "SUPPORTED_LANGUAGES",
    "ToolSpec",
    "TurnDeliveryMetrics",
    "TurnOutcome",
    "TurnRecord",
    "VoiceAgentConfig",
    "build_engine",
    "generate_call_id",
    "list_recorded_calls",
    "load_recorded_call",
]
