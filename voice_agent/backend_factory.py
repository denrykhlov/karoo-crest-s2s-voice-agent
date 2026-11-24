"""Lazy backend factory selecting MLX (macOS), vLLM-Omni, or PyTorch/Transformers (CUDA)."""

from __future__ import annotations

from typing import Any

from .config import EngineSettings


def build_engine(settings: EngineSettings) -> Any:
    """Instantiate the configured S2S backend without importing unused frameworks."""
    if settings.backend == "vllm":
        from .vllm_backend import VllmOmniEngine

        return VllmOmniEngine(settings)
    if settings.backend == "transformers":
        from .transformers_backend import TransformersOmniEngine

        return TransformersOmniEngine(settings)
    from .engine import Lfm3OmniEngine

    return Lfm3OmniEngine(settings)
