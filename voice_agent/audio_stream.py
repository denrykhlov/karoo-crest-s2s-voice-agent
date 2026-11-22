"""Causal streaming log-mel, audio encoder, and backbone cache snapshot utilities for LFM3-Omni."""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import numpy as np
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask
from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
)
from mlx_vlm.models.lfm3_audio.audio import (
    HOP,
    LOG_FLOOR,
    N_FFT,
    NUM_MEL_BINS,
    SAMPLE_RATE,
    num_frames as calc_num_frames,
)

_STREAM_MEL_FILTERS: Optional[np.ndarray] = None
_HANN_WINDOW_400: Optional[np.ndarray] = None


def _get_mel_filters_and_window() -> tuple[np.ndarray, np.ndarray]:
    global _STREAM_MEL_FILTERS, _HANN_WINDOW_400
    if _STREAM_MEL_FILTERS is None:
        from mlx_audio.dsp import mel_filters

        _STREAM_MEL_FILTERS = np.array(
            mel_filters(
                SAMPLE_RATE,
                N_FFT,
                NUM_MEL_BINS,
                norm="slaney",
                mel_scale="slaney",
                precise=True,
            ),
            dtype=np.float64,
        )
    if _HANN_WINDOW_400 is None:
        _HANN_WINDOW_400 = 0.5 - 0.5 * np.cos(
            2.0 * np.pi * np.arange(N_FFT, dtype=np.float64) / N_FFT
        )
    return _STREAM_MEL_FILTERS, _HANN_WINDOW_400


class StreamingLogMel:
    """Incremental 16 kHz mono -> 128-bin Slaney log-mel feature extractor.

    Bit-exact port of Liquid AI's `Lfm3OmniFeatureExtractor.stream`, emitting
    10 ms frames `(128, count)` as samples arrive while carrying the <400 sample overlap tail.
    """

    def __init__(self) -> None:
        self._mel_filters, self._window = _get_mel_filters_and_window()
        self.reset()

    def reset(self) -> None:
        self._samples = np.zeros((0,), dtype=np.float64)
        self.num_samples: int = 0
        self.num_frames: int = 0
        self.finished: bool = False

    def push(self, chunk_16k: np.ndarray, is_final: bool = False) -> np.ndarray:
        """Process a 1D float32/float64 16 kHz chunk and return `[128, count]` float32 mel frames."""
        if self.finished:
            raise ValueError("StreamingLogMel is finalized; call reset() first.")

        chunk = np.asarray(chunk_16k, dtype=np.float64)
        if chunk.ndim != 1:
            raise ValueError(f"Expected 1D mono waveform, got {chunk.shape}")

        samples = (
            np.concatenate([self._samples, chunk]) if self._samples.size else chunk
        )
        total_samples = self.num_samples + int(chunk.size)

        if is_final:
            if total_samples == 0:
                count = 0
            else:
                target_frames = calc_num_frames(total_samples)
                count = max(0, target_frames - self.num_frames)
                if count > 0:
                    needed = (count - 1) * HOP + N_FFT
                    if samples.size < needed:
                        samples = np.pad(samples, (0, needed - samples.size))
        else:
            if samples.size < N_FFT:
                count = 0
            else:
                count = 1 + (samples.size - N_FFT) // HOP

        if count > 0:
            used_len = (count - 1) * HOP + N_FFT
            frames = np.lib.stride_tricks.sliding_window_view(
                samples[:used_len], N_FFT
            )[::HOP]
            power = np.abs(np.fft.rfft(frames * self._window, axis=-1)) ** 2
            with np.errstate(divide="ignore"):
                mel = np.maximum(np.log10(self._mel_filters @ power.T), LOG_FLOOR)
            features = ((mel + 4.0) / 4.0).astype(np.float32)
        else:
            features = np.zeros((NUM_MEL_BINS, 0), dtype=np.float32)

        if is_final:
            self._samples = np.zeros((0,), dtype=np.float64)
        else:
            self._samples = samples[count * HOP :].copy()

        self.num_samples = total_samples
        self.num_frames += count
        self.finished = is_final
        return features


class StreamingAudioEncoder:
    """Causal O(1)-memory streaming audio encoder + projector for LFM3-Omni-2B.

    Exact MLX port of Liquid AI's `Lfm3OmniEncoder.stream`: carries the 1-token
    (8 mel-frame) left overlap for the 3 stride-2 convolutions and bounded
    `ArraysCache` + `RotatingKVCache(max_size=32)` across the 17-layer LFM2 trunk.
    Output embeddings have shape `[count, hidden_size]` (one embedding per 80 ms).
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.tower = model.audio_tower
        self.projector = model.audio_projector
        self.window_size: int = int(self.tower.config.sliding_window)
        self.hidden_size: int = int(model.config.text_config.hidden_size)
        self.reset()

    def reset(self) -> None:
        self._features_buf: Optional[np.ndarray] = None  # [128, T]
        self.num_frames: int = 0
        self.offset: int = 0
        self.num_tokens: int = 0
        self.finished: bool = False
        self.cache = [
            (
                RotatingKVCache(max_size=self.window_size)
                if layer.is_attention_layer
                else ArraysCache(size=1)
            )
            for layer in self.tower.lfm.layers
        ]

    def push(self, mel_frames: np.ndarray, is_final: bool = False) -> mx.array:
        """Consume `[128, F]` mel frames and emit newly completed `[count, 2048]` embeddings."""
        if self.finished:
            raise ValueError("StreamingAudioEncoder is finalized; call reset() first.")

        mel = np.asarray(mel_frames, dtype=np.float32)
        if mel.ndim != 2 or mel.shape[0] != NUM_MEL_BINS:
            raise ValueError(f"Expected mel shape [128, F], got {mel.shape}")

        if self._features_buf is not None and self._features_buf.shape[1] > 0:
            features = np.concatenate([self._features_buf, mel], axis=1)
        else:
            features = mel

        total_frames = self.num_frames + mel.shape[1]
        total_tokens = (total_frames + (7 if is_final else 0)) // 8
        count = total_tokens - self.num_tokens

        if count > 0:
            start = self.num_tokens - self.offset // 8
            feats_mx = mx.array(features[None])  # [1, 128, T]
            lengths = mx.array([features.shape[1]], dtype=mx.int32)
            down = self.tower.downsample(feats_mx, lengths)[:, start : start + count]

            h = down
            for layer, c in zip(self.tower.lfm.layers, self.cache):
                mask = (
                    create_attention_mask(h, c, window_size=self.window_size)
                    if layer.is_attention_layer
                    else create_ssm_mask(h, c)
                )
                h = layer(h, mask, c)
            h = self.tower.lfm.embedding_norm(h)
            projected = self.projector(h[0])  # [count, hidden_size]

            eval_cache(self.cache, extra=[projected])
        else:
            projected = mx.zeros((0, self.hidden_size), dtype=mx.bfloat16)

        new_offset = max(0, (total_tokens - 1) * 8)
        if is_final:
            self._features_buf = None
        else:
            keep_from = new_offset - self.offset
            self._features_buf = features[:, keep_from:].copy()

        self.num_frames = total_frames
        self.offset = new_offset
        self.num_tokens = total_tokens
        self.finished = is_final
        return projected


def eval_cache(cache: list[Any], extra: list[mx.array] | None = None) -> None:
    """Evaluate all array state held inside an LFM2/LFM3 cache list."""
    targets: list[mx.array] = list(extra) if extra else []
    for c in cache:
        if isinstance(c, (KVCache, RotatingKVCache, BatchKVCache, BatchRotatingKVCache)):
            if c.keys is not None:
                targets.append(c.keys)
            if c.values is not None:
                targets.append(c.values)
        elif isinstance(c, ArraysCache):
            for arr in c.cache:
                if arr is not None:
                    targets.append(arr)
    if targets:
        mx.eval(*targets)


def cache_token_count(cache: list[Any] | None) -> int:
    """Return the current sequence offset stored in an LFM3 backbone cache."""
    if not cache:
        return 0
    for c in cache:
        if isinstance(c, KVCache):
            return int(c.offset)
        if isinstance(c, BatchKVCache) and len(c.offset) > 0:
            return int(c.offset[0])
    for c in cache:
        if isinstance(c, RotatingKVCache):
            return int(c.offset)
        if isinstance(c, BatchRotatingKVCache) and len(c.offset) > 0:
            return int(c.offset[0])
    return 0


def clone_lfm3_cache(cache: list[Any]) -> list[Any]:
    """Deep-clone a `B=1` LFM3 backbone cache so parallel branches can mutate safely."""
    cloned: list[Any] = []
    for c in cache:
        if isinstance(c, RotatingKVCache):
            nc = RotatingKVCache(max_size=c.max_size, keep=c.keep)
            nc.keys = mx.array(c.keys) if c.keys is not None else None
            nc.values = mx.array(c.values) if c.values is not None else None
            nc.meta_state = c.meta_state
            cloned.append(nc)
        elif isinstance(c, KVCache):
            nc = KVCache()
            nc.keys = mx.array(c.keys) if c.keys is not None else None
            nc.values = mx.array(c.values) if c.values is not None else None
            nc.offset = int(c.offset)
            cloned.append(nc)
        elif isinstance(c, ArraysCache):
            nc = ArraysCache(size=len(c.cache))
            nc.cache = [mx.array(x) if x is not None else None for x in c.cache]
            if c.left_padding is not None:
                nc.left_padding = mx.array(c.left_padding)
            if c.lengths is not None:
                nc.lengths = mx.array(c.lengths)
            cloned.append(nc)
        else:
            raise TypeError(f"Unsupported cache layer type for clone: {type(c)}")
    eval_cache(cloned)
    return cloned


def merge_lfm3_caches(caches: list[list[Any]]) -> list[Any]:
    """Merge `B` single-sequence LFM3 caches into one batched MLX cache."""
    if not caches:
        raise ValueError("Need at least one cache to merge.")
    merged: list[Any] = []
    for layer_items in zip(*caches):
        first = layer_items[0]
        if isinstance(first, RotatingKVCache):
            merged.append(BatchRotatingKVCache.merge(list(layer_items)))
        elif isinstance(first, KVCache):
            merged.append(BatchKVCache.merge(list(layer_items)))
        elif isinstance(first, ArraysCache):
            merged.append(ArraysCache.merge(list(layer_items)))
        else:
            raise TypeError(f"Unsupported cache layer type for merge: {type(first)}")
    eval_cache(merged)
    return merged


def extract_lfm3_cache(batch_cache: list[Any], idx: int) -> list[Any]:
    """Extract row `idx` from a batched MLX cache back into a standalone `B=1` cache."""
    single: list[Any] = []
    for c in batch_cache:
        if hasattr(c, "extract"):
            single.append(c.extract(idx))
        else:
            raise TypeError(f"Cache layer {type(c)} does not support extract().")
    eval_cache(single)
    return single


def snapshot_lfm3_cache(cache: list[Any]) -> list[tuple[Any, ...]]:
    """Capture an exact rollback snapshot of an LFM3 backbone cache (~3 MB)."""
    eval_cache(cache)
    snap: list[tuple[Any, ...]] = []
    for c in cache:
        if isinstance(c, RotatingKVCache):
            # Copy keys/values because RotatingKVCache._update_in_place mutates in-place.
            k_copy = mx.array(c.keys) * 1 if c.keys is not None else None
            v_copy = mx.array(c.values) * 1 if c.values is not None else None
            if k_copy is not None or v_copy is not None:
                mx.eval(*(x for x in (k_copy, v_copy) if x is not None))
            snap.append(("rot", k_copy, v_copy, c.meta_state))
        elif isinstance(c, KVCache):
            snap.append(("kv", int(c.offset)))
        elif isinstance(c, ArraysCache):
            arrs = [x * 1 if x is not None else None for x in c.cache]
            if any(a is not None for a in arrs):
                mx.eval(*(a for a in arrs if a is not None))
            lp = c.left_padding * 1 if c.left_padding is not None else None
            ln = c.lengths * 1 if c.lengths is not None else None
            snap.append(("arr", arrs, lp, ln))
        else:
            snap.append(("other", None))
    return snap


def restore_lfm3_cache(cache: list[Any], snapshot: list[tuple[Any, ...]]) -> None:
    """Restore an LFM3 backbone cache to a previously captured snapshot."""
    if len(cache) != len(snapshot):
        raise ValueError(
            f"Cache length {len(cache)} does not match snapshot length {len(snapshot)}"
        )
    for c, item in zip(cache, snapshot):
        kind = item[0]
        if kind == "kv":
            _, saved_offset = item
            delta = int(c.offset) - int(saved_offset)
            if delta > 0:
                c.trim(delta)
            elif delta < 0:
                raise ValueError(
                    f"Cannot restore KVCache forward from {c.offset} to {saved_offset}"
                )
        elif kind == "rot":
            _, k_copy, v_copy, meta = item
            c.keys = k_copy * 1 if k_copy is not None else None
            c.values = v_copy * 1 if v_copy is not None else None
            c.meta_state = meta
        elif kind == "arr":
            _, arrs, lp, ln = item
            c.cache = [a * 1 if a is not None else None for a in arrs]
            c.left_padding = lp * 1 if lp is not None else None
            c.lengths = ln * 1 if ln is not None else None
    eval_cache(cache)
