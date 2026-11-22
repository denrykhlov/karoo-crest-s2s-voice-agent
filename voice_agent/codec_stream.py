"""Bit-exact streaming LiquidCodec 24 kHz neural audio decoder for MLX."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import mlx.core as mx
import numpy as np
from mlx_vlm.models.cache import ArraysCache, KVCache, RotatingKVCache


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert mono float32 samples in [-1, 1] to little-endian signed 16-bit PCM bytes."""
    if audio.size == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2", copy=False)
    return pcm16.tobytes()


class StreamingLiquidCodecDecoder:
    """Stateful chunked decoder matching batch `LiquidCodec.decode` sample-for-sample.

    Carries both the 8-layer causal LFM3 decoder cache (`ArraysCache` for short convs,
    `RotatingKVCache(max_size=30)` for sliding-window attention) and the Hann-windowed
    ISTFT overlap-add tail across consecutive code-frame chunks.
    """

    def __init__(self, codec) -> None:
        self.codec = codec
        cfg = codec.config
        lfm = codec.decoder.lfm
        lfm_cfg = lfm.config
        self.cache = [
            (
                ArraysCache(size=1)
                if layer.layer_type == "conv"
                else (
                    RotatingKVCache(max_size=lfm_cfg.sliding_window)
                    if layer.layer_type == "sliding_attention"
                    else KVCache()
                )
            )
            for layer in lfm.layers
        ]
        self.trim: int = (cfg.stft_win_length - cfg.stft_hop_length) // 2
        self.tail: int = 2 * self.trim
        self.prev_audio: Optional[mx.array] = None
        self.prev_env: Optional[mx.array] = None
        self.started: bool = False
        self.finished: bool = False

    def decode_chunk(
        self,
        codes_chunk: np.ndarray | mx.array,
        finalize: bool = False,
    ) -> np.ndarray:
        """Decode `[12, T]` (or `[1, 12, T]`) FSQ speech codes into 24 kHz float32 audio.

        Callers must strip `audio_eos_code` (2016) frames before calling this method.
        When `finalize=True`, flushes the held-back overlap-add tail (supports `T == 0`).
        """
        if self.finished:
            raise ValueError("StreamingLiquidCodecDecoder is already finalized.")

        codes = mx.array(codes_chunk, dtype=mx.int32)
        if codes.ndim == 2:
            codes = codes[None]
        if codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[1] != self.codec.config.n_q:
            raise ValueError(
                f"Expected codes shape [12, T] or [1, 12, T], got {codes.shape}"
            )

        num_frames = codes.shape[2]
        if num_frames > 0:
            codebook_size = self.codec.config.codebook_size
            c_min = int(codes.min().item())
            c_max = int(codes.max().item())
            if c_min < 0 or c_max >= codebook_size:
                raise ValueError(
                    f"Codes must be in [0, {codebook_size}); got range [{c_min}, {c_max}]. "
                    "Strip EOS frames (2016) and padding first."
                )

        if num_frames == 0 and not (finalize and self.started):
            if finalize:
                self.finished = True
            return np.zeros((0,), dtype=np.float32)

        if num_frames > 0:
            latents = self.codec.from_quantizer(self.codec.quantizer(codes))
            repeated = mx.repeat(latents, self.codec.decoder.stack, axis=1)
            h = self.codec.decoder.lfm(input_embeddings=repeated, cache=self.cache)
            log_mag, angle = mx.split(
                self.codec.decoder.lin(h).astype(mx.float32), 2, axis=-1
            )
            mag = mx.exp(log_mag)
            spec = mag * mx.cos(angle) + 1j * (mag * mx.sin(angle))

            istft = self.codec.decoder.istft
            window = istft.window.astype(mx.float32)
            frames = mx.fft.irfft(spec, n=istft.n_fft, axis=-1) * window
            audio = istft._overlap_add(frames)
            env = istft._overlap_add(
                mx.broadcast_to(window * window, (1, spec.shape[1], istft.n_fft))
            )
        else:
            audio = mx.zeros((1, self.tail), dtype=mx.float32)
            env = mx.zeros((1, self.tail), dtype=mx.float32)

        if self.prev_audio is not None and self.prev_env is not None:
            audio = mx.concatenate(
                [audio[:, : self.tail] + self.prev_audio, audio[:, self.tail :]],
                axis=1,
            )
            env = mx.concatenate(
                [env[:, : self.tail] + self.prev_env, env[:, self.tail :]],
                axis=1,
            )

        if not self.started:
            audio = audio[:, self.trim :]
            env = env[:, self.trim :]
            self.started = True

        if finalize:
            audio = audio[:, : -self.trim]
            env = env[:, : -self.trim]
            self.prev_audio = None
            self.prev_env = None
            self.finished = True
        else:
            self.prev_audio = audio[:, -self.tail :]
            self.prev_env = env[:, -self.tail :]
            audio = audio[:, : -self.tail]
            env = env[:, : -self.tail]

        out = audio / mx.maximum(env, 1e-8)

        # Evaluate state and output immediately so MLX lazy graphs do not accumulate.
        eval_targets = [out]
        if self.prev_audio is not None:
            eval_targets.extend([self.prev_audio, self.prev_env])
        for c in self.cache:
            if hasattr(c, "keys") and c.keys is not None:
                eval_targets.append(c.keys)
            if hasattr(c, "values") and c.values is not None:
                eval_targets.append(c.values)
            if hasattr(c, "cache") and c.cache is not None:
                eval_targets.extend(x for x in c.cache if x is not None)
        mx.eval(*eval_targets)

        return np.asarray(out[0], dtype=np.float32).copy()


class StreamingAudioFlusher:
    """Shared Frame-1 incremental LiquidCodec flusher for S2S and B=2 generators."""

    def __init__(
        self,
        codec: Any,
        first_chunk_frames: int,
        stream_chunk_frames: int,
        on_event: Callable[[dict[str, Any]], None],
        segment_name: str = "answer",
    ) -> None:
        self.decoder = StreamingLiquidCodecDecoder(codec)
        self.first_chunk_frames = max(1, int(first_chunk_frames))
        self.stream_chunk_frames = max(1, int(stream_chunk_frames))
        self.on_event = on_event
        self.segment_name = segment_name

        self.pending: list[np.ndarray] = []
        self.first_chunk_emitted: bool = False
        self.chunks_sent: int = 0
        self.first_chunk_samples: int = 0
        self.t_first_audio: Optional[float] = None
        self.codec_ms_list: list[float] = []

    def push_frame(self, codes_12: np.ndarray) -> None:
        """Queue one non-EOS speech frame and flush immediately when chunk threshold hits."""
        self.pending.append(codes_12)
        target = (
            self.stream_chunk_frames
            if self.first_chunk_emitted
            else self.first_chunk_frames
        )
        if len(self.pending) >= target:
            self.flush(finalize=False)

    def flush(self, finalize: bool = False) -> None:
        """Decode pending frames immediately and dispatch `audio_chunk` over `on_event`."""
        if self.decoder.finished:
            return
        if not self.pending and not (finalize and self.decoder.started):
            if finalize:
                self.decoder.finished = True
            return
        if self.pending:
            codes_chunk = np.stack(self.pending, axis=-1)
            n_frames = len(self.pending)
            self.pending.clear()
        else:
            codes_chunk = np.zeros((12, 0), dtype=np.int32)
            n_frames = 0

        t0 = time.perf_counter()
        wav = self.decoder.decode_chunk(codes_chunk, finalize=finalize)
        codec_ms = (time.perf_counter() - t0) * 1000.0
        self.codec_ms_list.append(codec_ms)

        if wav.size > 0:
            self.chunks_sent += 1
            if not self.first_chunk_emitted:
                self.first_chunk_emitted = True
                self.first_chunk_samples = int(wav.size)
                self.t_first_audio = time.perf_counter()
            self.on_event(
                {
                    "type": "audio_chunk",
                    "pcm16": float32_to_pcm16(wav),
                    "frames": n_frames,
                    "samples": int(wav.size),
                    "segment": self.segment_name,
                    "chunk_index": self.chunks_sent,
                    "codec_ms": round(codec_ms, 2),
                }
            )
