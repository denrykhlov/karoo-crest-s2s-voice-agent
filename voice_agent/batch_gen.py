"""Heterogeneous B=2 MLX batched generator for parallel fillers, routing & Frame-1 streaming."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import mlx.core as mx
import numpy as np
from mlx_vlm.models.lfm3_audio.depth import (
    embed_frames,
    normalize_eos,
    sample,
    sample_frame,
)

from .audio_stream import (
    clone_lfm3_cache,
    eval_cache,
    extract_lfm3_cache,
    merge_lfm3_caches,
)
from .codec_stream import StreamingAudioFlusher
from .config import EngineSettings, PersonaConfig, ToolSpec
from .languages import LanguageProfile, get_language_profile
from .protocol import (
    BatchPlanOutcome,
    TurnOutcome,
    TurnRecord,
    clean_display_text,
    parse_tool_calls,
)

TEXT, AUDIO, INTERLEAVED = "text", "audio", "interleaved"


def _parse_router_query(raw_router_text: str, fallback_text: str) -> tuple[str, str]:
    """Parse `DIRECT` or `DELEGATE: <question>` into `(kind, standalone_question)`."""
    cleaned = clean_display_text(raw_router_text).strip()
    upper = cleaned.upper()
    if "DELEGATE" in upper:
        idx = upper.find("DELEGATE")
        tail = cleaned[idx + len("DELEGATE") :].lstrip(" :-\n\t")
        return "delegate", (tail.strip() or fallback_text.strip() or "(banking question)")
    if "DIRECT" in upper:
        return "direct", ""
    return "direct", ""


def _build_partial_turn(
    raw_token_ids: list[int],
    raw_audio_codes: list[np.ndarray],
    text_token_ids: list[int],
    tokenizer: Any,
    cfg: Any,
    im_end_id: int,
) -> Optional[TurnRecord]:
    if not raw_token_ids and not raw_audio_codes:
        return None
    partial_text = clean_display_text(
        tokenizer.decode(text_token_ids, skip_special_tokens=True)
    )
    p_ids = list(raw_token_ids)
    p_codes = [np.asarray(c, dtype=np.int32) for c in raw_audio_codes]
    if p_codes and not np.any(p_codes[-1] == cfg.audio_eos_code):
        p_codes.append(np.full((12,), cfg.audio_eos_code, dtype=np.int32))
        p_ids.append(int(cfg.audio_out_token_id))
    if not p_ids or p_ids[-1] != im_end_id:
        p_ids.append(im_end_id)
    return TurnRecord(
        role="assistant",
        text=partial_text,
        raw_token_ids=p_ids,
        raw_audio_codes=(
            np.stack(p_codes, axis=0).astype(np.int32) if p_codes else None
        ),
    )


class BatchLfm3Generator:
    """Runs B=2 heterogeneous MLX generation (speech filler + text routing/query) on one worker."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        codec: Any,
        settings: EngineSettings,
        tts_fallback_fn: Optional[Callable[..., tuple[Optional[np.ndarray], list[int]]]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.codec = codec
        self.settings = settings
        self.cfg = model.config
        self.backbone = model.language_model.model
        self._tts_fallback_fn = tts_fallback_fn

        self._im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
        self._newline_ids = list(tokenizer.encode("\n", add_special_tokens=False))
        eos_ids = self.cfg.eos_token_id
        self._eos_id_set = set(
            eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        )
        self._suppress = list(model.placeholder_token_ids())

    def _encode_text(self, text: str) -> list[int]:
        if not text:
            return []
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _sample_text_token(
        self,
        h_row: mx.array,
        history: deque[int],
        greedy: bool = False,
    ) -> int:
        if greedy:
            logits = self.backbone.embed_tokens.as_linear(h_row).astype(mx.float32)
            logits[..., self._suppress] = -mx.inf
            return int(mx.argmax(logits, axis=-1).item())
        hist_mx = mx.array([list(history)]) if history else None
        return int(
            sample(
                self.backbone.embed_tokens.as_linear(h_row),
                self.settings.text_temperature,
                None,
                self.settings.text_repetition_penalty,
                hist_mx,
                self._suppress,
                self.settings.text_top_p,
                None,
            ).item()
        )

    def _prefill_single(self, cache: list[Any], token_ids: list[int]) -> mx.array:
        embeds = self.backbone.embed_tokens(mx.array([token_ids], dtype=mx.int32))
        h = self.backbone(None, cache, embeds)[:, -1]
        eval_cache(cache, extra=[h])
        return h

    def _build_clean_row1_cache(
        self,
        system_prompt: str,
        latest_user_text: str,
        user_audio_embeds: Optional[mx.array] = None,
        prev_assistant_text: str = "",
    ) -> tuple[list[Any], mx.array]:
        """Build a clean, unpolluted Row 1 cache using native audio embeds when available."""
        c1 = self.model.make_cache()
        prefix_str = f"<|startoftext|><|im_start|>system\n{system_prompt}<|im_end|>\n"
        if prev_assistant_text and prev_assistant_text.strip():
            prefix_str += (
                f"<|im_start|>assistant\n{prev_assistant_text.strip()[:180]}<|im_end|>\n"
            )
        if user_audio_embeds is not None and user_audio_embeds.shape[0] > 0:
            pre_ids = self._encode_text(prefix_str + "<|im_start|>user\n<|vision_start|>")
            post_ids = self._encode_text(
                "<|vision_end|><|im_end|>\n<|im_start|>assistant\n<think></think>"
            )
            embeds = mx.concatenate(
                [
                    self.backbone.embed_tokens(mx.array([pre_ids], dtype=mx.int32)),
                    user_audio_embeds[None],
                    self.backbone.embed_tokens(mx.array([post_ids], dtype=mx.int32)),
                ],
                axis=1,
            )
        else:
            user_msg = (latest_user_text or "").strip() or "(spoken message)"
            full_ids = self._encode_text(
                prefix_str
                + f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n<think></think>"
            )
            embeds = self.backbone.embed_tokens(mx.array([full_ids], dtype=mx.int32))
        h1 = self.backbone(None, c1, embeds)[:, -1]
        eval_cache(c1, extra=[h1])
        return c1, h1

    def run_filler_and_query_batch(
        self,
        conv_cache: list[Any],
        language_code: str,
        latest_user_text: str,
        cancel_event: threading.Event,
        on_event: Callable[[dict[str, Any]], None],
        poll_mic_fn: Optional[Callable[[], list[dict[str, Any]]]] = None,
        user_audio_embeds: Optional[mx.array] = None,
        prev_assistant_text: str = "",
    ) -> BatchPlanOutcome:
        """Clone `conv_cache` for Row 0 (voice filler) || clean Row 1 (delegation query)."""
        if cancel_event.is_set():
            return BatchPlanOutcome(decision="cancelled")

        t_start = time.perf_counter()
        profile: LanguageProfile = get_language_profile(language_code)

        c0 = clone_lfm3_cache(conv_cache)
        r0_ids = self._encode_text(
            f"<|im_start|>user\n{profile.filler_instruction}<|im_end|>\n"
            "<|im_start|>assistant\n<think></think>"
        ) + [int(self.cfg.mixed_start_token_id)]
        h0 = self._prefill_single(c0, r0_ids)

        query_sys = (
            "You are a banking query forwarder. Do NOT answer the customer's question. "
            f"Translate and rewrite the customer's request into one concise standalone question in {profile.router_language_name} "
            "prefixed with `DELEGATE: `. Example: `DELEGATE: What is the monthly fee on Ubuntu Gold?`"
        )
        c1, h1 = self._build_clean_row1_cache(
            system_prompt=query_sys,
            latest_user_text=latest_user_text,
            user_audio_embeds=user_audio_embeds,
            prev_assistant_text=prev_assistant_text,
        )

        bc = merge_lfm3_caches([c0, c1])
        h = mx.concatenate([h0, h1], axis=0)
        mx.eval(h)
        fork_merge_ms = (time.perf_counter() - t_start) * 1000.0

        return self._run_filler_query_loop(
            bc=bc,
            h=h,
            initial_r1_tokens=[],
            r1_initially_done=False,
            strip_delegate_prefix=True,
            latest_user_text=latest_user_text,
            t_start=t_start,
            fork_merge_ms=fork_merge_ms,
            route_decision_ms=fork_merge_ms,
            cancel_event=cancel_event,
            on_event=on_event,
            poll_mic_fn=poll_mic_fn,
        )

    def _run_filler_query_loop(
        self,
        bc: list[Any],
        h: mx.array,
        initial_r1_tokens: list[int],
        r1_initially_done: bool,
        strip_delegate_prefix: bool,
        latest_user_text: str,
        t_start: float,
        fork_merge_ms: float,
        route_decision_ms: float,
        cancel_event: threading.Event,
        on_event: Callable[[dict[str, Any]], None],
        poll_mic_fn: Optional[Callable[[], list[dict[str, Any]]]] = None,
    ) -> BatchPlanOutcome:
        """Shared B=2 -> B=1 execution loop for Row 0 filler speech + Row 1 standalone query."""
        r0_raw_ids: list[int] = [int(self.cfg.mixed_start_token_id)]
        r0_text_ids: list[int] = []
        r0_codes: list[np.ndarray] = []
        r0_text_hist: deque[int] = deque(maxlen=32)
        r0_audio_hist: deque[mx.array] = deque(maxlen=self.settings.audio_repetition_window)
        slots_left = int(self.cfg.interleaved_n_text)
        speaking = False
        text_done = False
        r0_done = False

        r1_tokens: list[int] = list(initial_r1_tokens)
        r1_done = bool(r1_initially_done)
        query_dispatched = False
        dispatched_query = ""
        emitted_filler_text = ""
        t_first_tok: Optional[float] = None

        flusher = StreamingAudioFlusher(
            self.codec,
            first_chunk_frames=self.settings.first_chunk_frames,
            stream_chunk_frames=self.settings.stream_chunk_frames,
            on_event=on_event,
            segment_name="filler",
        )

        def maybe_dispatch_query() -> None:
            nonlocal query_dispatched, dispatched_query
            if query_dispatched:
                return
            query_dispatched = True
            raw_q = self.tokenizer.decode(r1_tokens, skip_special_tokens=True)
            if strip_delegate_prefix:
                kind, parsed_q = _parse_router_query(raw_q, latest_user_text)
                if kind == "delegate" and parsed_q:
                    q = parsed_q
                else:
                    q = (
                        clean_display_text(raw_q).strip()
                        or latest_user_text.strip()
                        or "(banking question)"
                    )
            else:
                q = (
                    clean_display_text(raw_q).strip()
                    or latest_user_text.strip()
                    or "(banking question)"
                )
            dispatched_query = q
            on_event(
                {
                    "type": "delegation_query_ready",
                    "query": q,
                    "elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
                }
            )

        def emit_filler_delta(force: bool = False) -> None:
            nonlocal emitted_filler_text
            if not r0_text_ids:
                return
            decoded = clean_display_text(
                self.tokenizer.decode(r0_text_ids, skip_special_tokens=True)
            )
            if not force and decoded.endswith("\ufffd"):
                return
            if decoded.startswith(emitted_filler_text):
                delta = decoded[len(emitted_filler_text) :]
                if delta:
                    emitted_filler_text = decoded
                    on_event(
                        {
                            "type": "text_delta",
                            "delta": delta,
                            "text": decoded,
                            "segment": "filler",
                        }
                    )

        if r1_done:
            # Query already ready at decision time; fire HTTP delegation immediately.
            maybe_dispatch_query()
            bc = extract_lfm3_cache(bc, 0)
            h = h[0:1]

        max_steps = max(self.settings.max_filler_slots, self.settings.max_router_tokens)
        total_non_eos_frames = 0
        steps_taken = 0

        for step in range(max_steps):
            if cancel_event.is_set():
                filler_turn = _build_partial_turn(
                    r0_raw_ids, r0_codes, r0_text_ids, self.tokenizer, self.cfg, self._im_end_id
                )
                return BatchPlanOutcome(
                    decision="cancelled",
                    filler_turn=filler_turn,
                    filler_audio_frames=total_non_eos_frames,
                )

            if (
                poll_mic_fn is not None
                and step > 0
                and step % self.settings.coop_listen_slot_interval == 0
            ):
                if any(ev.get("type") == "barge_in_detected" for ev in poll_mic_fn()):
                    cancel_event.set()
                    filler_turn = _build_partial_turn(
                        r0_raw_ids, r0_codes, r0_text_ids, self.tokenizer, self.cfg, self._im_end_id
                    )
                    return BatchPlanOutcome(
                        decision="cancelled",
                        filler_turn=filler_turn,
                        filler_audio_frames=total_non_eos_frames,
                    )

            steps_taken += 1

            if not r0_done and not r1_done:
                # Heterogeneous B=2 step: Row 0 speech/text + Row 1 text query.
                if speaking:
                    audio_hist = (
                        mx.stack(list(r0_audio_hist))[None] if r0_audio_hist else None
                    )
                    codes = sample_frame(
                        self.model.depth_decoder,
                        h[0:1],
                        self.cfg.maskgit_schedule,
                        self.settings.audio_temperature,
                        self.settings.audio_top_k,
                        self.settings.audio_repetition_penalty,
                        audio_hist,
                        None,
                    )[0]
                    codes = normalize_eos(codes, self.cfg.audio_eos_code)
                    codes_np = np.asarray(codes, dtype=np.int32)
                    r0_raw_ids.append(int(self.cfg.audio_out_token_id))
                    r0_codes.append(codes_np)
                    if int(codes_np[0]) == self.cfg.audio_eos_code:
                        r0_done = True
                        flusher.flush(finalize=True)
                    else:
                        r0_audio_hist.append(codes)
                        total_non_eos_frames += 1
                        flusher.push_frame(codes_np)
                    e0 = embed_frames(self.model.codebook_embedding, codes).reshape(1, 1, -1)
                else:
                    t0_id = self._sample_text_token(h[0:1], r0_text_hist, greedy=False)
                    r0_text_hist.append(t0_id)
                    r0_raw_ids.append(t0_id)
                    r0_text_ids.append(t0_id)
                    if t_first_tok is None:
                        t_first_tok = time.perf_counter()
                    if t0_id == self.cfg.text_end_token_id:
                        text_done = speaking = True
                    elif t0_id in self._eos_id_set:
                        r0_done = True
                    emit_filler_delta(force=False)
                    e0 = self.backbone.embed_tokens(mx.array([[t0_id]], dtype=mx.int32))

                if not text_done:
                    slots_left -= 1
                    if slots_left == 0:
                        speaking = not speaking
                        slots_left = (
                            int(self.cfg.interleaved_n_audio)
                            if speaking
                            else int(self.cfg.interleaved_n_text)
                        )

                # Row 1 text query token
                t1_id = self._sample_text_token(h[1:2], deque(), greedy=True)
                if (
                    t1_id == self._im_end_id
                    or t1_id in self._eos_id_set
                    or t1_id == self.cfg.text_end_token_id
                    or len(r1_tokens) >= self.settings.max_router_tokens
                ):
                    r1_done = True
                    maybe_dispatch_query()
                    e1 = self.backbone.embed_tokens(mx.array([[self._im_end_id]], dtype=mx.int32))
                else:
                    r1_tokens.append(t1_id)
                    e1 = self.backbone.embed_tokens(mx.array([[t1_id]], dtype=mx.int32))

                if step + 1 >= self.settings.max_filler_slots:
                    r0_done = True

                if r0_done and r1_done:
                    break
                if r1_done and not r0_done:
                    bc = extract_lfm3_cache(bc, 0)
                    h = self.backbone(None, bc, e0)[:, -1]
                elif r0_done and not r1_done:
                    bc = extract_lfm3_cache(bc, 1)
                    h = self.backbone(None, bc, e1)[:, -1]
                else:
                    h = self.backbone(None, bc, mx.concatenate([e0, e1], axis=0))[:, -1]

            elif not r0_done:
                # Fast B=1 continuation for remaining Row 0 filler audio frames.
                if speaking:
                    audio_hist = (
                        mx.stack(list(r0_audio_hist))[None] if r0_audio_hist else None
                    )
                    codes = sample_frame(
                        self.model.depth_decoder,
                        h,
                        self.cfg.maskgit_schedule,
                        self.settings.audio_temperature,
                        self.settings.audio_top_k,
                        self.settings.audio_repetition_penalty,
                        audio_hist,
                        None,
                    )[0]
                    codes = normalize_eos(codes, self.cfg.audio_eos_code)
                    codes_np = np.asarray(codes, dtype=np.int32)
                    r0_raw_ids.append(int(self.cfg.audio_out_token_id))
                    r0_codes.append(codes_np)
                    if int(codes_np[0]) == self.cfg.audio_eos_code:
                        r0_done = True
                        flusher.flush(finalize=True)
                        break
                    r0_audio_hist.append(codes)
                    total_non_eos_frames += 1
                    flusher.push_frame(codes_np)
                    e0 = embed_frames(self.model.codebook_embedding, codes).reshape(1, 1, -1)
                else:
                    t0_id = self._sample_text_token(h, r0_text_hist, greedy=False)
                    r0_text_hist.append(t0_id)
                    r0_raw_ids.append(t0_id)
                    r0_text_ids.append(t0_id)
                    if t_first_tok is None:
                        t_first_tok = time.perf_counter()
                    if t0_id == self.cfg.text_end_token_id:
                        text_done = speaking = True
                    elif t0_id in self._eos_id_set:
                        r0_done = True
                        break
                    emit_filler_delta(force=False)
                    e0 = self.backbone.embed_tokens(mx.array([[t0_id]], dtype=mx.int32))

                if not text_done:
                    slots_left -= 1
                    if slots_left == 0:
                        speaking = not speaking
                        slots_left = (
                            int(self.cfg.interleaved_n_audio)
                            if speaking
                            else int(self.cfg.interleaved_n_text)
                        )
                if step + 1 >= self.settings.max_filler_slots:
                    break
                h = self.backbone(None, bc, e0)[:, -1]

            elif not r1_done:
                # Row 0 finished early; finish emitting Row 1 query at B=1.
                t1_id = self._sample_text_token(h, deque(), greedy=True)
                if (
                    t1_id == self._im_end_id
                    or t1_id in self._eos_id_set
                    or t1_id == self.cfg.text_end_token_id
                    or len(r1_tokens) >= self.settings.max_router_tokens
                ):
                    r1_done = True
                    maybe_dispatch_query()
                    break
                r1_tokens.append(t1_id)
                e1 = self.backbone.embed_tokens(mx.array([[t1_id]], dtype=mx.int32))
                h = self.backbone(None, bc, e1)[:, -1]

        if not flusher.decoder.finished:
            flusher.flush(finalize=True)
        if not query_dispatched:
            maybe_dispatch_query()

        emit_filler_delta(force=True)
        filler_text = clean_display_text(
            self.tokenizer.decode(r0_text_ids, skip_special_tokens=True)
        )
        filler_turn = _build_partial_turn(
            r0_raw_ids, r0_codes, r0_text_ids, self.tokenizer, self.cfg, self._im_end_id
        )
        t_end = time.perf_counter()
        ttft_ms = ((t_first_tok - t_start) * 1000.0) if t_first_tok else fork_merge_ms
        ttfa_filler_ms = (
            ((flusher.t_first_audio - t_start) * 1000.0)
            if flusher.t_first_audio
            else ((t_end - t_start) * 1000.0)
        )
        gen_time = max(1e-4, t_end - t_start - (fork_merge_ms / 1000.0))

        mx.clear_cache()
        return BatchPlanOutcome(
            decision="delegate",
            delegation_query=dispatched_query,
            filler_text=filler_text,
            filler_turn=filler_turn,
            filler_audio_frames=total_non_eos_frames,
            fork_merge_ms=fork_merge_ms,
            route_decision_ms=route_decision_ms,
            ttft_ms=ttft_ms,
            ttfa_filler_ms=ttfa_filler_ms,
            codec_chunk_ms_list=list(flusher.codec_ms_list),
            audio_chunks_sent=flusher.chunks_sent,
            first_chunk_samples=flusher.first_chunk_samples,
            slots_per_sec=steps_taken / gen_time if steps_taken else 0.0,
            batch_steps=steps_taken,
        )

    def run_speculative_selective_batch(
        self,
        conv_cache: list[Any],
        tools: list[ToolSpec] | None,
        persona: PersonaConfig,
        language_code: str,
        latest_user_text: str,
        dynamic_filler_enabled: bool,
        cancel_event: threading.Event,
        on_event: Callable[[dict[str, Any]], None],
        poll_mic_fn: Optional[Callable[[], list[dict[str, Any]]]] = None,
        user_audio_embeds: Optional[mx.array] = None,
        prev_assistant_text: str = "",
    ) -> BatchPlanOutcome:
        """Speculatively run Row 0 (assistant direct/tool) || Row 1 (clean audio router).

        Caller must have already closed `user_turn_open` and captured `pre_turn_snapshot`
        before invoking this method.
        """
        if cancel_event.is_set():
            return BatchPlanOutcome(decision="cancelled")

        t_start = time.perf_counter()
        profile = get_language_profile(language_code)

        c0 = clone_lfm3_cache(conv_cache)

        use_tools = bool(tools)
        header_ids = self._encode_text("<|im_start|>assistant\n<think></think>")
        if not use_tools:
            header_ids.append(int(self.cfg.mixed_start_token_id))
            mode = INTERLEAVED
            raw_token_ids = [int(self.cfg.mixed_start_token_id)]
        else:
            mode = TEXT
            raw_token_ids = []

        router_sys = (
            "You are a banking call router. Listen to or read the customer's latest utterance and output ONLY one line:\n"
            "- If it is a greeting, acknowledgement, yes/no, small talk, or provides personal income/loan/contact numbers for pre-qualification, repayment calculation, application submission, or callback -> output exactly: DIRECT\n"
            f"- If it contains ANY question or information request about credit cards, tiers, fees, interest rates, limits, cashback, foreign exchange, airport lounges, travel insurance, 55-day grace period, FICA, POPIA, NCA, fraud, or bank policies (even alongside personal preferences) -> output: DELEGATE: <concise standalone question in {profile.router_language_name}>\n"
            "Never answer the question. Output ONLY DIRECT or DELEGATE: <question>."
        )

        h0 = self._prefill_single(c0, header_ids)
        c1, h1 = self._build_clean_row1_cache(
            system_prompt=router_sys,
            latest_user_text=latest_user_text,
            user_audio_embeds=user_audio_embeds,
            prev_assistant_text=prev_assistant_text,
        )
        bc = merge_lfm3_caches([c0, c1])
        h = mx.concatenate([h0, h1], axis=0)
        mx.eval(h)
        fork_merge_ms = (time.perf_counter() - t_start) * 1000.0

        # Speculative phase state for Row 0 (buffered up to <=8 steps while Row 1 decides)
        text_token_ids: list[int] = []
        raw_audio_codes: list[np.ndarray] = []
        text_history: deque[int] = deque(maxlen=32)
        audio_history: deque[mx.array] = deque(maxlen=self.settings.audio_repetition_window)
        slots_left = int(self.cfg.interleaved_n_text)
        speaking = False
        text_done = False
        r0_finished = False
        last_token: Optional[int] = None
        last_unfed_embed: Optional[mx.array] = None
        t_first_tok: Optional[float] = None

        r1_tokens: list[int] = []
        r1_done = False
        decision_kind: Optional[str] = None

        max_spec_steps = 8
        for step in range(max_spec_steps):
            if cancel_event.is_set():
                return BatchPlanOutcome(decision="cancelled")

            interleaving = mode == INTERLEAVED
            if mode == AUDIO or (interleaving and speaking):
                audio_hist = (
                    mx.stack(list(audio_history))[None] if audio_history else None
                )
                codes = sample_frame(
                    self.model.depth_decoder,
                    h[0:1],
                    self.cfg.maskgit_schedule,
                    self.settings.audio_temperature,
                    self.settings.audio_top_k,
                    self.settings.audio_repetition_penalty,
                    audio_hist,
                    None,
                )[0]
                codes = normalize_eos(codes, self.cfg.audio_eos_code)
                codes_np = np.asarray(codes, dtype=np.int32)
                raw_token_ids.append(int(self.cfg.audio_out_token_id))
                raw_audio_codes.append(codes_np)
                if int(codes_np[0]) == self.cfg.audio_eos_code:
                    mode = TEXT
                    r0_finished = True
                else:
                    audio_history.append(codes)
                e0 = embed_frames(self.model.codebook_embedding, codes).reshape(1, 1, -1)
                last_token = int(self.cfg.audio_out_token_id)
            else:
                tok0 = self._sample_text_token(h[0:1], text_history, greedy=False)
                text_history.append(tok0)
                raw_token_ids.append(tok0)
                text_token_ids.append(tok0)
                last_token = tok0
                if t_first_tok is None and tok0 not in (
                    self.cfg.mixed_start_token_id,
                    self.cfg.audio_start_token_id,
                ):
                    t_first_tok = time.perf_counter()

                if tok0 == self.cfg.text_end_token_id:
                    if interleaving:
                        text_done = speaking = True
                    else:
                        r0_finished = True
                elif tok0 in self._eos_id_set:
                    r0_finished = True
                elif interleaving:
                    pass
                elif tok0 == self.cfg.audio_start_token_id:
                    mode = AUDIO
                elif tok0 == self.cfg.mixed_start_token_id:
                    mode = INTERLEAVED
                    slots_left = int(self.cfg.interleaved_n_text)
                    speaking = text_done = False

                e0 = self.backbone.embed_tokens(mx.array([[tok0]], dtype=mx.int32))

            if interleaving and not text_done:
                slots_left -= 1
                if slots_left == 0:
                    speaking = not speaking
                    slots_left = (
                        int(self.cfg.interleaved_n_audio)
                        if speaking
                        else int(self.cfg.interleaved_n_text)
                    )

            # Step Row 1 router (text only)
            t1_id = self._sample_text_token(h[1:2], deque(), greedy=True)
            if (
                t1_id == self._im_end_id
                or t1_id in self._eos_id_set
                or t1_id == self.cfg.text_end_token_id
            ):
                r1_done = True
                e1 = self.backbone.embed_tokens(mx.array([[self._im_end_id]], dtype=mx.int32))
            else:
                r1_tokens.append(t1_id)
                e1 = self.backbone.embed_tokens(mx.array([[t1_id]], dtype=mx.int32))

            # Advance batched hidden states so whichever row wins is ready cleanly.
            h = self.backbone(None, bc, mx.concatenate([e0, e1], axis=0))[:, -1]

            # Check if Row 0 already initiated a local CRM tool call.
            spec_probe = clean_display_text(
                self.tokenizer.decode(text_token_ids, skip_special_tokens=True)
            ).lstrip()
            if use_tools and (
                spec_probe.startswith("<tool_call") or spec_probe.startswith("<function")
            ):
                decision_kind = "direct"
                break

            r1_probe = clean_display_text(
                self.tokenizer.decode(r1_tokens, skip_special_tokens=True)
            ).strip()
            r1_upper = r1_probe.upper()

            if "DELEGATE" in r1_upper:
                decision_kind = "delegate"
                break
            if "DIRECT" in r1_upper or r1_done:
                decision_kind = "direct"
                break

        route_decision_ms = (time.perf_counter() - t_start) * 1000.0
        if decision_kind is None:
            decision_kind = "direct"

        if decision_kind == "delegate":
            c_router = extract_lfm3_cache(bc, 1)
            h_router = h[1:2]
            if dynamic_filler_enabled:
                c_filler = clone_lfm3_cache(conv_cache)
                filler_hdr = self._encode_text(
                    f"<|im_start|>user\n{profile.filler_instruction}<|im_end|>\n"
                    "<|im_start|>assistant\n<think></think>"
                ) + [int(self.cfg.mixed_start_token_id)]
                h_filler = self._prefill_single(c_filler, filler_hdr)
                bc2 = merge_lfm3_caches([c_filler, c_router])
                h_comb = mx.concatenate([h_filler, h_router], axis=0)
                mx.eval(h_comb)
                return self._run_filler_query_loop(
                    bc=bc2,
                    h=h_comb,
                    initial_r1_tokens=r1_tokens,
                    r1_initially_done=r1_done,
                    strip_delegate_prefix=True,
                    latest_user_text=latest_user_text,
                    t_start=t_start,
                    fork_merge_ms=fork_merge_ms,
                    route_decision_ms=route_decision_ms,
                    cancel_event=cancel_event,
                    on_event=on_event,
                    poll_mic_fn=poll_mic_fn,
                )

            # Dynamic filler disabled: finish Row 1 query at B=1 quickly and delegate directly.
            while not r1_done and len(r1_tokens) < self.settings.max_router_tokens:
                t1_id = self._sample_text_token(h_router, deque(), greedy=True)
                if (
                    t1_id == self._im_end_id
                    or t1_id in self._eos_id_set
                    or t1_id == self.cfg.text_end_token_id
                ):
                    r1_done = True
                    break
                r1_tokens.append(t1_id)
                h_router = self.backbone(
                    None,
                    c_router,
                    self.backbone.embed_tokens(mx.array([[t1_id]], dtype=mx.int32)),
                )[:, -1]
            _, q = _parse_router_query(
                self.tokenizer.decode(r1_tokens, skip_special_tokens=True),
                latest_user_text,
            )
            on_event(
                {
                    "type": "delegation_query_ready",
                    "query": q,
                    "elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
                }
            )
            mx.clear_cache()
            return BatchPlanOutcome(
                decision="delegate",
                delegation_query=q,
                fork_merge_ms=fork_merge_ms,
                route_decision_ms=route_decision_ms,
                ttft_ms=route_decision_ms,
            )

        # DIRECT (or local CRM tool_calls): extract Row 0 cache & continue at fast B=1!
        c0_only = extract_lfm3_cache(bc, 0)
        h0_cur = h[0:1]

        return self._continue_direct_b1(
            c0=c0_only,
            h=h0_cur,
            raw_token_ids=raw_token_ids,
            raw_audio_codes=raw_audio_codes,
            text_token_ids=text_token_ids,
            text_history=text_history,
            audio_history=audio_history,
            mode=mode,
            slots_left=slots_left,
            speaking=speaking,
            text_done=text_done,
            finished=r0_finished,
            last_token=last_token,
            last_unfed_embed=last_unfed_embed,
            use_tools=use_tools,
            persona=persona,
            t_start=t_start,
            t_first_tok=t_first_tok,
            fork_merge_ms=fork_merge_ms,
            route_decision_ms=route_decision_ms,
            cancel_event=cancel_event,
            on_event=on_event,
            poll_mic_fn=poll_mic_fn,
        )

    def _continue_direct_b1(
        self,
        c0: list[Any],
        h: mx.array,
        raw_token_ids: list[int],
        raw_audio_codes: list[np.ndarray],
        text_token_ids: list[int],
        text_history: deque[int],
        audio_history: deque[mx.array],
        mode: str,
        slots_left: int,
        speaking: bool,
        text_done: bool,
        finished: bool,
        last_token: Optional[int],
        last_unfed_embed: Optional[mx.array],
        use_tools: bool,
        persona: PersonaConfig,
        t_start: float,
        t_first_tok: Optional[float],
        fork_merge_ms: float,
        route_decision_ms: float,
        cancel_event: threading.Event,
        on_event: Callable[[dict[str, Any]], None],
        poll_mic_fn: Optional[Callable[[], list[dict[str, Any]]]] = None,
    ) -> BatchPlanOutcome:
        """Replay speculative Row 0 buffered frames/text, then stream remainder in B=1."""
        flusher = StreamingAudioFlusher(
            self.codec,
            first_chunk_frames=self.settings.first_chunk_frames,
            stream_chunk_frames=self.settings.stream_chunk_frames,
            on_event=on_event,
            segment_name="answer",
        )
        total_non_eos_frames = 0
        for row in raw_audio_codes:
            if int(row[0]) != self.cfg.audio_eos_code:
                total_non_eos_frames += 1
                flusher.push_frame(row)
            else:
                flusher.flush(finalize=True)

        emitted_text = ""
        probe_init = clean_display_text(
            self.tokenizer.decode(text_token_ids, skip_special_tokens=True)
        ).lstrip()
        text_streaming_allowed = (
            mode == INTERLEAVED
            or not use_tools
            or (len(probe_init) >= 16 and "<" not in probe_init)
        )
        mode_decided = mode in (INTERLEAVED, AUDIO) or (
            probe_init.startswith("<tool_call") or probe_init.startswith("<function")
        )

        def maybe_emit_text(force: bool = False) -> None:
            nonlocal emitted_text
            if not text_streaming_allowed or not text_token_ids:
                return
            decoded_full = clean_display_text(
                self.tokenizer.decode(text_token_ids, skip_special_tokens=True)
            )
            if not force and decoded_full.endswith("\ufffd"):
                return
            if decoded_full.startswith(emitted_text):
                delta = decoded_full[len(emitted_text) :]
                if delta:
                    emitted_text = decoded_full
                    on_event(
                        {
                            "type": "text_delta",
                            "delta": delta,
                            "text": decoded_full,
                            "segment": "answer",
                        }
                    )

        if text_token_ids and not (
            probe_init.startswith("<tool_call") or probe_init.startswith("<function")
        ):
            maybe_emit_text(force=False)

        slot_count = len(raw_token_ids)
        if not finished:
            for step in range(slot_count, self.settings.max_slots):
                if cancel_event.is_set():
                    p_turn = _build_partial_turn(
                        raw_token_ids, raw_audio_codes, text_token_ids, self.tokenizer, self.cfg, self._im_end_id
                    )
                    return BatchPlanOutcome(
                        decision="cancelled",
                        filler_turn=p_turn,
                        direct_outcome=TurnOutcome(
                            kind="cancelled",
                            cancelled=True,
                            turn_record=p_turn,
                            text=p_turn.text if p_turn else "",
                            audio_frames=total_non_eos_frames,
                        ),
                    )

                if (
                    poll_mic_fn is not None
                    and step > 0
                    and step % self.settings.coop_listen_slot_interval == 0
                ):
                    if any(ev.get("type") == "barge_in_detected" for ev in poll_mic_fn()):
                        cancel_event.set()
                        p_turn = _build_partial_turn(
                            raw_token_ids, raw_audio_codes, text_token_ids, self.tokenizer, self.cfg, self._im_end_id
                        )
                        return BatchPlanOutcome(
                            decision="cancelled",
                            filler_turn=p_turn,
                            direct_outcome=TurnOutcome(
                                kind="cancelled",
                                cancelled=True,
                                turn_record=p_turn,
                                text=p_turn.text if p_turn else "",
                                audio_frames=total_non_eos_frames,
                            ),
                        )

                interleaving = mode == INTERLEAVED
                if mode == AUDIO or (interleaving and speaking):
                    audio_hist = (
                        mx.stack(list(audio_history))[None] if audio_history else None
                    )
                    codes = sample_frame(
                        self.model.depth_decoder,
                        h,
                        self.cfg.maskgit_schedule,
                        self.settings.audio_temperature,
                        self.settings.audio_top_k,
                        self.settings.audio_repetition_penalty,
                        audio_hist,
                        None,
                    )[0]
                    codes = normalize_eos(codes, self.cfg.audio_eos_code)
                    codes_np = np.asarray(codes, dtype=np.int32)
                    raw_token_ids.append(int(self.cfg.audio_out_token_id))
                    raw_audio_codes.append(codes_np)
                    if int(codes_np[0]) == self.cfg.audio_eos_code:
                        mode = TEXT
                        flusher.flush(finalize=True)
                    else:
                        audio_history.append(codes)
                        total_non_eos_frames += 1
                        flusher.push_frame(codes_np)
                    embed = embed_frames(self.model.codebook_embedding, codes)
                    last_token = int(self.cfg.audio_out_token_id)
                else:
                    token = self._sample_text_token(h, text_history, greedy=False)
                    text_history.append(token)
                    raw_token_ids.append(token)
                    text_token_ids.append(token)
                    last_token = token
                    if t_first_tok is None and token not in (
                        self.cfg.mixed_start_token_id,
                        self.cfg.audio_start_token_id,
                    ):
                        t_first_tok = time.perf_counter()

                    if token == self.cfg.text_end_token_id:
                        if interleaving:
                            text_done = speaking = True
                        else:
                            finished = True
                    elif token in self._eos_id_set:
                        finished = True
                    elif interleaving:
                        pass
                    elif token == self.cfg.audio_start_token_id:
                        mode = AUDIO
                    elif token == self.cfg.mixed_start_token_id:
                        mode = INTERLEAVED
                        slots_left = int(self.cfg.interleaved_n_text)
                        speaking = text_done = False

                    if not mode_decided:
                        if token == self.cfg.mixed_start_token_id:
                            mode_decided = True
                            text_streaming_allowed = True
                        elif token == self.cfg.audio_start_token_id:
                            mode_decided = True
                            text_streaming_allowed = False
                        else:
                            probe = self.tokenizer.decode(
                                text_token_ids, skip_special_tokens=True
                            ).lstrip()
                            if probe.startswith("<tool_call") or probe.startswith("<function"):
                                mode_decided = True
                                text_streaming_allowed = False
                            elif len(probe) >= 16 and "<" not in probe:
                                mode_decided = True
                                text_streaming_allowed = True
                    maybe_emit_text(force=False)
                    embed = self.backbone.embed_tokens(mx.array([token], dtype=mx.int32))

                slot_count += 1
                if interleaving and not text_done:
                    slots_left -= 1
                    if slots_left == 0:
                        speaking = not speaking
                        slots_left = (
                            int(self.cfg.interleaved_n_audio)
                            if speaking
                            else int(self.cfg.interleaved_n_text)
                        )

                if finished or step == self.settings.max_slots - 1:
                    last_unfed_embed = embed
                    break

                h = self.backbone(None, c0, embed.reshape(1, 1, -1))[:, -1]

        if not flusher.decoder.finished:
            flusher.flush(finalize=True)

        raw_full_text = self.tokenizer.decode(text_token_ids, skip_special_tokens=True)
        tool_calls = parse_tool_calls(raw_full_text) if use_tools else []
        if tool_calls:
            ttft_ms = ((t_first_tok - t_start) * 1000.0) if t_first_tok else route_decision_ms
            return BatchPlanOutcome(
                decision="tool_calls",
                fork_merge_ms=fork_merge_ms,
                route_decision_ms=route_decision_ms,
                ttft_ms=ttft_ms,
                direct_outcome=TurnOutcome(
                    kind="tool_calls",
                    tool_calls=tool_calls,
                    text=raw_full_text,
                    prefill_ms=fork_merge_ms,
                    ttft_ms=ttft_ms,
                ),
            )

        clean_text = clean_display_text(raw_full_text)
        if not emitted_text and clean_text:
            text_streaming_allowed = True
            maybe_emit_text(force=True)

        used_tts_fallback = False
        if (
            total_non_eos_frames == 0
            and clean_text
            and self._tts_fallback_fn is not None
            and not cancel_event.is_set()
        ):
            def on_tts_event(ev: dict[str, Any]) -> None:
                if ev.get("type") == "audio_chunk":
                    flusher.chunks_sent += 1
                    if flusher.first_chunk_samples == 0:
                        flusher.first_chunk_samples = int(ev.get("samples") or 0)
                    if flusher.t_first_audio is None:
                        flusher.t_first_audio = time.perf_counter()
                    if "codec_ms" in ev:
                        flusher.codec_ms_list.append(float(ev["codec_ms"]))
                on_event(ev)

            tts_codes, _tts_ids = self._tts_fallback_fn(
                text=clean_text,
                persona=persona,
                cancel_event=cancel_event,
                on_event=on_tts_event,
            )
            if cancel_event.is_set():
                return BatchPlanOutcome(decision="cancelled")
            if tts_codes is not None and tts_codes.shape[0] > 0:
                used_tts_fallback = True
                total_non_eos_frames = int(
                    np.sum(~np.any(tts_codes == self.cfg.audio_eos_code, axis=-1))
                )

        if used_tts_fallback:
            # Keep already-generated text tokens in c0 so the model stays aware of its prior reply.
            turn_record = TurnRecord(
                role="assistant",
                text=clean_text,
                raw_token_ids=None,
                raw_audio_codes=None,
            )
            need_eos_frame = False
        else:
            need_eos_frame = bool(
                raw_audio_codes and not np.any(raw_audio_codes[-1] == self.cfg.audio_eos_code)
            )
            if need_eos_frame:
                raw_audio_codes.append(
                    np.full((12,), self.cfg.audio_eos_code, dtype=np.int32)
                )
                raw_token_ids.append(int(self.cfg.audio_out_token_id))
            if not raw_token_ids or raw_token_ids[-1] != self._im_end_id:
                raw_token_ids.append(self._im_end_id)

            turn_record = TurnRecord(
                role="assistant",
                text=clean_text,
                raw_token_ids=raw_token_ids,
                raw_audio_codes=(
                    np.stack(raw_audio_codes, axis=0).astype(np.int32)
                    if raw_audio_codes
                    else None
                ),
            )

        # Always advance c0 cleanly past closing `<|im_end|>\n` (preserving readable text).
        tail_embeds: list[mx.array] = []
        if last_unfed_embed is not None:
            tail_embeds.append(last_unfed_embed.reshape(1, 1, -1))
        if need_eos_frame:
            eos_codes = mx.full((12,), self.cfg.audio_eos_code, dtype=mx.int32)
            tail_embeds.append(
                embed_frames(self.model.codebook_embedding, eos_codes).reshape(1, 1, -1)
            )
        closing_ids: list[int] = []
        if last_token != self._im_end_id:
            closing_ids.append(self._im_end_id)
        closing_ids.extend(self._newline_ids)
        if closing_ids:
            tail_embeds.append(
                self.backbone.embed_tokens(mx.array([closing_ids], dtype=mx.int32))
            )
        if tail_embeds:
            joined = mx.concatenate(tail_embeds, axis=1)
            self.backbone(None, c0, joined)
        eval_cache(c0)

        t_end = time.perf_counter()
        ttft_ms = ((t_first_tok - t_start) * 1000.0) if t_first_tok else route_decision_ms
        ttfa_ms = (
            ((flusher.t_first_audio - t_start) * 1000.0)
            if flusher.t_first_audio
            else ttft_ms
        )
        gen_time = max(1e-4, t_end - t_start - (fork_merge_ms / 1000.0))
        slots_per_sec = slot_count / gen_time if slot_count else 0.0
        avg_codec_ms = (
            sum(flusher.codec_ms_list) / len(flusher.codec_ms_list)
            if flusher.codec_ms_list
            else 0.0
        )

        outcome = TurnOutcome(
            kind="response",
            turn_record=turn_record,
            text=clean_text,
            audio_frames=total_non_eos_frames,
            prefill_ms=fork_merge_ms,
            ttfa_ms=ttfa_ms,
            ttft_ms=ttft_ms,
            slots_per_sec=slots_per_sec,
            audio_chunks_sent=flusher.chunks_sent,
            first_chunk_samples=flusher.first_chunk_samples,
            codec_chunk_avg_ms=avg_codec_ms,
            gen_wall_ms=(t_end - t_start) * 1000.0,
        )
        return BatchPlanOutcome(
            decision="direct",
            direct_outcome=outcome,
            updated_conv_cache=c0,
            fork_merge_ms=fork_merge_ms,
            route_decision_ms=route_decision_ms,
            ttft_ms=ttft_ms,
            ttfa_direct_ms=ttfa_ms,
            codec_chunk_ms_list=list(flusher.codec_ms_list),
            audio_chunks_sent=flusher.chunks_sent,
            first_chunk_samples=flusher.first_chunk_samples,
            slots_per_sec=slots_per_sec,
            batch_steps=slot_count,
        )
