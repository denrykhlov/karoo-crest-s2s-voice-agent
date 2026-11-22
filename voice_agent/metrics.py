"""Speech delivery, latency breakdown, and barge-in TTFA telemetry tracker."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional


def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return round(ordered[lo], 1)
    val = ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)
    return round(val, 1)


def _metric_summary(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"last": None, "avg": None, "p50": None, "p95": None, "count": 0}
    return {
        "last": round(values[-1], 1),
        "avg": round(sum(values) / len(values), 1),
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "count": len(values),
    }


@dataclass(slots=True)
class TurnDeliveryMetrics:
    """Per-turn speech delivery and delegation timing record (milliseconds unless noted)."""

    turn_id: str
    turn_index: int
    language: str
    delegation_mode: str
    route_decision: str  # "greeting" | "direct" | "delegate" | "tool_calls"
    after_barge_in: bool = False
    interrupted: bool = False
    endpoint_ms: Optional[float] = None
    prefill_ms: float = 0.0
    ttft_ms: float = 0.0
    ttfa_server_ms: float = 0.0
    ttfa_filler_ms: Optional[float] = None
    ttfa_answer_ms: Optional[float] = None
    ttfa_perceived_ms: Optional[float] = None
    delegation_ms: Optional[float] = None
    delegation_source: Optional[str] = None
    barge_in_cancel_ms: Optional[float] = None
    ttfa_after_barge_in_ms: Optional[float] = None
    audio_chunks_sent: int = 0
    first_chunk_samples: int = 0
    codec_chunk_avg_ms: float = 0.0
    audio_duration_sec: float = 0.0
    gen_rtf: float = 0.0
    slots_per_sec: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeliveryStatsTracker:
    """Tracks per-turn and call-level streaming speech delivery metrics."""

    def __init__(self) -> None:
        self.turns: list[TurnDeliveryMetrics] = []
        self._pending_perceived_ms: dict[str, float] = {}
        self._pending_barge_in: bool = False
        self._pending_barge_in_cancel_ms: Optional[float] = None
        self._pending_barge_in_ts: float = 0.0
        self._barge_in_cancel_samples: list[float] = []
        self.barge_in_count: int = 0

    def reset(self) -> None:
        self.turns.clear()
        self._pending_perceived_ms.clear()
        self._pending_barge_in = False
        self._pending_barge_in_cancel_ms = None
        self._pending_barge_in_ts = 0.0
        self._barge_in_cancel_samples.clear()
        self.barge_in_count = 0

    def mark_barge_in(self, cancel_ms: Optional[float] = None) -> None:
        """Flag that playback/generation was interrupted so next response records barge-in TTFA."""
        self.barge_in_count += 1
        self._pending_barge_in = True
        rounded = round(float(cancel_ms), 1) if cancel_ms is not None else None
        self._pending_barge_in_cancel_ms = rounded
        if rounded is not None:
            self._barge_in_cancel_samples.append(rounded)
        self._pending_barge_in_ts = time.perf_counter()

    def consume_pending_barge_in(self) -> tuple[bool, Optional[float]]:
        """Consume pending barge-in state when recording the next assistant turn."""
        if not self._pending_barge_in:
            return False, None
        cancel_ms = self._pending_barge_in_cancel_ms
        self._pending_barge_in = False
        self._pending_barge_in_cancel_ms = None
        return True, cancel_ms

    def record_turn(self, metrics: TurnDeliveryMetrics) -> TurnDeliveryMetrics:
        """Append one completed assistant turn's delivery metrics."""
        if not metrics.timestamp:
            metrics.timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        pending_perc = self._pending_perceived_ms.pop(metrics.turn_id, None)
        if pending_perc is not None and metrics.ttfa_perceived_ms is None:
            metrics.ttfa_perceived_ms = pending_perc
        if metrics.after_barge_in:
            if metrics.ttfa_perceived_ms is not None:
                metrics.ttfa_after_barge_in_ms = metrics.ttfa_perceived_ms
            elif metrics.ttfa_after_barge_in_ms is None:
                metrics.ttfa_after_barge_in_ms = metrics.ttfa_server_ms
        self.turns.append(metrics)
        return metrics

    def update_perceived_ttfa(self, turn_id: str, perceived_ms: float) -> bool:
        """Attach actual browser AudioWorklet audible onset latency to an active or finished turn."""
        rounded = round(max(0.0, float(perceived_ms)), 1)
        for turn in reversed(self.turns):
            if turn.turn_id == turn_id:
                if turn.ttfa_perceived_ms is None:
                    turn.ttfa_perceived_ms = rounded
                    if turn.after_barge_in:
                        turn.ttfa_after_barge_in_ms = rounded
                    return True
                return False
        if turn_id and turn_id not in self._pending_perceived_ms:
            self._pending_perceived_ms[turn_id] = rounded
        return False

    def summary(self, include_all_turns: bool = False) -> dict[str, Any]:
        """Compute aggregate last/avg/p50/p95 stats and turn history."""
        metric_fields = (
            "endpoint_ms",
            "prefill_ms",
            "ttft_ms",
            "ttfa_server_ms",
            "ttfa_perceived_ms",
            "ttfa_filler_ms",
            "ttfa_answer_ms",
            "delegation_ms",
            "barge_in_cancel_ms",
            "ttfa_after_barge_in_ms",
            "codec_chunk_avg_ms",
            "gen_rtf",
            "slots_per_sec",
        )
        aggregates: dict[str, dict[str, Optional[float]]] = {}
        for field_name in metric_fields:
            if field_name == "barge_in_cancel_ms" and self._barge_in_cancel_samples:
                aggregates[field_name] = _metric_summary(self._barge_in_cancel_samples)
                continue
            vals = [
                float(getattr(t, field_name))
                for t in self.turns
                if getattr(t, field_name) is not None
            ]
            aggregates[field_name] = _metric_summary(vals)

        result: dict[str, Any] = {
            "turn_count": len(self.turns),
            "barge_in_count": self.barge_in_count,
            "aggregates": aggregates,
            "recent_turns": [t.to_dict() for t in self.turns[-20:]],
        }
        if include_all_turns:
            result["turns"] = [t.to_dict() for t in self.turns]
        return result
