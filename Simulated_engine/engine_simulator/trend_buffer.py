"""
trend_buffer.py
----------------
Closes the trend-analysis gap: until now, rolling mean/std only existed as
internal ML feature inputs (see feature_adapter.py), and the Angular
dashboard's charts computed their own history purely client-side from the
live WS stream (resets on refresh, not queryable by anyone else). This
gives the backend an actual, queryable answer to "give me sensor X's
trend over the last N ticks for this engine" -- a real function, not
implicit client bookkeeping.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional


class TrendBuffer:
    """Keeps a rolling per-key history of (timestamp_ms, value) points for
    one simulator session, and answers min/max/avg/last summary stats."""

    def __init__(self, maxlen: int = 3000):
        self.maxlen = maxlen
        self._series: Dict[str, deque] = {}

    def reset(self) -> None:
        self._series.clear()

    def record(self, t_ms: int, variables: Dict[str, Any]) -> None:
        for key, value in variables.items():
            if value is None or isinstance(value, (str, bool)):
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            buf = self._series.get(key)
            if buf is None:
                buf = deque(maxlen=self.maxlen)
                self._series[key] = buf
            buf.append((t_ms, v))

    def keys(self) -> List[str]:
        return sorted(self._series.keys())

    def get_trend(self, key: str, n: Optional[int] = None) -> List[Dict[str, float]]:
        buf = self._series.get(key)
        if not buf:
            return []
        points = list(buf)[-n:] if n else list(buf)
        return [{"t_ms": t, "value": v} for t, v in points]

    def get_stats(self, key: str) -> Optional[Dict[str, float]]:
        buf = self._series.get(key)
        if not buf:
            return None
        values = [v for _, v in buf]
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "last": values[-1],
            "n": len(values),
        }
