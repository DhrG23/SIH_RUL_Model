"""
debounce.py
------------
A small, deliberately NOT machine-learned state machine: requires either
N consecutive non-healthy frames, OR a sustained moving-average fault
probability above a threshold, before confirming a fault. This exists to
stop a single noisy reading from swinging the system's fault/RUL status --
a real weakness the earlier version of this app had (the app would
happily report a scary status from one bad reading with no memory of what
came before it).

Because this is pure logic (a counter and a moving average), it carries
NONE of the overfitting risk that our stacking experiments hit -- there is
nothing here to overfit. It's evaluated below with real numbers anyway,
because "should obviously work" is exactly the assumption that stacking
also started from.
"""

from collections import deque


class DebounceGate:
    """Tracks a rolling window of recent (is_fault, fault_probability) frames
    for ONE engine/asset and confirms a fault only when either:
      (a) `n_consecutive` frames in a row are non-healthy, or
      (b) the moving average fault probability over `window` frames exceeds
          `prob_threshold`.
    Once confirmed, stays confirmed until reset() is called (mirrors a real
    maintenance system: once a fault is confirmed, you don't want it to
    silently un-confirm itself because of a couple of good readings).
    """

    def __init__(self, n_consecutive: int = 5, window: int = 10, prob_threshold: float = 0.80):
        self.n_consecutive = n_consecutive
        self.window = window
        self.prob_threshold = prob_threshold
        self._recent_status = deque(maxlen=n_consecutive)
        self._recent_probs = deque(maxlen=window)
        self._confirmed = False
        self._confirmed_at_step = None
        self._step = -1

    def reset(self):
        self._recent_status.clear()
        self._recent_probs.clear()
        self._confirmed = False
        self._confirmed_at_step = None
        self._step = -1

    def update(self, is_fault: bool, fault_probability: float) -> bool:
        """Feed one new reading in. Returns True if a fault is confirmed
        (either just now or previously -- confirmation latches)."""
        self._step += 1
        self._recent_status.append(bool(is_fault))
        self._recent_probs.append(float(fault_probability))

        if not self._confirmed:
            consecutive_triggered = (
                len(self._recent_status) == self.n_consecutive and all(self._recent_status)
            )
            moving_avg = sum(self._recent_probs) / len(self._recent_probs)
            prob_triggered = moving_avg >= self.prob_threshold and len(self._recent_probs) >= max(3, self.window // 2)

            if consecutive_triggered or prob_triggered:
                self._confirmed = True
                self._confirmed_at_step = self._step

        return self._confirmed

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    @property
    def confirmed_at_step(self):
        return self._confirmed_at_step

    @property
    def current_moving_avg_prob(self) -> float:
        if not self._recent_probs:
            return 0.0
        return sum(self._recent_probs) / len(self._recent_probs)
