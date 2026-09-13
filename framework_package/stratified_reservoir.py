"""
Stratified reservoir buffer -- replaces the plain FIFO deque used for
replay. A FIFO deque forgets whatever falls off the front regardless of
how rare or important it was; a long stable flight full of "healthy"
frames would silently flush out the few precious early-onset fault
examples. This buffer instead reserves a fixed capacity PER CLASS (for
classification) or PER SEVERITY BUCKET (for regression, via quantile
binning), so critical, rare examples survive indefinitely once captured
rather than aging out.

Classic reservoir sampling per stratum: each stratum accepts new items
until full, then accepts a new item with probability capacity/n_seen_in_
stratum, replacing a uniformly random existing item -- this keeps each
stratum an unbiased random sample of everything ever seen in it, not just
the most recent window.
"""
import numpy as np


class StratifiedReservoir:
    def __init__(self, capacity_per_stratum=150, n_severity_bins=5, seed=0):
        self.capacity = capacity_per_stratum
        self.n_bins = n_severity_bins
        self.rng = np.random.default_rng(seed)
        self.buckets = {}       # stratum key -> list[(x, y)]
        self.seen_count = {}    # stratum key -> total items ever offered to this stratum

    def _stratum_key(self, y, is_classification):
        if is_classification:
            return y  # class label IS the stratum
        # regression: bucket by quantile of the values seen so far (coarse,
        # recomputed lazily -- good enough for a replay buffer, not meant
        # to be an exact online quantile estimator)
        all_y = [yy for b in self.buckets.values() for _, yy in b] + [y]
        edges = np.quantile(all_y, np.linspace(0, 1, self.n_bins + 1)[1:-1]) if len(all_y) > 1 else []
        return int(np.searchsorted(edges, y))

    def add(self, x, y, is_classification=True):
        key = self._stratum_key(y, is_classification)
        self.seen_count[key] = self.seen_count.get(key, 0) + 1
        bucket = self.buckets.setdefault(key, [])
        if len(bucket) < self.capacity:
            bucket.append((x, y))
        else:
            j = self.rng.integers(0, self.seen_count[key])
            if j < self.capacity:
                bucket[j] = (x, y)

    def add_batch(self, X, y, is_classification=True):
        for xi, yi in zip(X, y):
            self.add(xi, yi, is_classification)

    def as_arrays(self):
        all_items = [item for b in self.buckets.values() for item in b]
        if not all_items:
            return np.array([]), np.array([])
        X = np.array([it[0] for it in all_items])
        y = np.array([it[1] for it in all_items])
        return X, y

    def __len__(self):
        return sum(len(b) for b in self.buckets.values())
