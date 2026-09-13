"""
online_adapter.py -- "new data gets randomly distributed to a random
family model" adaptive learning.

Why per-variant, not per-family or global: if every new reading updated
EVERY model, the whole family would drift together and lose the
diversity that makes the family-of-variants idea useful in the first
place (that's the "overfitting/hallucination" risk you flagged). Instead,
each new labeled sample is routed to exactly ONE randomly chosen variant
per family, appended to that variant's own small buffer, and only THAT
variant gets refit -- so at any moment, different variants have seen
slightly different slices of recent data, preserving the disagreement the
family-level combiner needs something to reconcile.

Honesty note: most of these estimators (k-NN, RandomForest, SVM, Ridge)
have no true incremental partial_fit in sklearn -- "online update" here
means "refit that one variant on its own bounded buffer," not a
per-sample gradient step. That's a real, if coarser, form of online
adaptation, not a claim of true incremental learning.
"""
import numpy as np
from collections import deque
from sklearn.base import clone


class OnlineFamilyAdapter:
    def __init__(self, families_dict, buffer_size=200, seed=0, refit_every=20):
        """families_dict: {family_name: [fitted_variant, ...]} -- pass the
        SAME dict a FamilyClassifierStack/FamilyRULStack already holds
        (self.fitted_variants_), so updates land on the live models."""
        self.families = families_dict
        self.buffer_size, self.refit_every = buffer_size, refit_every
        self.rng = np.random.default_rng(seed)
        self._buffers = {fam: [deque(maxlen=buffer_size) for _ in variants]
                          for fam, variants in families_dict.items()}
        self._since_refit = {fam: [0] * len(variants) for fam, variants in families_dict.items()}

    def add_labeled_example(self, x, y):
        """Call once a new reading's true label/TTF becomes known. Routes
        it to ONE randomly chosen variant index per family."""
        for fam, variants in self.families.items():
            j = int(self.rng.integers(len(variants)))
            self._buffers[fam][j].append((x, y))
            self._since_refit[fam][j] += 1
            if self._since_refit[fam][j] >= self.refit_every and len(self._buffers[fam][j]) >= 10:
                self._refit_variant(fam, j)
                self._since_refit[fam][j] = 0

    def _refit_variant(self, fam, j):
        buf = self._buffers[fam][j]
        X_buf = np.array([b[0] for b in buf])
        y_buf = np.array([b[1] for b in buf])
        try:
            self.families[fam][j] = clone(self.families[fam][j]).fit(X_buf, y_buf)
        except Exception:
            pass  # a variant that can't be refit on this buffer (e.g. too few classes) keeps its last good fit
