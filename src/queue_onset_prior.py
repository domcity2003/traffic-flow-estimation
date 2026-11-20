"""Onset features that carry link identity as transferable statistics.

For each link: how often train onset windows had it queued at step 6, overall
and weighted by time-of-day distance to the window (Gaussian, 60 min), plus
the same for its neighbours. A pooled model on these (without raw link ids)
can learn "recurrent bottleneck near its usual hour" across panels.
"""
from __future__ import annotations

import numpy as np

from src.queue_model_lab import xy


class OnsetPrior:
    def __init__(self, z_train, bandwidth=60.0):
        self.hits = z_train["truth"][:, 5, :].astype(np.float32)          # (n, L) queued at step 6
        self.elig = z_train["elig"][:, 5, :].astype(np.float32)
        self.minute = z_train["minute"].astype(np.float32)
        self.bw = bandwidth
        self.all = self.hits.sum(0) / np.maximum(self.elig.sum(0), 1)

    def tod(self, minutes):
        d = np.abs(minutes[:, None] - self.minute[None, :])
        d = np.minimum(d, 1440 - d)
        w = np.exp(-.5 * (d / self.bw) ** 2) + .02                          # (m, n)
        return (w @ self.hits) / np.maximum(w @ self.elig, 1e-6)           # (m, L)

    def features(self, z):
        """(n, 6, L, k) base features plus prior columns."""
        x, names = xy(z)
        n, F, L, _ = x.shape
        t = self.tod(z["minute"].astype(np.float32))
        a = np.broadcast_to(self.all[None, :], (n, L))
        extra = [a, t]
        for k in (-1, 1):
            extra.append(np.roll(t, k, axis=1))
        rank = np.argsort(np.argsort(-t, axis=1), axis=1).astype(np.float32)
        extra.append(np.minimum(rank, 50))
        e = np.stack(extra, -1)                                            # (n, L, 5)
        e = np.broadcast_to(e[:, None], (n, F, L, e.shape[-1]))
        return np.concatenate([x, e], -1), names + ["prior_all", "prior_tod", "prior_tod_m1", "prior_tod_p1", "prior_rank"]
