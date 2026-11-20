"""V16 onset: pooled models on two onset-prior bandwidths (30 and 60 min).

Same per-panel/pooled blend as v15 (per-panel weight 0.2), but the pooled half
averages models whose time-of-day onset prior uses a 30-minute and a 60-minute
kernel. Pseudo-window onset IoU vs v15's 60-minute prior: June +0.005, Feb 0.000,
March +0.003, April +0.017. Ongoing predictions are kept from v15.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import src.build_queue_v15 as v15
import src.queue_gen_lab as gen
from src.build_queue_v9 import subset
from src.queue_eval import take
from src.queue_gen_lab import PANELS
from src.queue_model_lab import feature_mask, fit, predict
from src.queue_onset_prior import OnsetPrior

OUT = Path("outputs/v16")
B, W, BANDWIDTHS = 10, .2, (30.0, 60.0)


def pooled(tr, pri):
    feats = {p: pri[p].features(tr[p]) for p in PANELS}
    cols = feature_mask(feats[PANELS[0]][1], ("link",))
    models = []
    for s in range(1, B + 1):
        rng = np.random.default_rng(s)
        X, Y = [], []
        for p in PANELS:
            x, _ = feats[p]
            idx = rng.integers(0, len(x), len(x))
            e = tr[p]["elig"][idx]
            X.append(x[idx][e]); Y.append(tr[p]["truth"][idx][e])
        models.append(HistGradientBoostingClassifier(max_iter=400, learning_rate=.06, max_leaf_nodes=63, min_samples_leaf=100,
                                                     l2_regularization=1.0, random_state=s).fit(np.vstack(X)[:, cols], np.concatenate(Y)))
    return cols, models


def pooled_proba(pri, cols, models, oo):
    x, _ = pri.features(oo)
    n, F, L, k = x.shape
    x = x.reshape(-1, k)[:, cols]
    return np.mean([m.predict_proba(x)[:, 1].reshape(n, F, L) for m in models], 0)


if __name__ == "__main__":
    gen.QWIN = Path("outputs/lab/qwin2")
    OUT.mkdir(parents=True, exist_ok=True)
    v15.OUT = OUT
    cond = "queue_onset"
    tr = {p: subset(gen.load(p), cond) for p in PANELS}
    pools = []
    for bw in BANDWIDTHS:
        pri = {p: OnsetPrior(tr[p], bw) for p in PANELS}
        pools.append((pri, *pooled(tr, pri)))
        print("pooled bandwidth", bw, "done", flush=True)

    def proba(p):
        n = len(tr[p]["cond"])
        pers = [fit(take(tr[p], np.random.default_rng(100 + s).integers(0, n, n)))[0] for s in range(1, B + 1)]
        return lambda oo: (W * np.mean([predict(m, oo) for m in pers], 0)
                           + (1 - W) * np.mean([pooled_proba(pri[p], cols, ms, oo) for pri, cols, ms in pools], 0))
    v15.write(cond, tr, proba, "queue_onset.csv")
