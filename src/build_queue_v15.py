"""V15 queue: lean both conditions toward the pooled (no link id) models.

Onset: 10 bootstrap bags, per-panel weight 0.2 (was 0.5). Ongoing: unbagged
model plus 8 bags, per-panel weight 0.35 (was 0.5). On March/April
pseudo-windows the pooled model transfers better; onset IoU +0.01 on both,
ongoing +0.002. Missed onsets are mostly the wrong one of two recurrent
bottlenecks, which per-link memorisation makes overconfident.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import src.queue_gen_lab as gen
import src.queue_model as qm
from src.build_queue_v9 import official_template, subset
from src.build_queue_v12 import onset_proba
from src.build_queue_v14 import pooled
from src.queue_eval import take
from src.queue_gen_lab import PANELS
from src.queue_model import choose_set, official_arrays
from src.queue_model_lab import feature_mask, fit, predict
from src.queue_onset_prior import OnsetPrior

OUT = Path("outputs/v15")
ONSET_B, ONSET_W = 10, .2
ONGOING_B, ONGOING_W = 8, .35


def write(cond, tr, proba, name):
    frames = []
    for p in PANELS:
        f = proba(p)
        for split in ("validation", "private"):
            o = official_arrays(p, split)
            idx = np.flatnonzero(o["cond"] == cond)
            oo = take(o, o["cond"] == cond)
            pred = choose_set(f(oo), oo["elig"], "eiou")
            for j, i in enumerate(idx):
                t = official_template(p, split, o["window_id"][i])
                t["queue_pred"] = pred[j, o["cell_step"][i], o["cell_link"][i]].astype(int)
                frames.append(t)
        print(cond, p, "done", flush=True)
    q = pd.concat(frames, ignore_index=True)
    q.to_csv(OUT / name, index=False)
    print(name, "rows", len(q), "positive", int(q.queue_pred.sum()), flush=True)


def onset():
    cond = "queue_onset"
    tr = {p: subset(gen.load(p), cond) for p in PANELS}
    pri = {p: OnsetPrior(tr[p]) for p in PANELS}
    feats = {p: pri[p].features(tr[p]) for p in PANELS}
    cols = feature_mask(feats[PANELS[0]][1], ("link",))
    pools = []
    for s in range(1, ONSET_B + 1):
        rng = np.random.default_rng(s)
        X, Y = [], []
        for p in PANELS:
            x, _ = feats[p]
            idx = rng.integers(0, len(x), len(x))
            e = tr[p]["elig"][idx]
            X.append(x[idx][e]); Y.append(tr[p]["truth"][idx][e])
        pools.append(HistGradientBoostingClassifier(max_iter=400, learning_rate=.06, max_leaf_nodes=63, min_samples_leaf=100,
                                                    l2_regularization=1.0, random_state=s).fit(np.vstack(X)[:, cols], np.concatenate(Y)))
    del feats

    def proba(p):
        n = len(tr[p]["cond"])
        pers = [fit(take(tr[p], np.random.default_rng(100 + s).integers(0, n, n)))[0] for s in range(1, ONSET_B + 1)]
        return lambda oo: (ONSET_W * np.mean([predict(m, oo) for m in pers], 0)
                           + (1 - ONSET_W) * np.mean([onset_proba(m, cols, pri[p], oo) for m in pools], 0))
    write(cond, tr, proba, "queue_onset.csv")


def ongoing():
    cond = "queue_ongoing"
    qm.RICH = True
    tr = {p: subset(gen.load(p), cond) for p in PANELS}
    pools = [pooled(tr, s) for s in range(ONGOING_B + 1)]

    def proba(p):
        n = len(tr[p]["cond"])
        pers = [fit(take(tr[p], np.random.default_rng(100 + s).integers(0, n, n) if s else np.arange(n)))[0]
                for s in range(ONGOING_B + 1)]
        return lambda oo: (ONGOING_W * np.mean([predict(m, oo) for m in pers], 0)
                           + (1 - ONGOING_W) * np.mean([predict(m, oo) for m in pools], 0))
    write(cond, tr, proba, "queue_ongoing.csv")


if __name__ == "__main__":
    import sys
    gen.QWIN = Path("outputs/lab/qwin2")
    OUT.mkdir(parents=True, exist_ok=True)
    {"onset": onset, "ongoing": ongoing}[sys.argv[1]]()
