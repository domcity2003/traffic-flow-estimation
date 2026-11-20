"""V13 onset: bag of 5 bootstrap (per-panel + pooled-prior) onset models.

Bagging improved onset IoU on February, March and April pseudo-windows
(+0.009, +0.005, +0.011). Ongoing predictions are kept from v12.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import src.queue_gen_lab as gen
from src.build_queue_v9 import official_template, subset
from src.build_queue_v12 import onset_proba
from src.queue_eval import take
from src.queue_gen_lab import PANELS
from src.queue_model import choose_set, official_arrays
from src.queue_model_lab import feature_mask, fit, predict
from src.queue_onset_prior import OnsetPrior

COND, B = "queue_onset", 5
OUT = Path("outputs/v13")

if __name__ == "__main__":
    gen.QWIN = Path("outputs/lab/qwin2")
    OUT.mkdir(parents=True, exist_ok=True)
    tr = {p: subset(gen.load(p), COND) for p in PANELS}
    pri = {p: OnsetPrior(tr[p]) for p in PANELS}
    feats = {p: pri[p].features(tr[p]) for p in PANELS}
    cols = feature_mask(feats[PANELS[0]][1], ("link",))
    pools = []
    for s in range(1, B + 1):
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
    frames = []
    for p in PANELS:
        n = len(tr[p]["cond"])
        pers = [fit(take(tr[p], np.random.default_rng(100 + s).integers(0, n, n)))[0] for s in range(1, B + 1)]
        for split in ("validation", "private"):
            o = official_arrays(p, split)
            idx = np.flatnonzero(o["cond"] == COND)
            oo = take(o, o["cond"] == COND)
            pr = .5 * np.mean([predict(m, oo) for m in pers], 0) + .5 * np.mean([onset_proba(m, cols, pri[p], oo) for m in pools], 0)
            pred = choose_set(pr, oo["elig"], "eiou")
            for j, i in enumerate(idx):
                t = official_template(p, split, o["window_id"][i])
                t["queue_pred"] = pred[j, o["cell_step"][i], o["cell_link"][i]].astype(int)
                frames.append(t)
        print(p, "done", flush=True)
    q = pd.concat(frames, ignore_index=True)
    q.to_csv(OUT / "queue_onset.csv", index=False)
    print("onset rows", len(q), "positive", int(q.queue_pred.sum()))
