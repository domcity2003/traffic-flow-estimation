"""V14 ongoing: average of the unbagged v12 blend and 3 bootstrap bags of it.

Pseudo-window ongoing IoU vs single: April +0.012, March +0.006, Feb +0.002,
June official -0.006. Onset is kept from v13.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import src.queue_gen_lab as gen
import src.queue_model as qm
from src.build_queue_v9 import official_template, subset
from src.queue_eval import take
from src.queue_gen_lab import PANELS
from src.queue_model import choose_set, official_arrays
from src.queue_model_lab import feature_mask, fit, predict, xy

COND, B = "queue_ongoing", 3
OUT = Path("outputs/v14")


def pooled(tr, seed):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for p in PANELS:
        z = tr[p]
        n = len(z["cond"])
        zz = take(z, rng.integers(0, n, n) if seed else np.arange(n))
        x, names = xy(zz)
        far = (x[..., names.index("dq12_up")] >= 10) & (x[..., names.index("dq12_dn")] >= 10) & ~(x[..., names.index("r_min")] <= 1.3)
        keep = zz["elig"] & (~far | (rng.random(far.shape) < .25))
        X.append(x[keep]); Y.append(zz["truth"][keep])
    cols = feature_mask(names, ("link",))
    m = HistGradientBoostingClassifier(max_iter=500, learning_rate=.08, max_leaf_nodes=127, min_samples_leaf=100,
                                       l2_regularization=1.0, random_state=seed).fit(np.vstack(X)[:, cols], np.concatenate(Y))
    m.cols_ = cols
    return m


if __name__ == "__main__":
    gen.QWIN = Path("outputs/lab/qwin2")
    qm.RICH = True
    OUT.mkdir(parents=True, exist_ok=True)
    tr = {p: subset(gen.load(p), COND) for p in PANELS}
    pools = [pooled(tr, s) for s in range(B + 1)]
    frames = []
    for p in PANELS:
        n = len(tr[p]["cond"])
        pers = [fit(take(tr[p], np.random.default_rng(100 + s).integers(0, n, n) if s else np.arange(n)))[0] for s in range(B + 1)]
        for split in ("validation", "private"):
            o = official_arrays(p, split)
            idx = np.flatnonzero(o["cond"] == COND)
            oo = take(o, o["cond"] == COND)
            pr = .5 * np.mean([predict(m, oo) for m in pers], 0) + .5 * np.mean([predict(m, oo) for m in pools], 0)
            pred = choose_set(pr, oo["elig"], "eiou")
            for j, i in enumerate(idx):
                t = official_template(p, split, o["window_id"][i])
                t["queue_pred"] = pred[j, o["cell_step"][i], o["cell_link"][i]].astype(int)
                frames.append(t)
        print(p, "done", flush=True)
    q = pd.concat(frames, ignore_index=True)
    q.to_csv(OUT / "queue_ongoing.csv", index=False)
    print("ongoing rows", len(q), "positive", int(q.queue_pred.sum()))
