"""V11 ongoing queue: equal blend of per-panel and pooled (all-panel, no link id) GBMs.

The pooled model transfers better to the less congested validation month
(March pseudo-windows 0.795 -> 0.821 at the 50/50 blend); the per-panel model
is stronger on the official June windows. Onset stays as v9.
"""
from __future__ import annotations

from pathlib import Path

import joblib  # models are produced locally by this script
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import src.queue_gen_lab as gen
from src.queue_gen_lab import PANELS
from src.queue_model import choose_set, official_arrays
from src.queue_model_lab import feature_mask, fit, predict, xy
from src.build_queue_v9 import official_template, subset

COND = "queue_ongoing"
OUT = Path("outputs/v11")


def pooled_model():
    X, Y = [], []
    rng = np.random.default_rng(0)
    for p in PANELS:
        z = subset(gen.load(p), COND)
        x, names = xy(z)
        far = (x[..., names.index("dq12_up")] >= 10) & (x[..., names.index("dq12_dn")] >= 10) & ~(x[..., names.index("r_min")] <= 1.3)
        keep = z["elig"] & (~far | (rng.random(far.shape) < .25))
        X.append(x[keep]); Y.append(z["truth"][keep])
    cols = feature_mask(names, ("link",))
    m = HistGradientBoostingClassifier(max_iter=500, learning_rate=.08, max_leaf_nodes=127, min_samples_leaf=100,
                                       l2_regularization=1.0, random_state=0)
    # Unweighted, exactly as evaluated: far cells kept at 25% (probabilities lean
    # slightly high there, which the expected-IoU selection tolerated in testing).
    m.fit(np.vstack(X)[:, cols], np.concatenate(Y))
    m.cols_ = cols
    return m


if __name__ == "__main__":
    gen.QWIN = Path("outputs/lab/qwin2")
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    pooled = pooled_model()
    joblib.dump(pooled, OUT / "models" / "pooled-ongoing.joblib")
    frames = []
    for panel in PANELS:
        per, _ = fit(subset(gen.load(panel), COND))
        joblib.dump(per, OUT / "models" / f"{panel}-ongoing.joblib")
        for split in ("validation", "private"):
            o = official_arrays(panel, split)
            idx = np.flatnonzero(o["cond"] == COND)
            oo = {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(o["cond"]) else v) for k, v in o.items()}
            pred = choose_set(.5 * predict(per, oo) + .5 * predict(pooled, oo), oo["elig"], "eiou")
            for j, i in enumerate(idx):
                t = official_template(panel, split, o["window_id"][i])
                t["queue_pred"] = pred[j, o["cell_step"][i], o["cell_link"][i]].astype(int)
                frames.append(t)
        print(panel, "done", flush=True)
    q = pd.concat(frames, ignore_index=True)
    q.to_csv(OUT / "queue_ongoing.csv", index=False)
    print("ongoing rows", len(q), "positive", int(q.queue_pred.sum()))
