"""V12 queue: v11-style ongoing blend with richer history features, plus an onset
blend of the per-panel model and a pooled model on transferable link-onset priors.

All models are fitted on every generated train month (organizer selection rules).
"""
from __future__ import annotations

from pathlib import Path

import joblib  # models are produced locally by this script
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import src.queue_gen_lab as gen
import src.queue_model as qm
from src.build_queue_v9 import official_template, subset
from src.queue_gen_lab import PANELS
from src.queue_model import choose_set, official_arrays
from src.queue_model_lab import feature_mask, fit, predict, xy
from src.queue_onset_prior import OnsetPrior

OUT = Path("outputs/v12")


def pooled_ongoing():
    X, Y = [], []
    rng = np.random.default_rng(0)
    for p in PANELS:
        z = subset(gen.load(p), "queue_ongoing")
        x, names = xy(z)
        far = (x[..., names.index("dq12_up")] >= 10) & (x[..., names.index("dq12_dn")] >= 10) & ~(x[..., names.index("r_min")] <= 1.3)
        keep = z["elig"] & (~far | (rng.random(far.shape) < .25))
        X.append(x[keep]); Y.append(z["truth"][keep])
    cols = feature_mask(names, ("link",))
    m = HistGradientBoostingClassifier(max_iter=500, learning_rate=.08, max_leaf_nodes=127, min_samples_leaf=100,
                                       l2_regularization=1.0, random_state=0).fit(np.vstack(X)[:, cols], np.concatenate(Y))
    m.cols_ = cols
    return m


def pooled_onset(priors):
    X, Y = [], []
    for p in PANELS:
        z = subset(gen.load(p), "queue_onset")
        x, names = priors[p].features(z)
        X.append(x[z["elig"]]); Y.append(z["truth"][z["elig"]])
    cols = feature_mask(names, ("link",))
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=.06, max_leaf_nodes=63, min_samples_leaf=100,
                                       l2_regularization=1.0, random_state=0).fit(np.vstack(X)[:, cols], np.concatenate(Y))
    return m, cols


def onset_proba(m, cols, prior, z):
    x, _ = prior.features(z)
    n, F, L, k = x.shape
    return m.predict_proba(x.reshape(-1, k)[:, cols])[:, 1].reshape(n, F, L)


if __name__ == "__main__":
    gen.QWIN = Path("outputs/lab/qwin2")
    OUT.mkdir(parents=True, exist_ok=True)
    priors = {p: OnsetPrior(subset(gen.load(p), "queue_onset")) for p in PANELS}
    on_pool, on_cols = pooled_onset(priors)
    qm.RICH = True
    og_pool = pooled_ongoing()
    frames = []
    for panel in PANELS:
        z = gen.load(panel)
        qm.RICH = False
        on_per, _ = fit(subset(z, "queue_onset"))
        qm.RICH = True
        og_per, _ = fit(subset(z, "queue_ongoing"))
        for split in ("validation", "private"):
            o = official_arrays(panel, split)
            for cond in ("queue_onset", "queue_ongoing"):
                idx = np.flatnonzero(o["cond"] == cond)
                oo = {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(o["cond"]) else v) for k, v in o.items()}
                if cond == "queue_onset":
                    qm.RICH = False
                    p = .5 * predict(on_per, oo) + .5 * onset_proba(on_pool, on_cols, priors[panel], oo)
                else:
                    qm.RICH = True
                    p = .5 * predict(og_per, oo) + .5 * predict(og_pool, oo)
                pred = choose_set(p, oo["elig"], "eiou")
                for j, i in enumerate(idx):
                    t = official_template(panel, split, o["window_id"][i])
                    t["queue_pred"] = pred[j, o["cell_step"][i], o["cell_link"][i]].astype(int)
                    frames.append(t)
        print(panel, "done", flush=True)
    q = pd.concat(frames, ignore_index=True)
    q.to_csv(OUT / "queue.csv", index=False)
    print("rows", len(q), "positive", int(q.queue_pred.sum()))
