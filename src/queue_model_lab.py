"""Train per-panel queue models on generated windows; test on later months."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.queue_gen_lab import PANELS, load, persistence, window_iou
from src.queue_model import choose_set, context, expand_steps, official_arrays

TRAIN = ["2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12"]
TEST = ["2031-01", "2031-02"]


def subset(z, cond, months):
    sel = (z["cond"] == cond) & np.isin(np.array([d[:7] for d in z["date"]]), months)
    return {k: (v[sel] if v.ndim and len(v) == len(z["cond"]) else v) for k, v in z.items()}


def xy(z):
    ctx, names = context(z["speed"], z["flow"], z["occ"], z["cut"], z["cap"], z["minute"], z["dow"])
    x = expand_steps(ctx)
    return x, names + ["step"]


IDENTITY = ("link", "msin", "mcos", "minute", "dow")


def feature_mask(names, drop=()):
    return np.array([n not in drop for n in names])


def fit(z, keep_far=1.0, drop=(), **kw):
    """keep_far < 1 down-samples cells far from any queue and reweights them to stay unbiased."""
    x, names = xy(z)
    e = z["elig"]
    w = None
    if keep_far < 1:
        far = (x[..., names.index("dq12_up")] >= 10) & (x[..., names.index("dq12_dn")] >= 10) & ~(x[..., names.index("r_min")] <= 1.3)
        rng = np.random.default_rng(0)
        keep = ~far | (rng.random(far.shape) < keep_far)
        e = e & keep
        w = np.where(far, 1 / keep_far, 1.0)[e]
    m = HistGradientBoostingClassifier(max_iter=kw.get("iters", 300), learning_rate=kw.get("lr", .08),
                                       max_leaf_nodes=kw.get("leaves", 63), min_samples_leaf=50,
                                       l2_regularization=1.0, random_state=0)
    cols = feature_mask(names, drop)
    m.fit(x[e][:, cols], z["truth"][e], sample_weight=w)
    m.cols_ = cols
    return m, names


def predict(m, z):
    x, _ = xy(z)
    n, F, L, k = x.shape
    cols = getattr(m, "cols_", np.ones(k, bool))
    return m.predict_proba(x.reshape(-1, k)[:, cols])[:, 1].reshape(n, F, L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="queue_ongoing")
    ap.add_argument("--panels", nargs="+", default=PANELS)
    a = ap.parse_args()
    rows = []
    for panel in a.panels:
        z = load(panel)
        tr, te = subset(z, a.cond, TRAIN), subset(z, a.cond, TEST)
        m, _ = fit(tr)
        p = predict(m, te)
        res = {"panel": panel, "n_test": len(te["cond"])}
        o = official_arrays(panel, "train")
        o = {k: (v[o["cond"] == a.cond] if isinstance(v, np.ndarray) and len(v) == len(o["cond"]) else v) for k, v in o.items()}
        po = predict(m, o)
        if a.cond == "queue_ongoing":
            res["off_persist_f1.2"] = window_iou(persistence(o, 12, 1.2), o["truth"], o["elig"]).mean()
        res["off_gbm_t0.4"] = window_iou(choose_set(po, o["elig"], "threshold", .4), o["truth"], o["elig"]).mean()
        res["off_gbm_eiou"] = window_iou(choose_set(po, o["elig"], "eiou"), o["truth"], o["elig"]).mean()
        if a.cond == "queue_ongoing":
            res["persist_f1.0"] = window_iou(persistence(te, 12, 1.0), te["truth"], te["elig"]).mean()
            res["persist_f1.2"] = window_iou(persistence(te, 12, 1.2), te["truth"], te["elig"]).mean()
        for thr in (.3, .4, .5):
            res[f"gbm_t{thr}"] = window_iou(choose_set(p, te["elig"], "threshold", thr), te["truth"], te["elig"]).mean()
        res["gbm_eiou"] = window_iou(choose_set(p, te["elig"], "eiou"), te["truth"], te["elig"]).mean()
        rows.append(res)
        print(pd.Series(res).to_string(), flush=True)
    df = pd.DataFrame(rows).set_index("panel")
    print(df.round(4).to_string())
    print(df.mean().round(4).to_string())
