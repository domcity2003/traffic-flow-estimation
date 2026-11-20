"""Score queue predictors on four out-of-fit sets.

june_official: organizer's 40+40 train windows (unmasked history).
train_feb / val_march / priv_april: pseudo-windows from the Task 1 masked layer
of each month (evaluation only).
Predictors are fn(panel, z, cond) -> probability array (n, 6, L).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import src.queue_gen_lab as gen
from src.queue_gen_lab import PANELS, window_iou
from src.queue_model import choose_set, official_arrays

SETS = {"train_feb": "outputs/lab/qtrain_masked", "val_march": "outputs/lab/qval_masked", "priv_april": "outputs/lab/qpriv_masked"}


def take(z, sel):
    return {k: (v[sel] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(sel) else v) for k, v in z.items()}


def sets_for(panel, cond):
    o = official_arrays(panel, "train")
    out = {"june_official": take(o, o["cond"] == cond)}
    for name, d in SETS.items():
        z = np.load(Path(d) / f"{panel}.npz")
        z = {k: z[k] for k in z.files}
        out[name] = take(z, z["cond"] == cond)
    return out


def evaluate(predictors: dict, cond: str, panels=PANELS) -> pd.DataFrame:
    rows = []
    for panel in panels:
        sets = sets_for(panel, cond)
        for name, fn in predictors.items():
            for sname, z in sets.items():
                if len(z["cond"]) == 0:
                    continue
                p = fn(panel, z, cond)
                rows.append(dict(panel=panel, model=name, set=sname, n=len(z["cond"]),
                                 iou=window_iou(choose_set(p, z["elig"], "eiou"), z["truth"], z["elig"]).mean()))
        print(panel, "done", flush=True)
    d = pd.DataFrame(rows)
    print(d.pivot_table(index="model", columns="set", values="iou").round(4).to_string())
    return d
