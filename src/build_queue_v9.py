"""V9 queue: per-panel, per-condition GBM trained on organizer-like train windows.

Predicts every official validation/private window from its released 60-minute
history only, choosing each window's cells by expected-IoU top-k.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import src.queue_gen_lab as gen
from src.queue_gen_lab import PANELS
from src.queue_model import choose_set, official_arrays
from src.queue_model_lab import fit, predict

MONTHS = ["2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12", "2031-01", "2031-02"]


def subset(z, cond):
    sel = z["cond"] == cond
    return {k: (v[sel] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(z["cond"]) else v) for k, v in z.items()}


def build(output: Path, models: Path):
    models.mkdir(parents=True, exist_ok=True)
    frames = []
    for panel in PANELS:
        z = gen.load(panel)
        for cond in ("queue_onset", "queue_ongoing"):
            m, names = fit(subset(z, cond))
            joblib.dump({"model": m, "features": names}, models / f"{panel}-{cond}.joblib")
            for split in ("validation", "private"):
                o = official_arrays(panel, split)
                idx = np.flatnonzero(o["cond"] == cond)
                oo = {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(o["cond"]) else v) for k, v in o.items()}
                p = predict(m, oo)
                pred = choose_set(p, oo["elig"], "eiou")
                for j, i in enumerate(idx):
                    s, li = o["cell_step"][i], o["cell_link"][i]
                    wid = o["window_id"][i]
                    t = official_template(panel, split, wid)
                    t["queue_pred"] = pred[j, s, li].astype(int)
                    frames.append(t)
            print(panel, cond, "done", flush=True)
    out = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print("rows", len(out), "positive", int(out.queue_pred.sum()), flush=True)


_templates = {}


def official_template(panel, split, wid):
    key = (panel, split)
    if key not in _templates:
        t = pd.read_csv(Path("data/release/task2") / panel / split / "sample_submission_queue.csv",
                        usecols=["window_id", "timestamp", "link_id"], dtype=str)
        _templates[key] = {k: g.reset_index(drop=True) for k, g in t.groupby("window_id")}
    return _templates[key][wid].copy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("outputs/v9/queue.csv"))
    ap.add_argument("--models", type=Path, default=Path("outputs/v9/models"))
    ap.add_argument("--windows", type=Path, help="generated window directory (default: queue_gen_lab.QWIN)")
    a = ap.parse_args()
    if a.windows:
        import src.queue_gen_lab as g
        g.QWIN = a.windows
    build(a.output, a.models)
