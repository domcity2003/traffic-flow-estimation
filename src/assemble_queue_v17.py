"""Turn saved test probabilities (src/queue_probs_v17.py) into queue CSVs.

Onset: per-panel weight W on the mean of the chosen per-panel components, the
rest on the mean of the chosen pooled components; expected-IoU top-k. Ongoing:
per-panel weight 0.35, then expected-IoU top-k or a fixed threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_queue_v9 import official_template
from src.queue_eval import take
from src.queue_gen_lab import PANELS
from src.queue_model import choose_set, official_arrays

PROBS = Path("outputs/lab/probs17")


def assemble(cond, proba, decide, out):
    frames = []
    for p in PANELS:
        for split in ("validation", "private"):
            o = official_arrays(p, split)
            idx = np.flatnonzero(o["cond"] == cond)
            oo = take(o, o["cond"] == cond)
            z = np.load(PROBS / f"{cond}_{p}_{split}.npz")
            pred = decide(proba({k: z[k].astype(np.float32) for k in z.files}, p, split), oo["elig"])
            for j, i in enumerate(idx):
                t = official_template(p, split, o["window_id"][i])
                t["queue_pred"] = pred[j, o["cell_step"][i], o["cell_link"][i]].astype(int)
                frames.append(t)
    q = pd.concat(frames, ignore_index=True)
    q.to_csv(out, index=False)
    print(out, "rows", len(q), "positive", int(q.queue_pred.sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["onset", "ongoing"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--w", type=float, default=None)
    ap.add_argument("--per", nargs="+", default=["per_plain"])
    ap.add_argument("--pool", nargs="+", default=["pool_plain_60"])
    ap.add_argument("--thr", type=float, help="ongoing: fixed threshold instead of expected-IoU top-k")
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.cond == "onset":
        w = .2 if a.w is None else a.w
        proba = lambda z, *_: w * np.mean([z[k] for k in a.per], 0) + (1 - w) * np.mean([z[k] for k in a.pool], 0)
    else:
        w = .35 if a.w is None else a.w
        proba = lambda z, *_: w * z["per"] + (1 - w) * z["pool"]
    decide = (lambda p, e: choose_set(p, e, "threshold", a.thr)) if a.thr else (lambda p, e: choose_set(p, e, "eiou"))
    assemble("queue_" + a.cond, proba, decide, a.output)
