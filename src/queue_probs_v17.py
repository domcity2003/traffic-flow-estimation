"""Save test-window queue probabilities per component, so decision variants are cheap.

Onset components (B=10 bags each): per-panel plain, per-panel rich, pooled plain
with 60- and 30-minute priors, pooled rich with the 60-minute prior. Ongoing: the
v15 per-panel and pooled bags. Written to outputs/lab/probs17/{cond}_{panel}_{split}.npz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import src.queue_gen_lab as gen
import src.queue_model as qm
from src.build_queue_v9 import subset
from src.build_queue_v14 import pooled as pooled_ongoing
from src.build_queue_v16 import pooled as pooled_onset, pooled_proba
from src.queue_eval import take
from src.queue_gen_lab import PANELS
from src.queue_model import official_arrays
from src.queue_model_lab import fit, predict
from src.queue_onset_prior import OnsetPrior

OUT = Path("outputs/lab/probs17")
B = 10


def per_panel(z, bags, full=False):
    n = len(z["cond"])
    return [fit(take(z, np.random.default_rng(100 + s).integers(0, n, n) if s else np.arange(n)))[0]
            for s in (range(bags + 1) if full else range(1, bags + 1))]


def onset():
    cond = "queue_onset"
    comps = {}
    for rich in (False, True):
        qm.RICH = rich
        tr = {p: subset(gen.load(p), cond) for p in PANELS}
        for bw in ((60.0, 30.0) if not rich else (60.0,)):
            pri = {p: OnsetPrior(tr[p], bw) for p in PANELS}
            cols, ms = pooled_onset(tr, pri)
            comps[f"pool_{'rich' if rich else 'plain'}_{int(bw)}"] = (rich, pri, cols, ms)
            print("pooled", rich, bw, flush=True)
        comps[f"per_{'rich' if rich else 'plain'}"] = (rich, {p: per_panel(tr[p], B) for p in PANELS})
        print("per-panel", rich, flush=True)
    for p in PANELS:
        for split in ("validation", "private"):
            out = {}
            for name, c in comps.items():
                qm.RICH = c[0]
                o = official_arrays(p, split)
                oo = take(o, o["cond"] == cond)
                if name.startswith("pool"):
                    out[name] = pooled_proba(c[1][p], c[2], c[3], oo).astype(np.float16)
                else:
                    out[name] = np.mean([predict(m, oo) for m in c[1][p]], 0).astype(np.float16)
            np.savez_compressed(OUT / f"{cond}_{p}_{split}.npz", **out)
        print(cond, p, flush=True)


def ongoing():
    cond = "queue_ongoing"
    qm.RICH = True
    tr = {p: subset(gen.load(p), cond) for p in PANELS}
    pools = [pooled_ongoing(tr, s) for s in range(9)]
    for p in PANELS:
        pers = per_panel(tr[p], 8, full=True)
        for split in ("validation", "private"):
            o = official_arrays(p, split)
            oo = take(o, o["cond"] == cond)
            np.savez_compressed(OUT / f"{cond}_{p}_{split}.npz",
                                per=np.mean([predict(m, oo) for m in pers], 0).astype(np.float16),
                                pool=np.mean([predict(m, oo) for m in pools], 0).astype(np.float16))
        print(cond, p, flush=True)


if __name__ == "__main__":
    gen.QWIN = Path("outputs/lab/qwin2")
    OUT.mkdir(parents=True, exist_ok=True)
    {"onset": onset, "ongoing": ongoing}[sys.argv[1]]()
