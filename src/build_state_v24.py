"""V24 state: v19's GBM state blended with the whole-day CNN residual (src/state_cnn.py).

  python -m src.build_state_v24 train SEED      # fit on every panel, July-February caches
  python -m src.build_state_v24 predict         # CNN state at every key row -> outputs/v24/state_cnn.parquet
  python -m src.build_state_v24 blend AG AN BASE OUT  # CNN weight AG inside all-link gaps, AN elsewhere; patched into BASE
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import state_cnn as sc
from src.build_state_v11 import patch
from src.physics_local import Panel
from src.state_lab import PANELS

ROOT = Path("data/release")
MONTHS = os.environ.get("ST_MONTHS", "2030-07,2030-08,2030-09,2030-10,2030-11,2030-12,2031-01,2031-02").split(",")
DIR = Path(os.environ.get("ST_DIR", "outputs/lab/stcnn_final"))
OUT = Path(os.environ.get("ST_OUT", "outputs/v24"))
EPOCHS, WIDTH = int(os.environ.get("ST_EPOCHS", "8")), int(os.environ.get("ST_WIDTH", "32"))


def train(seed):
    torch.set_num_threads(int(os.environ.get("ST_THREADS", "6")))  # leave the machine responsive
    DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    days = sc.load(PANELS, MONTHS)
    print("days", len(days), flush=True)
    net = sc.train(days, epochs=EPOCHS, width=WIDTH, seed=seed, batch=int(os.environ.get("ST_BATCH", "4")), lam=float(os.environ.get("ST_LAM", "0")), log=lambda s: print(s, round(time.time() - t0), flush=True))
    torch.save(net.state_dict(), DIR / f"seed{seed}.pt")
    print("saved seed", seed, flush=True)


def predict():
    nets = []
    for f in sorted(DIR.glob("seed*.pt")):
        sd = torch.load(f, weights_only=True)
        n = sc.Net(w=sd["inp.weight"].shape[0]); n.load_state_dict(sd); nets.append(n)
    print("seeds", len(nets), flush=True)
    key = pd.read_csv(ROOT / "submission_key.csv", dtype=str, usecols=["submission_id", "task", "split", "panel", "timestamp", "link_id"])
    key = key[key.task == "state"]
    key["date"] = key.timestamp.str[:10]
    out = []
    for panel in PANELS:
        pr = sc.profile(panel)
        for split in ("validation", "private"):
            ctx = Panel(panel, split)
            z = dict(np.load(sc.CACHE / f"test_{panel}_{split}.npz"))
            ks = key[(key.panel == panel) & (key.split == split)]
            for i, date in enumerate(z["date"]):
                t = ks[ks.date == str(date)]
                if t.empty:
                    continue
                path = next((ctx.dir / split / "mainline_states_masked").glob(f"**/*{str(date).replace('-', '_')}*.parquet"))
                times = pd.DatetimeIndex(sorted(pd.to_datetime(pd.read_parquet(path, columns=["timestamp"]).timestamp, utc=True).unique()))
                x, _, _ = sc.inputs(z, pr, i)
                r = np.mean([sc.predict(n, x) for n in nets], 0)
                pv, pq = sc.denorm(z, r, i)
                ti = times.get_indexer(pd.to_datetime(t.timestamp, utc=True))
                li = pd.Index(ctx.order).get_indexer(t.link_id)
                if (ti < 0).any() or (li < 0).any():
                    raise ValueError(f"{panel} {date}: target outside grid")
                out.append(pd.DataFrame({"submission_id": t.submission_id.astype(int).to_numpy(),
                                         "speed_kmh": np.maximum(pv[ti, li], 6.0), "flow_vph": np.maximum(pq[ti, li], 0.0),
                                         "gap": z["gap"][i][ti]}))
            print(panel, split, "done", flush=True)
    res = pd.concat(out, ignore_index=True).sort_values("submission_id")
    if res.submission_id.duplicated().any() or len(res) != len(key):
        raise ValueError(f"state coverage {len(res)} vs key {len(key)}")
    if not np.isfinite(res[["speed_kmh", "flow_vph"]].to_numpy()).all():
        raise ValueError("non-finite state prediction")
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT / "state_cnn.parquet")
    print("rows", len(res))


def blend(a_gap, a_nogap, base, output):
    g = pd.read_parquet("outputs/v19/state.parquet").sort_values("submission_id").reset_index(drop=True)
    c = pd.read_parquet(OUT / "state_cnn.parquet").sort_values("submission_id").reset_index(drop=True)
    if not (g.submission_id.to_numpy() == c.submission_id.to_numpy()).all():
        raise ValueError("state id mismatch")
    s = g.copy()
    a = np.where(c.gap.to_numpy(), a_gap, a_nogap)
    s["speed_kmh"] = np.maximum((1 - a) * g.speed_kmh + a * c.speed_kmh, 6.0)
    s["flow_vph"] = np.maximum((1 - a) * g.flow_vph + a * c.flow_vph, 0.0)
    s[["submission_id", "speed_kmh", "flow_vph"]].to_parquet(OUT / "state.parquet")
    patch(Path(base), OUT / "state.parquet", Path(output))


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(int(sys.argv[2]))
    elif sys.argv[1] == "predict":
        predict()
    else:
        blend(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5])
