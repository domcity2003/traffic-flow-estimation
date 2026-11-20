"""V11 state: v6 reconstruction plus the learned residual, for validation and private.

Writes a submission_id-indexed state table so the full CSV can be patched in
key order. Gap flags mark timestamps with no visible link, which on scored
splits are the Task 2 horizon plus its 60-minute buffer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib  # models are produced locally by src.state_model_lab
import numpy as np
import pandas as pd

from src.physics_local import Panel
from src.state_lab import PANELS, v6
from src.state_model import features

ROOT = Path("data/release")


def build(models: Path, output: Path, shrink: float = 1.0, gap_shrink: float | None = None):
    key = pd.read_csv(ROOT / "submission_key.csv", dtype=str,
                      usecols=["submission_id", "task", "split", "panel", "timestamp", "link_id", "mask_regime"])
    key = key[key.task == "state"]
    key["date"] = key.timestamp.str[:10]
    out = []
    for panel in PANELS:
        m = joblib.load(models / f"{panel}.joblib")
        kp = key[key.panel == panel]
        for split in ("validation", "private"):
            ctx = Panel(panel, split)
            ks = kp[kp.split == split]
            for path in sorted((ctx.dir / split / "mainline_states_masked").glob("**/*.parquet")):
                date = path.stem.removeprefix("synthetic_mainline_").replace("_", "-")
                t = ks[ks.date == date]
                if t.empty:
                    continue
                masked = pd.read_parquet(path)
                masked["timestamp"] = pd.to_datetime(masked.timestamp, utc=True)
                times = pd.DatetimeIndex(sorted(masked.timestamp.unique()))
                grids = {f: ctx.grid(masked, f, times) for f in ("speed_kmh", "flow_vph")}
                grids["gap"] = ~np.isfinite(grids["speed_kmh"]).any(1)
                bv, bq = v6(ctx, {f: grids[f] for f in ("speed_kmh", "flow_vph")})
                ti = times.get_indexer(pd.to_datetime(t.timestamp, utc=True))
                li = pd.Index(ctx.order).get_indexer(t.link_id)
                if (ti < 0).any() or (li < 0).any():
                    raise ValueError(f"{panel} {date}: target outside grid")
                x, names = features(ctx, grids, {"speed_kmh": bv, "flow_vph": bq}, grids["gap"],
                                    times[0].dayofweek >= 5, (ti, li))
                if names != m["features"]:
                    raise ValueError("feature mismatch")
                k = np.where(grids["gap"][ti], shrink if gap_shrink is None else gap_shrink, shrink)
                sv = np.maximum(bv[ti, li] + k * m["speed_kmh"].predict(x), 6.0)  # lowest eligible speed seen in train
                sq = np.maximum(bq[ti, li] + k * m["flow_vph"].predict(x), 0.0)
                out.append(pd.DataFrame({"submission_id": t.submission_id.astype(int).to_numpy(),
                                         "speed_kmh": sv, "flow_vph": sq}))
            print(panel, split, "done", flush=True)
    res = pd.concat(out, ignore_index=True).sort_values("submission_id")
    if res.submission_id.duplicated().any() or len(res) != len(key):
        raise ValueError(f"state coverage {len(res)} vs key {len(key)}")
    if not np.isfinite(res[["speed_kmh", "flow_vph"]].to_numpy()).all():
        raise ValueError("non-finite state prediction")
    output.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(output)
    print("rows", len(res))


def patch(base: Path, state: Path, output: Path):
    """Replace state speed/flow in a full submission, keyed by submission_id."""
    s = pd.read_parquet(state)
    sid = s.submission_id.to_numpy()
    sv, sq = s.speed_kmh.to_numpy().tolist(), s.flow_vph.to_numpy().tolist()
    pos = {int(i): j for j, i in enumerate(sid)} if not (np.diff(sid) == 1).all() else None
    first = int(sid[0])
    n = 0
    with open(base) as f, open(output, "w") as o:
        o.write(f.readline())
        for line in f:
            p = line.split(",", 4)
            if p[1] == "state":
                i = int(p[0])
                j = pos[i] if pos is not None else i - first
                if sid[j] != i:
                    raise ValueError(f"state id mismatch at {i}")
                line = f"{i},state,{sv[j]!r},{sq[j]!r},{p[4]}"
                n += 1
            o.write(line)
    if n != len(sid):
        raise ValueError(f"patched {n} of {len(sid)} state rows")
    print("patched", n, "state rows into", output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=Path("outputs/lab/state_models"))
    ap.add_argument("--state", type=Path, default=Path("outputs/v11/state.parquet"))
    ap.add_argument("--base", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--shrink", type=float, default=1.0)
    ap.add_argument("--gap-shrink", type=float, help="residual scale inside all-link gaps (default: --shrink)")
    a = ap.parse_args()
    if not a.state.exists():
        build(a.models, a.state, a.shrink, a.gap_shrink)
    if a.base:
        patch(a.base, a.state, a.output)
