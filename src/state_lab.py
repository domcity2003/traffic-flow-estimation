"""Evaluate state predictors on train months with S_state and faithful S_physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.physics_local import ROOT, Accumulator, Panel, score_day, targets_for
from src.state import interpolate_grid
from src.state_v4 import complete_grid

PANELS = [p["corridor_id"] for p in json.loads((ROOT / "config/corridors.json").read_text())["panels"]]
WEIGHTS = json.loads(Path("config/state_weights.json").read_text())
LOWRANK = json.loads(Path("config/state_v4.json").read_text())


def bounded(amount, cap_scale=1.0):
    """v3 interpolation plus a bounded share of the v4 low-rank correction (v6: 0.75)."""
    def fn(ctx, grids, **_):
        return _bounded(ctx, grids, amount, cap_scale)
    return fn


def v6(ctx, grids, **_):
    """Replica of the accepted v6 state: interpolation plus bounded low-rank correction."""
    return _bounded(ctx, grids, .75, 1.0)


def _bounded(ctx, grids, amount, cap_scale):
    out = {}
    for field, g in grids.items():
        w = WEIGHTS[ctx.panel][field]
        t, s = interpolate_grid(g, 0), interpolate_grid(g, 1)
        base = np.where(np.isfinite(t) & np.isfinite(s), w * t + (1 - w) * s, np.where(np.isfinite(t), t, s))
        rule = LOWRANK[ctx.panel][field]
        cand = (1 - rule["blend"]) * base + rule["blend"] * complete_grid(g, rule["rank"])
        cap = cap_scale * (5.0 if field == "speed_kmh" else 50.0 * ctx.lanes[None, :])
        out[field] = np.maximum(base + amount * np.clip(cand - base, -cap, cap), 0)
    return out["speed_kmh"], out["flow_vph"]


def window_gaps(panel: str) -> dict[str, list[int]]:
    """Organizer-like window forecast starts (minute of day) per date, from generated windows."""
    path = Path("outputs/lab/qwin2") / f"{panel}.npz"
    out = {}
    if not path.exists():  # D12_I405 has no Task 2 windows, hence no gaps
        return out
    z = np.load(path)
    for d, m in zip(z["date"], z["minute"]):
        out.setdefault(str(d), []).append(int(m))
    return out


def iter_days(ctx: Panel, month: str, gaps: bool = False):
    """Yield (date, regime, times, masked grids, eligible obs grids, target grid) for a train month."""
    targets = targets_for(ctx.panel, "train")
    targets = targets[targets.date.str.startswith(month)]
    truth_files = {p.name: p for p in (ctx.dir / "train/mainline_states").glob("**/*.parquet")}
    for path in sorted((ctx.dir / "train/mainline_states_masked").glob(f"**/*{month.replace('-', '_')}*.parquet")):
        date = path.stem.removeprefix("synthetic_mainline_").replace("_", "-")
        t = targets[targets.date == date]
        if t.empty:
            continue
        masked = pd.read_parquet(path)
        masked["timestamp"] = pd.to_datetime(masked.timestamp, utc=True)
        times = pd.DatetimeIndex(sorted(masked.timestamp.unique()))
        truth = pd.read_parquet(truth_files[path.name])
        truth["timestamp"] = pd.to_datetime(truth.timestamp, utc=True)
        truth = truth[truth.is_score_eligible.astype(bool)]
        grids = {f: ctx.grid(masked, f, times) for f in ("speed_kmh", "flow_vph", "occupancy")}
        grids["pct_observed"] = ctx.grid(masked, "pct_observed", times)
        gap = np.zeros(len(times), bool)
        if gaps:
            # Validation/private blank every link from T+5 through T+90 after each queue window.
            minutes = times.hour * 60 + times.minute
            for start in window_gaps(ctx.panel).get(date, []):
                gap |= (minutes >= start) & (minutes < start + 90)
            for f in ("speed_kmh", "flow_vph", "occupancy"):
                grids[f][gap] = np.nan
        grids["gap"] = gap
        obs_v, obs_q = ctx.grid(truth, "speed_kmh", times), ctx.grid(truth, "flow_vph", times)
        target = np.zeros(obs_v.shape, bool)
        ti = times.get_indexer(pd.to_datetime(t.timestamp, utc=True))
        li = pd.Index(ctx.order).get_indexer(t.link_id)
        target[ti, li] = True
        yield date, t.mask_regime.iloc[0], times, grids, obs_v, obs_q, target


def evaluate(predictors: dict, months: list[str], panels: list[str], gaps: bool = False, split_gap: bool = False) -> pd.DataFrame:
    rows = []
    for panel in panels:
        ctx = Panel(panel)
        accs = {name: Accumulator() for name in predictors}
        if split_gap:
            pass
        for month in months:
            for date, regime, times, grids, obs_v, obs_q, target in iter_days(ctx, month, gaps):
                g = {"speed_kmh": grids["speed_kmh"], "flow_vph": grids["flow_vph"]}
                for name, fn in predictors.items():
                    pv, pq = fn(ctx, g, allg=grids, times=times, date=date)
                    score_day(ctx, accs[name], regime, date, times, obs_v, obs_q, target, pv, pq)
                    if split_gap and grids["gap"].any():
                        for part, sel in (("gap", grids["gap"]), ("nogap", ~grids["gap"])):
                            t2 = target & sel[:, None]
                            score_day(ctx, accs.setdefault(f"{name}@{part}", Accumulator()), regime, date, times, obs_v, obs_q, t2, pv, pq)
        for name, acc in accs.items():
            s = acc.scores().assign(panel=panel, model=name)
            rows.append(s)
        summary = pd.concat(rows).query("panel == @panel").groupby("model")[["S_state", "S_FD", "S_LWR", "S_physics"]].mean()
        print(panel, "\n", summary.round(4).to_string(), flush=True)
    return pd.concat(rows, ignore_index=True)


def report(df: pd.DataFrame) -> pd.DataFrame:
    per_panel = df.groupby(["model", "panel"])[["S_state", "S_FD", "S_LWR", "S_physics"]].mean()
    macro = per_panel.groupby("model").mean()
    macro["composite_part"] = .35 * macro.S_state + .15 * macro.S_physics
    return macro


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=["2031-02"])
    ap.add_argument("--panels", nargs="+", default=PANELS)
    ap.add_argument("--output", type=Path, default=Path("outputs/lab/v6.csv"))
    a = ap.parse_args()
    df = evaluate({"v6": v6}, a.months, a.panels)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output, index=False)
    print(report(df).round(5).to_string())
