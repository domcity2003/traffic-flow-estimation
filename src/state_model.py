"""Learned Task 1 residual model on top of the v6 reconstruction.

Features for a target cell come from the same masked day only (plus a train
time-of-day profile), so the model applies unchanged to validation and private
days, including the 90-minute all-link gaps after each Task 2 window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.state import interpolate_grid

PROFILE = Path("outputs/lab/profiles")
FIELDS = ("speed_kmh", "flow_vph")
TREND = False  # gap-trend features (set by the caller that trained with them)
GAPX = False  # neighbouring links' last/next readings (set by the caller that trained with them)


def neighbours_in_time(g):
    """Last visible value before and first after each cell, with step distances (capped)."""
    T, L = g.shape
    prev_v = np.full((T, L), np.nan, np.float32); prev_d = np.full((T, L), 300, np.float32)
    next_v = np.full((T, L), np.nan, np.float32); next_d = np.full((T, L), 300, np.float32)
    cur = np.full(L, np.nan, np.float32); cd = np.full(L, 300, np.float32)
    for t in range(T):
        prev_v[t], prev_d[t] = cur, np.minimum(cd, 300)
        ok = np.isfinite(g[t]); cur = np.where(ok, g[t], cur); cd = np.where(ok, 1, cd + 1)
    cur = np.full(L, np.nan, np.float32); cd = np.full(L, 300, np.float32)
    for t in range(T - 1, -1, -1):
        next_v[t], next_d[t] = cur, np.minimum(cd, 300)
        ok = np.isfinite(g[t]); cur = np.where(ok, g[t], cur); cd = np.where(ok, 1, cd + 1)
    return prev_v, prev_d, next_v, next_d


def neighbours_in_space(g):
    """Nearest visible value upstream (lower index) and downstream at the same step."""
    T, L = g.shape
    up_v = np.full((T, L), np.nan, np.float32); up_d = np.full((T, L), 99, np.float32)
    dn_v = np.full((T, L), np.nan, np.float32); dn_d = np.full((T, L), 99, np.float32)
    cur = np.full(T, np.nan, np.float32); cd = np.full(T, 99, np.float32)
    for l in range(L):
        up_v[:, l], up_d[:, l] = cur, np.minimum(cd, 99)
        ok = np.isfinite(g[:, l]); cur = np.where(ok, g[:, l], cur); cd = np.where(ok, 1, cd + 1)
    cur = np.full(T, np.nan, np.float32); cd = np.full(T, 99, np.float32)
    for l in range(L - 1, -1, -1):
        dn_v[:, l], dn_d[:, l] = cur, np.minimum(cd, 99)
        ok = np.isfinite(g[:, l]); cur = np.where(ok, g[:, l], cur); cd = np.where(ok, 1, cd + 1)
    return up_v, up_d, dn_v, dn_d


def profile(panel: str, weekend: bool) -> dict[str, np.ndarray]:
    z = np.load(PROFILE / f"{panel}.npz")
    k = "we" if weekend else "wd"
    return {f: z[f"{f}_{k}"] for f in FIELDS}


def build_profiles(panel: str, order: list[str], months: list[str], root=Path("data/release")):
    """Median (tod x link) profiles of eligible train observations, weekday and weekend."""
    acc = {(f, k): [] for f in FIELDS for k in ("wd", "we")}
    for m in months:
        for p in sorted((root / "corridors" / panel / "train/mainline_states").glob(f"**/*{m.replace('-', '_')}*.parquet")):
            d = pd.read_parquet(p, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"])
            ts = pd.to_datetime(d.timestamp.iloc[0], utc=True)
            k = "we" if ts.dayofweek >= 5 else "wd"
            e = d.is_score_eligible.astype(bool)
            for f in FIELDS:
                g = d.assign(v=d[f].where(e)).pivot(index="timestamp", columns="link_id", values="v").reindex(columns=order)
                if len(g) == 288:
                    acc[f, k].append(g.to_numpy(np.float32))
    PROFILE.mkdir(parents=True, exist_ok=True)
    out = {}
    for (f, k), arrs in acc.items():
        with np.errstate(all="ignore"):
            prof = np.nanmedian(np.stack(arrs), 0)
        out[f"{f}_{k}"] = interpolate_grid(interpolate_grid(prof, 0), 1).astype(np.float32)
    np.savez_compressed(PROFILE / f"{panel}.npz", **out)


def features(ctx, grids: dict, base: dict, gap: np.ndarray, weekend: bool, cells=None):
    """Feature matrix for cells (ti, li); returns (X, names)."""
    ti, li = cells
    T, L = grids["speed_kmh"].shape
    prof = profile(ctx.panel, weekend)
    cols, names = [], []

    def add(name, arr):
        names.append(name)
        cols.append(np.asarray(arr, np.float32)[ti, li] if np.ndim(arr) == 2 else np.asarray(arr, np.float32))

    tod = np.repeat(np.arange(T, dtype=np.float32)[:, None], L, 1)
    link = np.repeat(np.arange(L, dtype=np.float32)[None, :], T, 0)
    for f in FIELDS:
        g = grids[f].astype(np.float32)
        s = "v" if f == "speed_kmh" else "q"
        pv, pd_, nv, nd = neighbours_in_time(g)
        uv, ud, dv, dd = neighbours_in_space(g)
        tmp = interpolate_grid(g, 0)
        add(f"{s}_base", base[f]); add(f"{s}_prof", prof[f])
        add(f"{s}_prev", pv); add(f"{s}_prevd", pd_); add(f"{s}_next", nv); add(f"{s}_nextd", nd)
        add(f"{s}_up", uv); add(f"{s}_upd", ud); add(f"{s}_dn", dv); add(f"{s}_dnd", dd)
        add(f"{s}_tmp", tmp)
        add(f"{s}_tmp_m1", np.pad(tmp, ((0, 0), (1, 0)), constant_values=np.nan)[:, :-1])
        add(f"{s}_tmp_p1", np.pad(tmp, ((0, 0), (0, 1)), constant_values=np.nan)[:, 1:])
        # Offsets of the nearest visible readings from the profile at their own
        # times, so a gap can be bridged along the typical daily shape.
        rows = np.arange(T)[:, None]
        at_prev = prof[f][np.clip(rows - pd_.astype(int), 0, T - 1), np.arange(L)[None, :]]
        at_next = prof[f][np.clip(rows + nd.astype(int), 0, T - 1), np.arange(L)[None, :]]
        add(f"{s}_prev_anom", pv - at_prev); add(f"{s}_next_anom", nv - at_next)
        if GAPX:
            # Inside an all-link gap the spatial neighbours are blank too; give
            # the model what the nearby links read just before and after it.
            for k in (-2, -1, 1, 2):
                for name, a in (("prev", pv - at_prev), ("next", nv - at_next)):
                    sh = np.full_like(a, np.nan)
                    if k < 0:
                        sh[:, -k:] = a[:, :k]
                    else:
                        sh[:, :-k] = a[:, k:]
                    add(f"{s}_{name}_anom_l{k:+d}", sh)
        if TREND:
            # Trend just outside the gap: interpolated value 3 steps beyond the
            # nearest readings, and where the cell sits between them.
            cols_ = np.arange(L)[None, :]
            before = tmp[np.clip(rows - pd_.astype(int) - 3, 0, T - 1), cols_]
            after = tmp[np.clip(rows + nd.astype(int) + 3, 0, T - 1), cols_]
            add(f"{s}_trend_prev", pv - before); add(f"{s}_trend_next", after - nv)
            add(f"{s}_gap_pos", pd_ / np.maximum(pd_ + nd, 1))
    add("gap", np.repeat(gap[:, None], L, 1).astype(np.float32))
    add("tod", tod); add("link", link)
    add("lanes", np.repeat(ctx.lanes[None, :], T, 0)); add("vfree", np.repeat(ctx.fd_free[None, :], T, 0))
    X = np.column_stack(cols)
    return X, names
