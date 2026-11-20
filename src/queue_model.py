"""Learned per-cell queue forecasts from history-only features.

Features are computed from a (windows, 12, links) history array, so the same
code serves generated train windows and official validation/private windows.
"""
from __future__ import annotations

import numpy as np

H, F = 12, 6


def ffill_steps(a):
    """Forward-fill along the step axis; also return steps since last valid."""
    out = np.full(a.shape, np.nan, np.float32)
    age = np.full(a.shape, 99, np.float32)
    cur = np.full(a[:, 0].shape, np.nan, np.float32)
    cage = np.full(a[:, 0].shape, 99, np.float32)
    for i in range(a.shape[1]):
        v = a[:, i]
        ok = np.isfinite(v)
        cur = np.where(ok, v, cur)
        cage = np.where(ok, 0, cage + 1)
        out[:, i], age[:, i] = cur, cage
    return out, age


def shift(x, k):
    """x[..., l] -> value at link l+k (NaN padded)."""
    out = np.full(x.shape, np.nan, np.float32)
    if k > 0:
        out[..., :-k] = x[..., k:]
    elif k < 0:
        out[..., -k:] = x[..., :k]
    else:
        out[:] = x
    return out


def nearest(q, direction):
    """Distance in links to nearest True in `direction` (+1 higher index, -1 lower), capped 30."""
    n, L = q.shape
    d = np.full((n, L), 30, np.float32)
    rng = range(L) if direction < 0 else range(L - 1, -1, -1)
    cur = np.full(n, 30, np.float32)
    for l in rng:
        cur = np.where(q[:, l], 0, np.minimum(cur + 1, 30))
        d[:, l] = cur
    return d


RICH = False  # adds full per-step history and past queue-edge distances


def context(speed, flow, occ, cut, cap, minute, dow):
    """Per (window, link) context features; returns (n, L, k) array and names."""
    n, _, L = speed.shape
    r = speed / cut[None, None, :]
    rf, age = ffill_steps(r)
    last = rf[:, -1]
    q_hist = np.where(np.isfinite(r), r <= 1.0, False)
    qlast = np.nan_to_num(last, nan=9) <= 1.0
    qlast12 = np.nan_to_num(last, nan=9) <= 1.2
    nq = q_hist.sum(2).astype(np.float32)  # queued links per step
    f = {}
    f["r_last"] = last
    f["age"] = age[:, -1]
    f["r_l3"] = rf[:, -3]
    f["r_l6"] = rf[:, -6]
    f["r_first"] = rf[:, 0]
    with np.errstate(all="ignore"):
        f["r_mean"] = np.nanmean(r, 1)
        f["r_min"] = np.nanmin(np.where(np.isfinite(r), r, np.inf), 1)
        f["r_max"] = np.nanmax(np.where(np.isfinite(r), r, -np.inf), 1)
    f["r_min"][~np.isfinite(f["r_min"])] = np.nan
    f["r_max"][~np.isfinite(f["r_max"])] = np.nan
    f["q_frac"] = q_hist.sum(1) / np.maximum(np.isfinite(r).sum(1), 1)
    f["n_valid"] = np.isfinite(r).sum(1).astype(np.float32)
    for k in (-3, -2, -1, 1, 2, 3):
        f[f"r_last_{k:+d}"] = shift(last, k)
        f[f"r_l6_{k:+d}"] = shift(rf[:, -6], k)
    if RICH:
        for k in (2, 4, 5, 7, 8, 9, 10, 11):
            f[f"r_l{k}"] = rf[:, -k]
        for k in (4, 8):
            qk = np.nan_to_num(rf[:, -k], nan=9) <= 1.0
            f[f"dq_up_l{k}"] = nearest(qk, -1)
            f[f"dq_dn_l{k}"] = nearest(qk, +1)
    f["dq_up"] = nearest(qlast, -1)
    f["dq_dn"] = nearest(qlast, +1)
    f["dq12_up"] = nearest(qlast12, -1)
    f["dq12_dn"] = nearest(qlast12, +1)
    f["nq_last"] = np.repeat(nq[:, -1:], L, 1)
    f["nq_trend"] = np.repeat((nq[:, -1] - nq[:, -6])[:, None], L, 1)
    f["nq_trend2"] = np.repeat((nq[:, -1] - nq[:, 0])[:, None], L, 1)
    ff, _ = ffill_steps(flow / cap[None, None, :])
    f["f_last"] = ff[:, -1]
    f["f_l6"] = ff[:, -6]
    of, _ = ffill_steps(occ)
    f["o_last"] = of[:, -1]
    f["o_l6"] = of[:, -6]
    f["link"] = np.repeat(np.arange(L, dtype=np.float32)[None, :], n, 0)
    f["msin"] = np.repeat(np.sin(2 * np.pi * minute / 1440)[:, None], L, 1).astype(np.float32)
    f["mcos"] = np.repeat(np.cos(2 * np.pi * minute / 1440)[:, None], L, 1).astype(np.float32)
    f["minute"] = np.repeat(minute[:, None].astype(np.float32), L, 1)
    f["dow"] = np.repeat(dow[:, None].astype(np.float32), L, 1)
    names = list(f)
    return np.stack([f[k].astype(np.float32) for k in names], -1), names


def expand_steps(ctx):
    """(n, L, k) -> (n, F, L, k+1) with horizon step appended."""
    n, L, k = ctx.shape
    x = np.broadcast_to(ctx[:, None], (n, F, L, k))
    s = np.broadcast_to(np.arange(1, F + 1, dtype=np.float32)[None, :, None, None], (n, F, L, 1))
    return np.concatenate([x, s], -1)


def choose_set(p, elig, mode="threshold", thr=0.5):
    """Binary decision per window from probabilities (n, F, L)."""
    if mode == "threshold":
        return (p >= thr) & elig
    # Expected-IoU maximizing top-k per window: max_k sum_top / (k + sum_rest)
    out = np.zeros(p.shape, bool)
    for i in range(p.shape[0]):
        pi = np.where(elig[i], p[i], 0).ravel()
        order = np.argsort(-pi)
        cs = np.cumsum(pi[order])
        total = cs[-1]
        k = np.arange(1, len(pi) + 1)
        score = cs / (k + total - cs)
        best = int(np.argmax(score)) + 1
        flat = np.zeros(len(pi), bool)
        flat[order[:best]] = True
        out[i] = flat.reshape(p[i].shape) & elig[i]
    return out


def official_arrays(panel: str, split: str):
    """Official windows of a split as arrays aligned with generated windows.

    Returns dict with speed/flow/occ (n, 12, L), minute/dow of forecast start,
    cond, window_id, template (per-window frames) and, on train, proxy truth.
    """
    import pandas as pd
    from src.queue_lab import Windows
    ws = Windows(panel, split)
    L = len(ws.order)
    pos = {l: i for i, l in enumerate(ws.order)}
    out = {k: [] for k in ("speed", "flow", "occ", "minute", "dow", "date", "cond", "window_id", "truth", "elig", "cell_step", "cell_link")}
    for w in ws.index.itertuples(index=False):
        h = ws.history[w.window_id]
        steps = sorted(h.timestamp.unique())
        if len(steps) != H:
            raise ValueError(f"{w.window_id}: expected {H} history steps, got {len(steps)}")
        for key, field in (("speed", "speed_kmh"), ("flow", "flow_vph"), ("occ", "occupancy")):
            out[key].append(ws.grid(w.window_id, field).reindex(index=steps).to_numpy(np.float32))
        start = pd.Timestamp(w.forecast_start)
        out["minute"].append(start.hour * 60 + start.minute)
        out["dow"].append(start.dayofweek)
        out["date"].append(str(start.date()))
        out["cond"].append(w.condition)
        out["window_id"].append(w.window_id)
        t = ws.template[w.window_id]
        ts = pd.to_datetime(t.timestamp, utc=True)
        s = ((ts - start).dt.total_seconds() // 300).astype(int).to_numpy()
        if s.min() < 0 or s.max() >= F:
            raise ValueError(f"{w.window_id}: template outside horizon")
        li = t.link_id.map(pos).to_numpy()
        out["cell_step"].append(s)
        out["cell_link"].append(li)
        tr = np.zeros((F, L), bool)
        el = np.zeros((F, L), bool)
        if ws.truth is not None:
            e, q = ws.truth[w.window_id]
            el[s, li] = e
            tr[s, li] = q
        else:
            el[s, li] = True
        out["truth"].append(tr)
        out["elig"].append(el)
    res = {k: np.array(v) if k not in ("cell_step", "cell_link") else v for k, v in out.items()}
    res["cut"] = np.array([ws.cut[l] for l in ws.order], np.float32)
    res["cap"] = ws.fd.capacity_vph.to_numpy(np.float32)
    res["order"] = np.array(ws.order)
    return res
