"""Evaluate queue policies on generated organizer-like windows (array form)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

QWIN = Path("outputs/lab/qwin2")
PANELS = ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]


def load(panel):
    z = np.load(QWIN / f"{panel}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def window_iou(pred, truth, elig):
    p = pred & elig
    inter = (p & truth).sum((1, 2))
    union = (p | truth).sum((1, 2))
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def last_valid(speed, lookback):
    """Latest eligible speed per link within the last `lookback` steps (NaN if none)."""
    s = speed[:, -lookback:, :]
    out = np.full(s[:, 0, :].shape, np.nan, np.float32)
    for i in range(s.shape[1]):
        v = s[:, i, :]
        out = np.where(np.isfinite(v), v, out)
    return out


def persistence(z, lookback=12, factor=1.0):
    lv = last_valid(z["speed"], lookback)
    q = lv <= z["cut"][None, :] * factor
    return np.repeat(q[:, None, :], 6, axis=1)


def evaluate(policies, panels=PANELS, condition="queue_ongoing", months=None):
    rows = []
    for panel in panels:
        z = load(panel)
        sel = z["cond"] == condition
        if months:
            sel &= np.isin(np.array([d[:7] for d in z["date"]]), months)
        zz = {k: (v[sel] if v.ndim and len(v) == len(z["cond"]) else v) for k, v in z.items()}
        for name, fn in policies.items():
            pred = fn(zz, panel)
            iou = window_iou(pred, zz["truth"], zz["elig"])
            rows.append(dict(model=name, panel=panel, iou=iou.mean(), n=len(iou)))
    df = pd.DataFrame(rows)
    return df.pivot(index="model", columns="panel", values="iou").assign(mean=lambda d: d.mean(1)).sort_values("mean")


if __name__ == "__main__":
    pol = {f"lb{lb}_f{f}": (lambda lb, f: lambda z, p: persistence(z, lb, f))(lb, f)
           for lb in (1, 3, 12) for f in (1.0, 1.1, 1.2, 1.3)}
    print(evaluate(pol).round(4).to_string())
