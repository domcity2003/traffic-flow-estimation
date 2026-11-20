"""Queue proxy on the organizer's own train windows.

The 80 official train windows were drawn by the organizer's selection rule, the
same rule as validation and private. Their horizons are visible in the unmasked
train observations, so the proxy label is eligible observed speed <= 0.6 x the
fd_parameters free speed. The real labels use the underlying latent state instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/release")
PANELS = ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]


class Windows:
    """Official windows of one split with histories, templates and (train) proxy truth."""

    def __init__(self, panel: str, split: str = "train", cut_factor: float = 0.6):
        self.panel, self.split = panel, split
        p = ROOT / "task2" / panel / split
        self.index = pd.read_csv(p / "window_index.csv", dtype=str)
        h = pd.read_parquet(p / "window_history.parquet")
        h["timestamp"] = pd.to_datetime(h.timestamp, utc=True)
        self.history = {k: g for k, g in h.groupby("window_id")}
        t = pd.read_csv(p / "sample_submission_queue.csv", usecols=["window_id", "timestamp", "link_id"], dtype=str)
        self.template = {k: g.reset_index(drop=True) for k, g in t.groupby("window_id")}
        net = ROOT / "corridors" / panel / "network"
        topo = pd.read_csv(net / "lwr_mainline_topology.csv")
        self.order = topo.sort_values("order_index").mainline_link_id.astype(str).tolist()
        self.fd = pd.read_csv(net / "fd_parameters.csv").drop_duplicates("link_id").set_index("link_id").reindex(self.order)
        self.cut = (cut_factor * self.fd.free_speed_kmh).to_dict()
        self.truth = self._truth() if split == "train" else None

    def _truth(self):
        files = {p.stem.removeprefix("synthetic_mainline_").replace("_", "-"): p
                 for p in (ROOT / "corridors" / self.panel / "train/mainline_states").glob("**/*.parquet")}
        out = {}
        for w in self.index.itertuples(index=False):
            d = pd.read_parquet(files[w.date], columns=["timestamp", "link_id", "speed_kmh", "is_score_eligible"])
            t = self.template[w.window_id]
            m = t.merge(d, on=["timestamp", "link_id"], how="left")
            elig = m.is_score_eligible.fillna(False).astype(bool).to_numpy()
            q = (m.speed_kmh.to_numpy(float) <= m.link_id.map(self.cut).to_numpy(float))
            out[w.window_id] = (elig, q & elig)
        return out

    def grid(self, wid, field="speed_kmh"):
        """History grid (steps x links in order), NaN where not eligible."""
        h = self.history[wid]
        v = pd.to_numeric(h[field], errors="coerce").where(h.is_score_eligible.astype(bool))
        return h.assign(v=v).pivot(index="timestamp", columns="link_id", values="v").reindex(columns=self.order)


def iou(pred, elig, truth):
    p = pred.astype(bool) & elig
    inter, union = (p & truth).sum(), (p | truth).sum()
    return 1.0 if union == 0 else inter / union


def evaluate(policies: dict, panels=PANELS, cut_factor=0.6) -> pd.DataFrame:
    """policies: name -> fn(ws, window_row, template) -> 0/1 array aligned to template."""
    rows = []
    for panel in panels:
        ws = Windows(panel, "train", cut_factor)
        for w in ws.index.itertuples(index=False):
            t = ws.template[w.window_id]
            elig, truth = ws.truth[w.window_id]
            for name, fn in policies.items():
                pred = fn(ws, w, t)
                if pred is None:
                    continue
                rows.append(dict(panel=panel, window_id=w.window_id, condition=w.condition, model=name,
                                 iou=iou(np.asarray(pred), elig, truth), n_true=int(truth.sum()),
                                 n_pred=int(np.asarray(pred).sum())))
    return pd.DataFrame(rows)


def summary(df):
    pc = df.groupby(["model", "condition", "panel"]).iou.mean().groupby(["model", "condition"]).mean().unstack()
    return pc
