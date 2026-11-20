"""Faithful local Task 1 + Task 3 scoring on train days.

The organizer's Task 3 evaluator uses withheld boundary flows "projected onto
the released observations, so a participant who reproduces what was published
satisfies the conservation identity exactly". On train the unmasked
observations are released, so that projection can be rebuilt: the conservation
right-hand side of every scored transition equals the observed change in
accumulation. The residual is then dN_pred - dN_obs, and

    S_LWR = 1 - sum|dN_pred - dN_obs| / sum|dN_obs|

over transitions with both endpoints eligible, an observable link and valid
ramp evidence. S_FD follows score_task3.score_panel exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "organizer-code" / "src"))
from task3.score_task3 import network_parameters, load_ramp_flows  # noqa: E402

ROOT = Path("data/release")
CACHE = Path("outputs/lab/cache")


class Panel:
    """Static network context for one panel on one split, in topology order."""

    def __init__(self, panel: str, split: str = "train"):
        self.panel, self.split = panel, split
        self.dir = ROOT / "corridors" / panel
        topo = pd.read_csv(self.dir / "network/lwr_mainline_topology.csv", dtype=str)
        self.order = topo.sort_values("order_index", key=lambda s: s.astype(int)).mainline_link_id.astype(str).tolist()
        fd = pd.read_csv(self.dir / "network/fd_parameters.csv").drop_duplicates("link_id").set_index("link_id")
        self.lanes = fd.lanes.reindex(self.order).to_numpy(float)
        self.fd_free = fd.free_speed_kmh.reindex(self.order).to_numpy(float)
        self.fd_cap = fd.capacity_vph.reindex(self.order).to_numpy(float)
        p = network_parameters(self.dir).set_index("link_id").reindex(self.order)
        self.p_len = p.length_km.to_numpy(float)
        self.p_lanes = p.lanes.clip(lower=1).to_numpy(float)
        self.p_free = p.free_speed_kmh.to_numpy(float)
        self.p_cap = p.capacity_vph.to_numpy(float)
        self.p_kcrit = p.critical_density.to_numpy(float)
        self.p_kjam = p.k_jam.to_numpy(float)
        # Observable links: every internal mainline neighbour is also scored.
        topo_ids = set(topo.link_id.astype(str))
        present = set(self.order)
        obs = {}
        for row in topo.itertuples(index=False):
            inc = {v.strip() for v in str(row.incoming_link_ids).split(";") if v.strip()}
            out = {v.strip() for v in str(row.outgoing_link_ids).split(";") if v.strip()}
            obs[str(row.link_id)] = (inc & topo_ids) <= present and (out & topo_ids) <= present
        self.observable = np.array([obs.get(l, False) for l in self.order])
        self.on_att = np.array([l in set(topo.loc[topo.on_ramp_link_ids.fillna("").str.len() > 0, "link_id"]) for l in self.order])
        self.off_att = np.array([l in set(topo.loc[topo.off_ramp_link_ids.fillna("").str.len() > 0, "link_id"]) for l in self.order])
        self._ramps = None

    @property
    def ramps(self) -> pd.DataFrame:
        if self._ramps is None:
            cache = CACHE / f"ramps-{self.panel}-{self.split}.parquet"
            if cache.exists():
                self._ramps = pd.read_parquet(cache)
            else:
                r = load_ramp_flows(self.dir, self.split)
                r["date"] = r.timestamp.dt.strftime("%Y-%m-%d")
                cache.parent.mkdir(parents=True, exist_ok=True)
                r.to_parquet(cache)
                self._ramps = r
        return self._ramps

    def ramp_valid(self, date: str, times: pd.DatetimeIndex) -> np.ndarray:
        """Boolean grid: transition starting at (t, link) has valid ramp evidence."""
        r = self.ramps[self.ramps.date == date]
        li = pd.Index(self.order).get_indexer(r.link_id)
        ti = times.get_indexer(r.timestamp)
        ok = (li >= 0) & (ti >= 0)
        on = np.zeros((len(times), len(self.order)), bool)
        off = np.zeros_like(on)
        on[ti[ok], li[ok]] = r.on_ramp_valid.to_numpy()[ok] > 0
        off[ti[ok], li[ok]] = r.off_ramp_valid.to_numpy()[ok] > 0
        on |= ~self.on_att[None, :]
        off |= ~self.off_att[None, :]
        return on & off

    def grid(self, frame: pd.DataFrame, field: str, times: pd.DatetimeIndex) -> np.ndarray:
        g = frame.pivot(index="timestamp", columns="link_id", values=field)
        g.index = pd.to_datetime(g.index, utc=True)
        return g.reindex(index=times, columns=self.order).to_numpy(float, copy=True)


def targets_for(panel: str, split: str = "train") -> pd.DataFrame:
    """Template target cells for a split, cached as parquet."""
    cache = CACHE / f"targets-{panel}-{split}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    t = pd.read_csv(ROOT / "task1" / panel / split / "sample_submission_state.csv",
                    usecols=["timestamp", "station_id", "link_id", "mask_regime"], dtype=str)
    t["date"] = t.timestamp.str[:10]
    cache.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(cache)
    return t


class Accumulator:
    """Per-regime sums for S_state and S_physics, pooled across days."""

    def __init__(self):
        self.s = {}

    def add(self, regime, **kw):
        d = self.s.setdefault(regime, dict.fromkeys(
            ["ss", "qq", "n", "fd_err", "fd_tot", "lwr_num", "lwr_den", "sub_n", "sub_low"], 0.0))
        for k, v in kw.items():
            d[k] += float(v)

    def scores(self) -> pd.DataFrame:
        rows = []
        for r, d in sorted(self.s.items()):
            sr, qr = np.sqrt(d["ss"] / max(d["n"], 1)), np.sqrt(d["qq"] / max(d["n"], 1))
            state = .54 * max(0, 1 - sr / 25) + .46 * max(0, 1 - qr / 600)
            fd = max(0.0, 1 - d["fd_err"] / (d["fd_tot"] + 1e-9))
            if d["sub_n"] and d["sub_low"] / d["sub_n"] > .2:
                fd = 0.0
            lwr = max(0.0, 1 - min(1.0, d["lwr_num"] / (d["lwr_den"] + 1e-9)))
            rows.append(dict(regime=r, speed_rmse=sr, flow_rmse_lane=qr, S_state=state,
                             S_FD=fd, S_LWR=lwr, E_LWR=d["lwr_num"] / (d["lwr_den"] + 1e-9),
                             S_physics=(fd + 2 * lwr) / 3))
        return pd.DataFrame(rows)


def score_day(ctx: Panel, acc: Accumulator, regime: str, date: str, times: pd.DatetimeIndex,
              obs_v: np.ndarray, obs_q: np.ndarray, target: np.ndarray,
              pred_v: np.ndarray, pred_q: np.ndarray) -> None:
    """Add one day's Task 1 and faithful Task 3 terms.

    obs_* are eligible observations (NaN where not eligible); target marks the
    template cells; pred_* are full grids whose target cells are scored.
    """
    y_ok = target & np.isfinite(obs_v) & np.isfinite(obs_q)
    ds = pred_v[y_ok] - obs_v[y_ok]
    dq = (pred_q[y_ok] - obs_q[y_ok]) / np.broadcast_to(ctx.lanes[None, :], target.shape)[y_ok]
    acc.add(regime, ss=np.sum(ds * ds), qq=np.sum(dq * dq), n=y_ok.sum())

    v = np.where(target, pred_v, obs_v)
    q = np.where(target, pred_q, obs_q)
    have = np.isfinite(v) & np.isfinite(q) & np.isfinite(ctx.p_len)[None, :]
    k = q / np.clip(v, 1.0, None)
    # FD term, per lane, over every scored cell.
    ql, kl = q / ctx.p_lanes, k / ctx.p_lanes
    capl, kcl, kjl = ctx.p_cap / ctx.p_lanes, ctx.p_kcrit / ctx.p_lanes, ctx.p_kjam / ctx.p_lanes
    qfd = np.where(kl <= kcl, ctx.p_free * kl, capl / np.maximum(kjl - kcl, 1e-9) * np.maximum(kjl - kl, 0))
    acc.add(regime, fd_err=np.abs(ql - qfd)[have].sum(), fd_tot=np.abs(ql)[have].sum(),
            sub_n=(target & have).sum(), sub_low=(target & have & (q < 50)).sum())
    # Conservation: residual against the projected organizer flux is dN_pred - dN_obs.
    n_pred = k * ctx.p_len
    n_obs = obs_q / np.clip(obs_v, 1.0, None) * ctx.p_len
    valid = have[:-1] & have[1:] & ctx.observable[None, :] & ctx.ramp_valid(date, times)[:-1]
    dp, do = np.diff(n_pred, axis=0), np.diff(n_obs, axis=0)
    acc.add(regime, lwr_num=np.abs(dp - do)[valid].sum(), lwr_den=np.abs(do)[valid].sum())
