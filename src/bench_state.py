"""Compare same-day state reconstruction on released train truth."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.state import predict_day


def template_month(path: Path, month: str) -> pd.DataFrame:
    selected = []
    for chunk in pd.read_csv(path, usecols=["timestamp", "station_id", "link_id", "mask_regime"],
                             dtype=str, chunksize=250_000):
        piece = chunk[chunk.timestamp.str.startswith(month)].copy()
        if not piece.empty:
            selected.append(piece)
    return pd.concat(selected, ignore_index=True)


def evaluate(panel: str, month: str, weights: list[float], release: Path,
             scale_spatial: bool = False, flow_weight: float | None = None,
             min_quality: float = 0) -> pd.DataFrame:
    panel_dir = release / "corridors" / panel
    topology = pd.read_csv(panel_dir / "network" / "lwr_mainline_topology.csv")
    link_order = topology.sort_values("order_index").mainline_link_id.astype(str).tolist()
    lanes = pd.read_csv(panel_dir / "network" / "fd_parameters.csv").set_index("link_id").lanes
    spatial_scale = None
    if scale_spatial:
        fd = pd.read_csv(panel_dir / "network" / "fd_parameters.csv").set_index("link_id")
        spatial_scale = {"flow_vph": fd.lanes.to_dict(), "speed_kmh": fd.free_speed_kmh.to_dict()}
    template = template_month(release / "task1" / panel / "train" / "sample_submission_state.csv", month)
    template["date"] = template.timestamp.str[:10]
    truth_files = {p.name: p for p in (panel_dir / "train" / "mainline_states").glob("**/*.parquet")}
    masked_files = sorted((panel_dir / "train" / "mainline_states_masked").glob(f"**/*{month.replace('-', '_')}*.parquet"))
    if not masked_files or template.empty:
        raise ValueError(f"no targets or masked days for {panel} {month}")
    stats = {(regime, weight): np.zeros(4, dtype=float) for regime in ("R1", "R2", "R3") for weight in weights}
    for masked_path in masked_files:
        date = masked_path.stem.removeprefix("synthetic_mainline_").replace("_", "-")
        targets = template[template.date == date].drop(columns="date")
        if targets.empty:
            continue
        masked = pd.read_parquet(masked_path, columns=["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "pct_observed"])
        truth_path = truth_files[masked_path.name]
        truth = pd.read_parquet(truth_path, columns=["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"])
        truth = truth[truth.is_score_eligible.astype(bool)].drop(columns="is_score_eligible")
        truth = targets.merge(truth, on=["timestamp", "station_id", "link_id"], how="left", validate="one_to_one")
        regime = targets.mask_regime.iloc[0]
        true_speed = truth.speed_kmh.to_numpy(dtype=float)
        true_flow = truth.flow_vph.to_numpy(dtype=float)
        lane = truth.link_id.map(lanes).fillna(1.0).to_numpy(dtype=float)
        valid = np.isfinite(true_speed) & np.isfinite(true_flow)
        for weight in weights:
            blend = weight if flow_weight is None else {"speed_kmh": weight, "flow_vph": flow_weight}
            pred = predict_day(masked, targets, link_order, temporal_weight=blend,
                               spatial_scale=spatial_scale, min_pct_observed=min_quality)
            ps = pred.speed_kmh.to_numpy(dtype=float)
            pf = pred.flow_vph.to_numpy(dtype=float)
            missing = valid & (~np.isfinite(ps) | ~np.isfinite(pf))
            ds = np.where(np.isfinite(ps), ps, 0)[valid] - true_speed[valid]
            dq = (np.where(np.isfinite(pf), pf, 0)[valid] - true_flow[valid]) / lane[valid]
            stats[regime, weight] += [np.sum(ds * ds), np.sum(dq * dq), valid.sum(), missing.sum()]
        print(date, regime, len(targets), flush=True)
    rows = []
    for (regime, weight), (ss, qq, n, missing) in stats.items():
        if n == 0:
            continue
        speed_rmse, flow_rmse = np.sqrt(ss / n), np.sqrt(qq / n)
        score = .54 * max(0, 1 - speed_rmse / 25) + .46 * max(0, 1 - flow_rmse / 600)
        rows.append(dict(panel=panel, month=month, regime=regime, temporal_weight=weight,
                         scale_spatial=scale_spatial,
                         flow_weight=weight if flow_weight is None else flow_weight,
                         min_quality=min_quality,
                         n=int(n), missing=int(missing), speed_rmse=speed_rmse,
                         flow_rmse_per_lane=flow_rmse, S_state=score))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=[0, .25, .5, .75, 1])
    parser.add_argument("--release", type=Path, default=Path("data/release"))
    parser.add_argument("--scale-spatial", action="store_true")
    parser.add_argument("--flow-weight", type=float)
    parser.add_argument("--min-quality", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.panel, args.month, args.weights, args.release,
                      args.scale_spatial, args.flow_weight, args.min_quality)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
