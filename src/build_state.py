"""Write Task 1 candidate predictions for the two scored splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.state import predict_day


OUTPUT_COLUMNS = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]


def build(release: Path, output: Path, config: dict[str, dict[str, float]],
          prediction_rules: dict | None = None) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    first = True
    total = 0
    for panel in sorted(config):
        panel_dir = release / "corridors" / panel
        topology = pd.read_csv(panel_dir / "network" / "lwr_mainline_topology.csv")
        link_order = topology.sort_values("order_index").mainline_link_id.astype(str).tolist()
        for split in ("validation", "private"):
            template = pd.read_csv(release / "task1" / panel / split / "sample_submission_state.csv",
                                   usecols=["panel", "timestamp", "station_id", "link_id", "mask_regime"],
                                   dtype=str)
            template["date"] = template.timestamp.str[:10]
            targets_by_date = {date: group.drop(columns="date") for date, group in template.groupby("date", sort=False)}
            paths = sorted((panel_dir / split / "mainline_states_masked").glob("**/*.parquet"))
            seen = set()
            count = 0
            for path in paths:
                date = path.stem.removeprefix("synthetic_mainline_").replace("_", "-")
                targets = targets_by_date.get(date)
                if targets is None or targets.empty:
                    continue
                if date in seen:
                    raise ValueError(f"multiple masked files for {panel} {split} {date}")
                seen.add(date)
                masked = pd.read_parquet(path, columns=["timestamp", "station_id", "link_id",
                                                        "speed_kmh", "flow_vph", "pct_observed"])
                if prediction_rules is None:
                    pred = predict_day(masked, targets, link_order, temporal_weight=config[panel])
                else:
                    from src.state_v4 import predict_day as predict_v4
                    pred = predict_v4(masked, targets, link_order, temporal_weight=config[panel],
                                      rules=prediction_rules.get(panel, {}))
                vals = pred[["speed_kmh", "flow_vph"]].to_numpy(dtype=float)
                if not np.isfinite(vals).all() or (vals < 0).any():
                    raise ValueError(f"invalid state predictions for {panel} {split} {date}")
                pred[OUTPUT_COLUMNS].to_csv(output, mode="w" if first else "a", header=first, index=False)
                first = False
                count += len(pred)
            if seen != set(targets_by_date) or count != len(template):
                raise ValueError(f"state target coverage mismatch for {panel} {split}")
            total += count
            print(panel, split, f"{count:,} rows", flush=True)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("data/release"))
    parser.add_argument("--config", type=Path, default=Path("config/state_weights.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v1/state.csv"))
    args = parser.parse_args()
    print(f"Wrote {build(args.release, args.output, json.loads(args.config.read_text())):,} rows")
