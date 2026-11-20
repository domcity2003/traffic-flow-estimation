"""Build the frozen v3 queue candidate for validation and private splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.build_queue import train_onset_counts
from src.queue import predict_window
from src.queue_v2 import predict_ongoing_propagation
from src.queue_v3 import predict_ongoing_lookback


def build(release: Path, output: Path, settings: dict, v2_overrides: dict,
          v3_overrides: dict, event_counts: dict) -> int:
    if (set(v2_overrides) | set(v3_overrides)) - set(settings):
        raise ValueError("queue override contains unknown panels")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    first = True
    total = 0
    for panel in sorted(settings):
        fd = pd.read_csv(release / "corridors" / panel / "network" / "fd_parameters.csv")
        cutoffs = dict(zip(fd.link_id.astype(str), .6 * fd.free_speed_kmh))
        base = settings[panel]
        change = v2_overrides.get(panel, {})
        v3_change = v3_overrides.get(panel, {})
        if set(change) - {"onset_source", "onset_top_k", "ongoing_propagation"}:
            raise ValueError(f"unknown v2 setting for {panel}")
        if set(v3_change) - {"ongoing_lookback", "ongoing_propagation"}:
            raise ValueError(f"unknown v3 setting for {panel}")
        source = change.get("onset_source", "v1")
        if source == "events":
            counts = event_counts[panel]["links"]
            ranking = sorted(counts, key=lambda link: (-counts[link], link))
        elif source == "v1":
            ranking = [link for link, _ in train_onset_counts(release, panel, cutoffs).most_common()]
        else:
            raise ValueError(f"unknown onset source for {panel}: {source}")
        onset_links = ranking[:change.get("onset_top_k", base["onset_top_k"])]
        propagation = v3_change.get("ongoing_propagation",
                                    change.get("ongoing_propagation"))
        link_order = None
        if propagation:
            topo = pd.read_csv(release / "corridors" / panel / "network" / "lwr_mainline_topology.csv")
            link_order = topo.sort_values("order_index").mainline_link_id.astype(str).tolist()
        print(panel, "onset", source, onset_links, "ongoing",
              v3_change or propagation or "v1", flush=True)
        scaled = {link: value * base["ongoing_cutoff_factor"] for link, value in cutoffs.items()}
        for split in ("validation", "private"):
            root = release / "task2" / panel / split
            windows = pd.read_csv(root / "window_index.csv", dtype=str)
            history = pd.read_parquet(root / "window_history.parquet")
            template = pd.read_csv(root / "sample_submission_queue.csv",
                                   usecols=["window_id", "timestamp", "link_id"], dtype=str)
            h_by_window = {key: group for key, group in history.groupby("window_id")}
            t_by_window = {key: group for key, group in template.groupby("window_id")}
            count = 0
            for window in windows.itertuples(index=False):
                h = h_by_window[window.window_id]
                t = t_by_window[window.window_id]
                if window.condition == "queue_ongoing" and v3_change:
                    prediction = predict_ongoing_lookback(
                        h, t, window.forecast_origin, scaled,
                        lookback_steps=v3_change["ongoing_lookback"],
                        link_order=link_order, **(propagation or {}))
                elif window.condition == "queue_ongoing" and propagation:
                    prediction = predict_ongoing_propagation(
                        h, t, window.forecast_origin, scaled, link_order, **propagation)
                else:
                    prediction = predict_window(h, t, window.forecast_origin,
                                                window.condition, scaled, onset_links)
                if not prediction.queue_pred.isin([0, 1]).all():
                    raise ValueError(f"invalid queue prediction for {window.window_id}")
                prediction[["window_id", "timestamp", "link_id", "queue_pred"]].to_csv(
                    output, mode="w" if first else "a", header=first, index=False)
                first = False
                count += len(prediction)
            if count != len(template):
                raise ValueError(f"queue target coverage mismatch for {panel} {split}")
            total += count
            print(panel, split, f"{count:,} rows", flush=True)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("data/release"))
    parser.add_argument("--settings", type=Path, default=Path("config/queue_settings.json"))
    parser.add_argument("--v2-overrides", type=Path, default=Path("config/queue_v2_overrides.json"))
    parser.add_argument("--v3-overrides", type=Path, default=Path("config/queue_v3_overrides.json"))
    parser.add_argument("--event-counts", type=Path, default=Path("reports/queue-event-counts-through-dec.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v3/queue.csv"))
    args = parser.parse_args()
    print(f"Wrote {build(args.release, args.output, json.loads(args.settings.read_text()), json.loads(args.v2_overrides.read_text()), json.loads(args.v3_overrides.read_text()), json.loads(args.event_counts.read_text())):,} rows")
