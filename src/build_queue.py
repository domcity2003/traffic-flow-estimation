"""Train recurrent onset links on released train windows, forecast scored splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.bench_queue import queue_proxy
from src.queue import predict_window


def train_onset_counts(release: Path, panel: str, cutoffs: dict[str, float]) -> Counter[str]:
    root = release / "task2" / panel / "train"
    windows = pd.read_csv(root / "window_index.csv", dtype=str)
    template = pd.read_csv(root / "sample_submission_queue.csv", dtype=str)
    by_window = {key: group for key, group in template.groupby("window_id")}
    counts: Counter[str] = Counter()
    for window in windows[windows.condition == "queue_onset"].itertuples(index=False):
        truth = queue_proxy(release, panel, window, by_window[window.window_id], cutoffs)
        counts.update(truth.loc[truth.queue_true, "link_id"].astype(str))
    return counts


def build(release: Path, output: Path, settings: dict) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    first = True
    total = 0
    for panel in sorted(settings):
        fd = pd.read_csv(release / "corridors" / panel / "network" / "fd_parameters.csv")
        cutoffs = dict(zip(fd.link_id.astype(str), .6 * fd.free_speed_kmh))
        counts = train_onset_counts(release, panel, cutoffs)
        onset_links = [link for link, _ in counts.most_common(settings[panel]["onset_top_k"])]
        print(panel, "onset links", onset_links, flush=True)
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
                scaled = {link: value * settings[panel]["ongoing_cutoff_factor"]
                          for link, value in cutoffs.items()}
                prediction = predict_window(h_by_window[window.window_id],
                                            t_by_window[window.window_id],
                                            window.forecast_origin, window.condition,
                                            scaled, onset_links)
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
    parser.add_argument("--config", type=Path, default=Path("config/queue_settings.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v1/queue.csv"))
    args = parser.parse_args()
    print(f"Wrote {build(args.release, args.output, json.loads(args.config.read_text())):,} rows")
