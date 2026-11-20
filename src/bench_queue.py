"""Proxy backtest for the withheld Task 2 labels on released train states."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.queue import predict_window


def queue_proxy(release: Path, panel: str, window: pd.Series, target: pd.DataFrame,
                cutoffs: dict[str, float]) -> pd.DataFrame:
    day = window.date.replace("-", "_")
    matches = list((release / "corridors" / panel / "train" / "mainline_states").glob(f"**/*{day}*.parquet"))
    if len(matches) != 1:
        raise ValueError(f"expected one unmasked train day for {panel} {day}")
    truth = pd.read_parquet(matches[0], columns=["timestamp", "link_id", "speed_kmh", "is_score_eligible"])
    out = target.merge(truth, on=["timestamp", "link_id"], how="left", validate="one_to_one")
    out["queue_true"] = out.speed_kmh.le(out.link_id.map(cutoffs)) & out.is_score_eligible.astype(bool)
    return out


def window_iou(truth: pd.DataFrame, prediction: pd.DataFrame) -> float:
    scored = truth[truth.is_score_eligible.astype(bool)].copy()
    pred = prediction.queue_pred.to_numpy(dtype=bool)[truth.is_score_eligible.astype(bool).to_numpy()]
    actual = scored.queue_true.to_numpy(dtype=bool)
    union = np.logical_or(pred, actual).sum()
    return float(np.logical_and(pred, actual).sum() / union) if union else 1.0


def evaluate_panel(release: Path, panel: str, k_values: list[int]) -> tuple[pd.DataFrame, dict[str, int]]:
    root = release / "task2" / panel / "train"
    windows = pd.read_csv(root / "window_index.csv", dtype=str)
    history = pd.read_parquet(root / "window_history.parquet")
    template = pd.read_csv(root / "sample_submission_queue.csv", dtype=str)
    fd = pd.read_csv(release / "corridors" / panel / "network" / "fd_parameters.csv")
    cutoffs = dict(zip(fd.link_id.astype(str), .6 * fd.free_speed_kmh))
    h_by_window = {key: value for key, value in history.groupby("window_id")}
    t_by_window = {key: value for key, value in template.groupby("window_id")}
    truth_by_window = {}
    onset_counts: Counter[str] = Counter()
    for w in windows.itertuples(index=False):
        truth = queue_proxy(release, panel, w, t_by_window[w.window_id], cutoffs)
        truth_by_window[w.window_id] = truth
        if w.condition == "queue_onset":
            onset_counts.update(truth.loc[truth.queue_true, "link_id"].astype(str))
    rows = []
    for w in windows.itertuples(index=False):
        truth = truth_by_window[w.window_id]
        own = Counter(truth.loc[truth.queue_true, "link_id"].astype(str)) if w.condition == "queue_onset" else Counter()
        training_counts = onset_counts - own
        settings = [(k, 1.) for k in k_values] if w.condition == "queue_onset" else [
            (0, factor) for factor in (.9, 1., 1.05, 1.1, 1.2, 1.3)]
        for top_k, factor in settings:
            learned = [link for link, count in training_counts.most_common(top_k) if count > 0]
            scaled = {link: value * factor for link, value in cutoffs.items()}
            pred = predict_window(h_by_window[w.window_id], t_by_window[w.window_id],
                                  w.forecast_origin, w.condition, scaled, learned)
            rows.append(dict(panel=panel, window_id=w.window_id, condition=w.condition,
                             top_k=top_k, cutoff_factor=factor, iou=window_iou(truth, pred),
                             n_true=int(truth.queue_true.sum()), n_pred=int(pred.queue_pred.sum())))
    return pd.DataFrame(rows), dict(onset_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("data/release"))
    parser.add_argument("--output", type=Path, default=Path("reports/queue-train-proxy.csv"))
    args = parser.parse_args()
    frames = []
    counts = {}
    for panel_dir in sorted((args.release / "task2").iterdir()):
        if not panel_dir.is_dir():
            continue
        print(panel_dir.name, flush=True)
        frame, panel_counts = evaluate_panel(args.release, panel_dir.name, [0, 1, 2, 3, 5, 8, 12, 20])
        frames.append(frame)
        counts[panel_dir.name] = panel_counts
    result = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    (args.output.parent / "queue-onset-train-counts.json").write_text(json.dumps(counts, indent=2))
    print(result.groupby(["condition", "top_k", "cutoff_factor"]).iou.mean().round(4).to_string())
