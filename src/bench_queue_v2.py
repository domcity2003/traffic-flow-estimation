"""Compare v1 and train-event queue rules on time-disjoint months."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.bench_queue import window_iou
from src.build_queue import train_onset_counts
from src.queue import predict_window
from src.queue_v2 import predict_ongoing_propagation


def proxy_windows(release: Path, panel: str, month: str, cutoffs: dict[str, float],
                  max_per_condition: int = 5):
    """Select observed onset and ongoing windows on distinct days."""
    selected = Counter()
    paths = sorted((release / "corridors" / panel / "train" / "mainline_states").glob(
        f"**/*{month.replace('-', '_')}_*.parquet"))
    for path in paths:
        if (selected["queue_onset"] >= max_per_condition and
                selected["queue_ongoing"] >= max_per_condition):
            break
        day = pd.read_parquet(path, columns=["timestamp", "link_id", "speed_kmh", "flow_vph",
                                                 "occupancy", "pct_observed", "is_score_eligible"])
        day["queue_true"] = day.speed_kmh.le(day.link_id.map(cutoffs)) & day.is_score_eligible.astype(bool)
        by_time = day.groupby("timestamp", sort=True)
        times = list(by_time.groups)
        q_count = by_time.queue_true.sum().to_numpy()
        coverage = by_time.is_score_eligible.mean().to_numpy()
        choice = {}
        for i in range(12, len(times) - 6):
            if np.mean(coverage[i-12:i]) < .7 or np.mean(coverage[i+1:i+7]) < .7:
                continue
            if (q_count[i-12:i] == 0).all() and (q_count[i+1:i+6] == 0).all() and q_count[i+6] > 0:
                choice.setdefault("queue_onset", i)
            if q_count[i-1] > 0 and q_count[i+1:i+7].sum() > 0:
                choice.setdefault("queue_ongoing", i)
            if len(choice) == 2:
                break
        for condition, i in choice.items():
            if selected[condition] >= max_per_condition:
                continue
            history = day[day.timestamp.isin(times[i-12:i])]
            future = day[day.timestamp.isin(times[i+1:i+7])]
            selected[condition] += 1
            yield condition, times[i], history, future


def evaluate(release: Path, month: str, event_counts: dict, settings: dict) -> pd.DataFrame:
    rows = []
    for panel in sorted(settings):
        fd = pd.read_csv(release / "corridors" / panel / "network" / "fd_parameters.csv")
        cutoffs = dict(zip(fd.link_id.astype(str), .6 * fd.free_speed_kmh))
        june_counts = train_onset_counts(release, panel, cutoffs)
        topo = pd.read_csv(release / "corridors" / panel / "network" / "lwr_mainline_topology.csv")
        link_order = topo.sort_values("order_index").mainline_link_id.astype(str).tolist()
        event_ranking = sorted(event_counts[panel]["links"],
                               key=lambda link: (-event_counts[panel]["links"][link], link))
        june_ranking = [link for link, _ in june_counts.most_common()]
        n = 0
        for condition, origin, history, future in proxy_windows(release, panel, month, cutoffs):
            n += 1
            target = future[["timestamp", "link_id"]]
            policies = [("v1", settings[panel]["onset_top_k"], june_ranking)]
            if condition == "queue_onset":
                policies.extend(("events", k, event_ranking) for k in (1, 2, 3, 5, 8, 12, 20))
            for policy, top_k, ranking in policies:
                factor = settings[panel]["ongoing_cutoff_factor"]
                scaled = {link: value * factor for link, value in cutoffs.items()}
                pred = predict_window(history, target, origin, condition, scaled, ranking[:top_k])
                rows.append(dict(month=month, panel=panel, condition=condition, origin=origin,
                                 policy=policy, top_k=top_k, iou=window_iou(future, pred),
                                 n_true=int(future.queue_true.sum()), n_pred=int(pred.queue_pred.sum())))
            if condition == "queue_ongoing":
                factor = settings[panel]["ongoing_cutoff_factor"]
                scaled = {link: value * factor for link, value in cutoffs.items()}
                for direction in ("upstream", "downstream"):
                    for radius in (1, 2, 3):
                        for start_step in (1, 3, 5):
                            pred = predict_ongoing_propagation(
                                history, target, origin, scaled, link_order,
                                direction=direction, radius=radius, start_step=start_step)
                            rows.append(dict(month=month, panel=panel, condition=condition, origin=origin,
                                             policy=f"{direction}-r{radius}-s{start_step}", top_k=-1,
                                             iou=window_iou(future, pred),
                                             n_true=int(future.queue_true.sum()),
                                             n_pred=int(pred.queue_pred.sum())))
        print(panel, n, "windows", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", choices=["2031-01", "2031-02"], required=True)
    parser.add_argument("--release", type=Path, default=Path("data/release"))
    parser.add_argument("--event-counts", type=Path, default=Path("reports/queue-event-counts-through-dec.json"))
    parser.add_argument("--settings", type=Path, default=Path("config/queue_settings.json"))
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    event_counts = json.loads(args.event_counts.read_text())
    frame = evaluate(args.release, args.month, event_counts, settings)
    out = Path(f"reports/queue-v2-{args.month}.csv")
    frame.to_csv(out, index=False)
    print(frame.groupby(["condition", "policy", "top_k"]).iou.mean().round(4).to_string())
