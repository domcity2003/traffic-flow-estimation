"""Strict streaming assembly onto the official submission key."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = {
    "state": ["panel", "timestamp", "station_id", "link_id", "mask_regime"],
    "queue": ["window_id", "timestamp", "link_id"],
    "odme": ["panel", "departure_time", "path_id"],
}
VALUES = {"state": ["speed_kmh", "flow_vph"], "queue": ["queue_pred"], "odme": ["path_flow"]}
OUTPUT_COLUMNS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]


def indexed(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    required = KEYS[task] + VALUES[task]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"missing {task} columns: {set(required) - set(frame.columns)}")
    result = frame[required].set_index(KEYS[task])
    if result.index.has_duplicates:
        raise ValueError(f"duplicate {task} natural keys")
    return result


def fill_chunk(key: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    unknown = set(key.task) - set(KEYS)
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    out = pd.DataFrame({"submission_id": pd.to_numeric(key.submission_id, errors="raise").astype("int64"),
                        "task": key.task.astype(str)}, index=key.index)
    for value in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
        out[value] = 0.0
    for task in KEYS:
        selected = key.task == task
        if not selected.any():
            continue
        table = tables[task]
        if list(table.index.names) != KEYS[task]:
            table = indexed(table, task)
        wanted = pd.MultiIndex.from_frame(key.loc[selected, KEYS[task]])
        found = table.reindex(wanted)
        values = found[VALUES[task]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"missing or nonfinite {task} predictions for {int((~np.isfinite(values)).sum())} values")
        if task == "queue" and not np.isin(values, [0, 1]).all():
            raise ValueError("queue predictions must be binary")
        if (values < 0).any():
            raise ValueError(f"negative {task} predictions")
        out.loc[selected, VALUES[task]] = values
    return out[OUTPUT_COLUMNS]


def split_state(path: Path, directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for chunk in pd.read_csv(path, chunksize=250_000, dtype={"panel": str, "timestamp": str,
                                                       "station_id": str, "link_id": str,
                                                       "mask_regime": str}):
        for panel, group in chunk.groupby("panel", sort=False):
            target = directory / f"{panel}.csv"
            first = panel not in paths
            group.to_csv(target, mode="w" if first else "a", header=first, index=False)
            paths[panel] = target
    return paths


def assemble(key_path: Path, state_path: Path, queue_path: Path, odme_path: Path,
             output: Path) -> int:
    state_paths = split_state(state_path, output.parent / "state-panels")
    queue_table = indexed(pd.read_csv(queue_path, dtype=str), "queue")
    odme_table = indexed(pd.read_csv(odme_path, dtype=str), "odme")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    count = 0
    previous_panel = None
    state_table = None
    first = True
    for chunk in pd.read_csv(key_path, chunksize=250_000, dtype=str):
        parts = []
        for (task, panel), group in chunk.groupby(["task", "panel"], sort=False):
            if task == "state":
                if panel != previous_panel:
                    if panel not in state_paths:
                        raise ValueError(f"missing state predictions for {panel}")
                    state_table = indexed(pd.read_csv(state_paths[panel], dtype={k: str for k in KEYS["state"]}), "state")
                    previous_panel = panel
                tables = {"state": state_table, "queue": queue_table, "odme": odme_table}
            else:
                tables = {"state": pd.DataFrame(), "queue": queue_table, "odme": odme_table}
            parts.append(fill_chunk(group, tables))
        result = pd.concat(parts).sort_index()
        result.to_csv(output, mode="w" if first else "a", header=first, index=False)
        first = False
        count += len(result)
        print(f"{count:,} rows assembled", flush=True)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=Path, default=Path("data/release/submission_key.csv"))
    parser.add_argument("--state", type=Path, default=Path("outputs/v1/state.csv"))
    parser.add_argument("--queue", type=Path, default=Path("outputs/v1/queue.csv"))
    parser.add_argument("--odme", type=Path, default=Path("outputs/v1/odme.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v1/submission.csv"))
    args = parser.parse_args()
    print(f"Wrote {assemble(args.key, args.state, args.queue, args.odme, args.output):,} rows")
