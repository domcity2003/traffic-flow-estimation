"""Prior-regularized nonnegative path-flow reconstruction from released counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls


def solve(A: np.ndarray, counts: np.ndarray, prior: np.ndarray,
          regularization: float = .05) -> np.ndarray:
    if regularization < 0:
        raise ValueError("regularization must be nonnegative")
    n_paths = A.shape[1]
    augmented = np.vstack([A, np.sqrt(regularization) * np.eye(n_paths)])
    target = np.concatenate([counts, np.sqrt(regularization) * prior])
    flows, _ = nnls(augmented, target)
    return flows


def build_panel(release: Path, panel: str, split: str) -> pd.DataFrame:
    network = release / "corridors" / panel / "network"
    path_set = pd.read_csv(network / "path_set.csv", dtype={"path_id": str})
    incidence = pd.read_csv(network / "path_link_incidence.csv",
                            dtype={"path_id": str, "link_id": str})
    path_ids = path_set.path_id.astype(str).tolist()
    link_ids = incidence.link_id.drop_duplicates().tolist()
    path_index = {name: i for i, name in enumerate(path_ids)}
    link_index = {name: i for i, name in enumerate(link_ids)}
    A = np.zeros((len(link_ids), len(path_ids)), dtype=float)
    A[incidence.link_id.map(link_index).to_numpy(dtype=int),
      incidence.path_id.map(path_index).to_numpy(dtype=int)] = 1.
    root = release / "task4" / panel / split
    counts_frame = pd.read_csv(root / "synthetic_link_counts.csv", dtype={"link_id": str})
    prior_frame = pd.read_csv(root / "synthetic_weak_prior.csv", dtype={"path_id": str})
    observed_links = set(counts_frame.link_id)
    measured = np.array([link in observed_links for link in link_ids])
    counts = counts_frame.set_index("link_id").reindex(link_ids)["count"].fillna(0.).to_numpy(dtype=float)
    prior = prior_frame.set_index("path_id").reindex(path_ids)["path_flow"].fillna(0.).to_numpy(dtype=float)
    flows = solve(A[measured], counts[measured], prior)
    out = path_set[["path_id", "origin_zone", "destination_zone"]].copy()
    out.insert(0, "panel", panel)
    out.insert(1, "departure_time", str(prior_frame.departure_time.iloc[0]))
    out["path_flow"] = flows
    return out[["panel", "departure_time", "path_id", "origin_zone", "destination_zone", "path_flow"]]


def build(release: Path, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    first = True
    total = 0
    for panel_dir in sorted((release / "task4").iterdir()):
        if not panel_dir.is_dir():
            continue
        for split in ("validation", "private"):
            frame = build_panel(release, panel_dir.name, split)
            frame.to_csv(output, mode="w" if first else "a", header=first, index=False)
            first = False
            total += len(frame)
            print(panel_dir.name, split, f"{len(frame):,} paths", flush=True)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("data/release"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v1/odme.csv"))
    args = parser.parse_args()
    print(f"Wrote {build(args.release, args.output):,} rows")
