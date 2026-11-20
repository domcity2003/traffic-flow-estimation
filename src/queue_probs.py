"""Save per-cell probabilities of saved queue models on official windows."""
import argparse
from pathlib import Path

import joblib  # models are produced locally by build_queue_v9
import numpy as np
import pandas as pd

from src.queue_gen_lab import PANELS
from src.queue_model import official_arrays
from src.queue_model_lab import predict


def probs(models: Path, split: str) -> pd.DataFrame:
    frames = []
    for panel in PANELS:
        o = official_arrays(panel, split)
        t = pd.read_csv(Path("data/release/task2") / panel / split / "sample_submission_queue.csv",
                        usecols=["window_id", "timestamp", "link_id"], dtype=str)
        tw = {k: g.reset_index(drop=True) for k, g in t.groupby("window_id")}
        for cond in ("queue_onset", "queue_ongoing"):
            m = joblib.load(models / f"{panel}-{cond}.joblib")["model"]
            idx = np.flatnonzero(o["cond"] == cond)
            oo = {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(o["cond"]) else v) for k, v in o.items()}
            p = predict(m, oo)
            for j, i in enumerate(idx):
                f = tw[o["window_id"][i]].copy()
                f["prob"] = p[j, o["cell_step"][i], o["cell_link"][i]]
                f["condition"] = cond
                frames.append(f)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    pd.concat([probs(a.models, s) for s in ("validation", "private")]).to_parquet(a.output)
