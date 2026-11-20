"""Check the complete upload file against the exact released submission key."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from importlib.metadata import version
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd


VALUES = ["speed_kmh", "flow_vph", "queue_pred", "path_flow"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(key_path: Path, submission_path: Path) -> dict:
    key_reader = pd.read_csv(key_path, usecols=["submission_id", "task"], chunksize=250_000)
    sub_reader = pd.read_csv(submission_path, chunksize=250_000)
    counts: Counter[str] = Counter()
    last_id = 0
    for key, sub in zip_longest(key_reader, sub_reader):
        if key is None or sub is None or len(key) != len(sub):
            raise ValueError("submission row count differs from official key")
        if list(sub.columns) != ["submission_id", "task", *VALUES]:
            raise ValueError("submission must have exactly the six official columns")
        ids = sub.submission_id.to_numpy(dtype=np.int64)
        if not np.array_equal(ids, key.submission_id.to_numpy(dtype=np.int64)):
            raise ValueError("submission IDs differ from official key")
        if not np.array_equal(sub.task.to_numpy(), key.task.to_numpy()):
            raise ValueError("task order differs from official key")
        if ids[0] != last_id + 1 or not np.array_equal(ids, np.arange(ids[0], ids[0] + len(ids))):
            raise ValueError("submission IDs are not contiguous")
        last_id = int(ids[-1])
        data = sub[VALUES].to_numpy(dtype=float)
        if not np.isfinite(data).all() or (data < 0).any():
            raise ValueError("submission has nonfinite or negative values")
        task = sub.task.to_numpy()
        queue = data[task == "queue", 2]
        if not np.isin(queue, [0, 1]).all():
            raise ValueError("queue predictions must be binary")
        if (data[task == "state", 2:] != 0).any() or (data[task == "queue"][:, [0, 1, 3]] != 0).any() or (data[task == "odme", :3] != 0).any():
            raise ValueError("unscored value columns must be zero")
        counts.update(sub.task.value_counts().to_dict())
    if last_id == 0:
        raise ValueError("empty submission")
    return {"rows": last_id, "task_counts": dict(counts), "sha256": sha256(submission_path),
            "file_bytes": submission_path.stat().st_size,
            "python": sys.version.split()[0],
            "packages": {name: version(name) for name in ("numpy", "pandas", "pyarrow", "scipy")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=Path, default=Path("data/release/submission_key.csv"))
    parser.add_argument("--submission", type=Path, default=Path("outputs/v1/submission.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/v1/manifest.json"))
    args = parser.parse_args()
    result = validate(args.key, args.submission)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
