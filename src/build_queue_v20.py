"""V20 ongoing: whole-corridor CNN (src/queue_cnn.py) blended 50/50 with v17's GBMs.

Trained on every train month of the 60-minute-spaced generated windows. With
July-December training, the 50/50 blend raised pseudo-window ongoing IoU
(threshold 0.45) from 0.845/0.811/0.864/0.838 to 0.867/0.839/0.893/0.852
(June/April/Feb/March).

  python -m src.build_queue_v20 train SEED    # fit one CNN, save test probabilities
  python -m src.build_queue_v20 assemble      # blend saved seeds with outputs/lab/probs17
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

import src.queue_gen_lab as gen
from src import queue_cnn as qc
from src.assemble_queue_v17 import PROBS, assemble
from src.build_queue_v9 import subset
from src.queue_eval import take
from src.queue_gen_lab import PANELS
from src.queue_model import choose_set, official_arrays

COND = "queue_ongoing"
WIDTH = int(os.environ.get("CNN_WIDTH", "64"))
CNN = Path("outputs/lab/cnn20" if WIDTH == 64 else f"outputs/lab/cnn20w{WIDTH}")
OUT = Path("outputs/v20")
EPOCHS, C, W, THR = int(os.environ.get("CNN_EPOCHS", "15")), .5, .35, .45


def train(seed):
    gen.QWIN = Path("outputs/lab/qwin60")
    CNN.mkdir(parents=True, exist_ok=True)
    data = [subset(gen.load(p), COND) for p in PANELS]
    print("windows", sum(len(z["cond"]) for z in data), flush=True)
    net = qc.train(data, epochs=EPOCHS, seed=seed, width=WIDTH, log=lambda s: print(s, flush=True))
    torch.save(net.state_dict(), CNN / f"seed{seed}.pt")
    for p in PANELS:
        for split in ("validation", "private"):
            o = official_arrays(p, split)
            np.save(CNN / f"seed{seed}_{p}_{split}.npy", qc.predict(net, take(o, o["cond"] == COND)).astype(np.float16))
    print("saved seed", seed, flush=True)


def proba(z, p, split):
    cnn = np.mean([np.load(f).astype(np.float32) for f in sorted(CNN.glob(f"seed*_{p}_{split}.npy"))], 0)
    return C * cnn + (1 - C) * (W * z["per"] + (1 - W) * z["pool"])


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(int(sys.argv[2]))
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        print("seeds", len(list(CNN.glob("seed*.pt"))))
        assemble(COND, proba, lambda pr, e: choose_set(pr, e, "threshold", THR), OUT / "queue_ongoing.csv")
