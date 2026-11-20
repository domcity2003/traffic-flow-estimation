"""Task 1 state: 2-D CNN residual over the v6 reconstruction on the whole day grid.

Each masked day is a (channels x 288 steps x links) image: visible speed and flow
(v6 filled where hidden) with their masks, the v6 bases, the all-link gap flag,
time of day, weekend, link free speed and lanes, and the train time-of-day
profile. Dilated convolutions reach ~5 hours along time, so the 90-minute
all-link gaps after each Task 2 window are bridged from both sides. The net has
no link identity and is shared by every panel. Output: speed and flow
residuals over v6, normalised by 100 km/h and 2000 veh/h per lane.

Day caches (masked grids, v6 bases, truth) are written by `cache()`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

CACHE = Path("outputs/lab/stcache")
PROFILE = Path("outputs/lab/profiles")
C_IN = 14


def profile(panel):
    z = np.load(PROFILE / f"{panel}.npz")
    return {k: z[k] for k in z.files}


def inputs(z, pr, i):
    """(x, y, mask) for day i of a cache dict; y/mask are zero when the cache has no truth."""
    lanes, vf = z["lanes"].astype(np.float32), z["vfree"].astype(np.float32)
    qs = 2000 * lanes[None, :]
    sv, sq, bv, bq = z["sv"][i], z["sq"][i], z["bv"][i], z["bq"][i]
    mv, mq = np.isfinite(sv), np.isfinite(sq)
    we = z["dow"][i] >= 5
    k = "we" if we else "wd"
    T, L = sv.shape
    ang = 2 * np.pi * np.arange(T) / T
    ch = [np.where(mv, sv, bv) / 100, mv, np.where(mq, sq, bq) / qs, mq, bv / 100, bq / qs,
          np.broadcast_to(z["gap"][i][:, None], (T, L)),
          np.broadcast_to(np.sin(ang)[:, None], (T, L)), np.broadcast_to(np.cos(ang)[:, None], (T, L)),
          np.full((T, L), float(we)), np.broadcast_to(vf[None] / 100, (T, L)), np.broadcast_to(lanes[None] / 5, (T, L)),
          pr[f"speed_kmh_{k}"] / 100, pr[f"flow_vph_{k}"] / qs]
    x = np.nan_to_num(np.stack(ch).astype(np.float32))
    if "ov" not in z:
        return x, None, None
    ov, oq, tg = z["ov"][i], z["oq"][i], z["tg"][i]
    ok = tg & np.isfinite(ov) & np.isfinite(oq)
    y = np.stack([np.where(ok, ov - bv, 0) / 100, np.where(ok, oq - bq, 0) / qs]).astype(np.float32)
    return x, np.nan_to_num(y), ok.astype(np.float32)


def denorm(z, r, i):
    """Speed and flow from a (2, T, L) residual for day i."""
    qs = 2000 * z["lanes"].astype(np.float32)[None, :]
    return z["bv"][i] + 100 * r[0], z["bq"][i] + qs * r[1]


class Block(nn.Module):
    def __init__(self, c, dt, dl):
        super().__init__()
        self.a = nn.Conv2d(c, c, (5, 3), padding=(2 * dt, dl), dilation=(dt, dl))
        self.b = nn.Conv2d(c, c, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.b(self.act(self.a(self.act(x))))


class Net(nn.Module):
    def __init__(self, cin=C_IN, w=32):
        super().__init__()
        dil = [(1, 1), (2, 2), (4, 4), (8, 1), (16, 2), (32, 1), (1, 1), (4, 2), (12, 1), (1, 1)]
        self.inp = nn.Conv2d(cin, w, 1)
        self.blocks = nn.Sequential(*[Block(w, a, b) for a, b in dil])
        self.out = nn.Conv2d(w, 2, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(nn.functional.gelu(self.blocks(self.inp(x))))


def load(panels, months):
    days = []
    for p in panels:
        pr = profile(p)
        for m in months:
            f = CACHE / f"{p}_{m}.npz"
            if not f.exists():
                raise FileNotFoundError(f)
            z = dict(np.load(f))
            for i in range(len(z["date"])):
                x, y, m = inputs(z, pr, i)
                # half precision in memory keeps all ten panels under ~1 GB
                days.append((p, str(z["date"][i]), x.astype(np.float16), y.astype(np.float16), m.astype(np.uint8)))
            del z
    return days


def train(days, epochs=8, width=32, seed=0, batch=4, lr=2e-3, lam=0.0, log=print):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(w=width)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    bypanel = {}
    for j, d in enumerate(days):
        bypanel.setdefault(d[0], []).append(j)

    def chunks():
        out = []
        for idx in bypanel.values():
            idx = rng.permutation(idx)
            out += [idx[s:s + batch] for s in range(0, len(idx), batch)]
        return [out[k] for k in rng.permutation(len(out))]

    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(chunks()))
    for ep in range(epochs):
        net.train(); tot = n = 0
        for ch in chunks():
            x = torch.from_numpy(np.stack([days[j][2] for j in ch]).astype(np.float32))
            y = torch.from_numpy(np.stack([days[j][3] for j in ch]).astype(np.float32))
            m = torch.from_numpy(np.stack([days[j][4] for j in ch]).astype(np.float32))[:, None]
            out = net(x)
            loss = (((out - y) ** 2) * m).sum() / m.sum().clamp(min=1)
            if lam:
                # per-lane density change between consecutive scored steps (the Task 3 conservation term)
                lanes = 5 * x[:, 11]; qs = 2000 * lanes
                bv, bq = 100 * x[:, 4], qs * x[:, 5]
                kp = (bq + qs * out[:, 1]) / (bv + 100 * out[:, 0]).clamp(min=1) / lanes
                ko = (bq + qs * y[:, 1]) / (bv + 100 * y[:, 0]).clamp(min=1) / lanes
                mm = m[:, 0]; pair = mm[:, 1:] * mm[:, :-1]
                dl = ((kp.diff(dim=1) - ko.diff(dim=1)).abs() * pair).sum() / pair.sum().clamp(min=1)
                loss = loss + lam * dl / 50
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); n += 1
        log(f"epoch {ep + 1} loss {tot / n:.5f}")
    return net


@torch.no_grad()
def predict(net, x):
    net.eval()
    return net(torch.from_numpy(x)[None])[0].numpy()
