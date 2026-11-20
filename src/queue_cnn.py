"""Whole-corridor queue forecaster: a dilated 1-D CNN along the links.

Each window is a (channels x links) image: 12 history steps of speed/cut ratio,
validity, queued flag, flow/capacity and occupancy, plus time of day. The net
is translation-equivariant along the corridor and carries no link identity,
so one model is shared by every panel. It outputs 6 forecast-step logits per
link; the loss is masked to eligible cells.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

H, F = 12, 6
C_IN = 5 * H + 3


def ffill(a, m):
    """Forward-fill along time (axis 1) where m is False."""
    out = a.copy()
    for t in range(1, a.shape[1]):
        out[:, t] = np.where(m[:, t], out[:, t], out[:, t - 1])
    return out


def inputs(z, idx=None, extra=None):
    """(n, C_IN + k, L) float32 inputs from window arrays; extra is an optional (n, k, L) block."""
    sel = slice(None) if idx is None else idx
    sp, fl, oc = z["speed"][sel], z["flow"][sel], z["occ"][sel]
    cut, cap = z["cut"], z["cap"]
    m = np.isfinite(sp)
    r = np.where(m, sp / cut[None, None, :], np.nan)
    q = (r < 1) & m
    r = np.nan_to_num(ffill(r, m), nan=2.0).clip(0, 3)
    f = np.nan_to_num(ffill(np.where(m, fl / cap[None, None, :], np.nan), m), nan=.3).clip(0, 2)
    o = np.nan_to_num(ffill(np.where(m, oc, np.nan), m), nan=.05).clip(0, 1)
    n, _, L = sp.shape
    minute = np.asarray(z["minute"])[sel].astype(np.float32)
    dow = np.asarray(z["dow"])[sel]
    ang = 2 * np.pi * minute / 1440
    glob = np.stack([np.sin(ang), np.cos(ang), (dow >= 5).astype(np.float32)], 1)
    x = np.concatenate([r, m, q, f, o * 4, np.broadcast_to(glob[:, :, None], (n, 3, L))]
                       + ([] if extra is None else [extra[sel]]), 1)
    return x.astype(np.float32)


class Block(nn.Module):
    def __init__(self, c, d, p=.1):
        super().__init__()
        self.a = nn.Conv1d(c, c, 5, padding=2 * d, dilation=d)
        self.b = nn.Conv1d(c, c, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return x + self.b(self.drop(self.act(self.a(self.act(x)))))


class Net(nn.Module):
    def __init__(self, width=64, dil=(1, 2, 4, 8, 16, 1, 2, 4, 8, 1), c_in=C_IN):
        super().__init__()
        self.inp = nn.Conv1d(c_in, width, 1)
        self.blocks = nn.Sequential(*[Block(width, d) for d in dil])
        self.out = nn.Conv1d(width, F, 1)

    def forward(self, x):
        return self.out(nn.functional.gelu(self.blocks(self.inp(x))))   # (n, F, L)


def train(data, epochs=15, batch=32, lr=2e-3, seed=0, log=print, extras=None, width=64):
    """data: list of window dicts (one per panel) with speed/flow/occ/truth/elig/cut/cap/minute/dow;
    extras: optional list of (n, k, L) per-link input blocks aligned with data."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = [torch.from_numpy(inputs(z, extra=None if extras is None else extras[i])) for i, z in enumerate(data)]
    Y = [torch.from_numpy(z["truth"].astype(np.float32)) for z in data]
    E = [torch.from_numpy(z["elig"].astype(np.float32)) for z in data]
    net = Net(width=width, c_in=X[0].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    chunks = [(k, s) for k, x in enumerate(X) for s in range(0, len(x), batch)]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(chunks))
    for ep in range(epochs):
        net.train()
        tot = cnt = 0.0
        perm = [rng.permutation(len(x)) for x in X]
        for j in rng.permutation(len(chunks)):
            k, s = chunks[j]
            i = torch.from_numpy(perm[k][s:s + batch])
            logit = net(X[k][i])
            e = E[k][i]
            loss = (nn.functional.binary_cross_entropy_with_logits(logit, Y[k][i], reduction="none") * e).sum() / e.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); cnt += 1
        log(f"epoch {ep + 1} loss {tot / cnt:.4f}")
    return net


@torch.no_grad()
def predict(net, z, batch=64, extra=None):
    net.eval()
    x = torch.from_numpy(inputs(z, extra=extra))
    out = [torch.sigmoid(net(x[s:s + batch])) for s in range(0, len(x), batch)]
    return torch.cat(out).numpy()                                          # (n, F, L)
