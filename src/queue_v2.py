"""Queue rules evaluated for the controlled v2 candidate."""

from __future__ import annotations

import pandas as pd


def predict_ongoing_propagation(
    history: pd.DataFrame,
    targets: pd.DataFrame,
    origin: str,
    cutoffs: dict[str, float],
    link_order: list[str],
    *,
    direction: str,
    radius: int,
    start_step: int,
) -> pd.DataFrame:
    """Persist the current queue, then include nearby links after a horizon step."""
    if direction not in ("upstream", "downstream") or radius < 0 or start_step < 1:
        raise ValueError("invalid propagation settings")
    origin_ts = pd.Timestamp(origin)
    h = history.copy()
    h["timestamp"] = pd.to_datetime(h.timestamp, utc=True)
    target_times = pd.to_datetime(targets.timestamp, utc=True)
    if h.empty or h.timestamp.max() >= origin_ts or (target_times <= origin_ts).any():
        raise ValueError("history must precede the origin and targets must follow it")
    last = h[h.timestamp == h.timestamp.max()].copy()
    last["queued"] = (pd.to_numeric(last.speed_kmh, errors="coerce") <=
                      last.link_id.astype(str).map(cutoffs))
    last["queued"] &= last.is_score_eligible.astype(bool)
    queued = set(last.loc[last.queued, "link_id"].astype(str))
    position = {link: i for i, link in enumerate(link_order)}
    propagated = set(queued)
    sign = -1 if direction == "upstream" else 1
    for link in queued:
        if link not in position:
            continue
        for distance in range(1, radius + 1):
            neighbor = position[link] + sign * distance
            if 0 <= neighbor < len(link_order):
                propagated.add(link_order[neighbor])
    steps = {timestamp: i + 1 for i, timestamp in enumerate(sorted(target_times.unique()))}
    out = targets.copy()
    later = target_times.map(steps).to_numpy() >= start_step
    out["queue_pred"] = ((out.link_id.astype(str).isin(queued) & ~later) |
                         (out.link_id.astype(str).isin(propagated) & later)).astype(int)
    return out
