"""Queue persistence using the latest eligible observation in recent history."""

from __future__ import annotations

import pandas as pd


def predict_ongoing_lookback(
    history: pd.DataFrame,
    targets: pd.DataFrame,
    origin: str,
    cutoffs: dict[str, float],
    *,
    lookback_steps: int,
    link_order: list[str] | None = None,
    direction: str | None = None,
    radius: int = 0,
    start_step: int = 1,
) -> pd.DataFrame:
    """Persist each link's latest eligible queue state, optionally expanding it."""
    if lookback_steps < 1 or radius < 0 or start_step < 1:
        raise ValueError("invalid lookback or propagation settings")
    if radius and (direction not in ("upstream", "downstream") or link_order is None):
        raise ValueError("propagation needs a direction and link order")
    origin_ts = pd.Timestamp(origin)
    h = history.copy()
    h["timestamp"] = pd.to_datetime(h.timestamp, utc=True)
    target_times = pd.to_datetime(targets.timestamp, utc=True)
    if h.empty or h.timestamp.max() >= origin_ts or (target_times <= origin_ts).any():
        raise ValueError("history must precede the origin and targets must follow it")
    recent_times = sorted(h.timestamp.unique())[-lookback_steps:]
    eligible = h[h.timestamp.isin(recent_times) & h.is_score_eligible.astype(bool)].copy()
    eligible["speed_kmh"] = pd.to_numeric(eligible.speed_kmh, errors="coerce")
    eligible = eligible[eligible.speed_kmh.notna()]
    latest = eligible.sort_values("timestamp").drop_duplicates("link_id", keep="last")
    queued = set(latest.loc[latest.speed_kmh.le(latest.link_id.astype(str).map(cutoffs)),
                            "link_id"].astype(str))
    expanded = set(queued)
    if radius:
        position = {link: i for i, link in enumerate(link_order)}
        sign = -1 if direction == "upstream" else 1
        for link in queued:
            if link not in position:
                continue
            for distance in range(1, radius + 1):
                neighbor = position[link] + sign * distance
                if 0 <= neighbor < len(link_order):
                    expanded.add(link_order[neighbor])
    steps = {timestamp: i + 1 for i, timestamp in enumerate(sorted(target_times.unique()))}
    later = target_times.map(steps).to_numpy() >= start_step
    out = targets.copy()
    links = out.link_id.astype(str)
    out["queue_pred"] = ((links.isin(queued) & ~later) |
                         (links.isin(expanded) & later)).astype(int)
    return out
