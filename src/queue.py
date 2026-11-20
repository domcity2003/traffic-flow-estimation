"""Forecast binary queues using only published pre-origin history."""

from __future__ import annotations

import pandas as pd


def predict_window(
    history: pd.DataFrame,
    targets: pd.DataFrame,
    origin: str,
    condition: str,
    cutoffs: dict[str, float],
    onset_links: list[str],
) -> pd.DataFrame:
    """Predict a published queue template for one window.

    Onset uses recurrent train bottlenecks at the final forecast step. Ongoing
    carries forward the last visible queued links. Future state is rejected.
    """
    origin_ts = pd.Timestamp(origin)
    h = history.copy()
    h["timestamp"] = pd.to_datetime(h.timestamp, utc=True)
    target_times = pd.to_datetime(targets.timestamp, utc=True)
    if h.empty or h.timestamp.max() >= origin_ts:
        raise ValueError("history must end before the forecast origin")
    if (target_times <= origin_ts).any():
        raise ValueError("targets must be after the forecast origin")
    if condition not in ("queue_onset", "queue_ongoing"):
        raise ValueError(f"unknown queue condition: {condition}")
    out = targets.copy()
    if condition == "queue_onset":
        is_final = target_times == target_times.max()
        out["queue_pred"] = (is_final & out.link_id.isin(onset_links)).astype(int)
    else:
        last = h[h.timestamp == h.timestamp.max()].copy()
        last["is_queue"] = (pd.to_numeric(last.speed_kmh, errors="coerce") <=
                            last.link_id.astype(str).map(cutoffs))
        last["is_queue"] &= last.is_score_eligible.astype(bool)
        queued = last.groupby("link_id").is_queue.any()
        out["queue_pred"] = out.link_id.astype(str).map(queued).fillna(False).astype(int)
    return out
