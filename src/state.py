"""Memory-bounded reconstruction of published Task 1 target cells."""

from __future__ import annotations

import numpy as np
import pandas as pd


def interpolate_grid(grid: np.ndarray, axis: int) -> np.ndarray:
    """Linearly fill internal gaps and extend endpoints on one grid axis."""
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    result = np.array(grid, dtype=float, copy=True)
    view = result if axis == 0 else result.T
    positions = np.arange(view.shape[0])
    for column in range(view.shape[1]):
        valid = np.isfinite(view[:, column])
        if valid.any():
            view[:, column] = np.interp(positions, positions[valid], view[valid, column])
    return result


def predict_day(
    masked: pd.DataFrame,
    targets: pd.DataFrame,
    link_order: list[str],
    temporal_weight: float | dict[str, float] = 0.8,
    fallback: dict[str, dict[str, float]] | None = None,
    min_pct_observed: float = 0.0,
    spatial_scale: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Predict only template rows from observations visible in one masked day.

    The spatial axis follows mainline topology. ``fallback`` supplies per-link
    profile values for a link/day with no visible observations anywhere nearby.
    """
    weights = ({field: temporal_weight for field in ("speed_kmh", "flow_vph")}
               if isinstance(temporal_weight, (float, int)) else temporal_weight)
    if any(field not in weights or not 0 <= weights[field] <= 1
           for field in ("speed_kmh", "flow_vph")):
        raise ValueError("temporal_weight must be in [0, 1]")
    if targets.empty:
        return targets.assign(speed_kmh=pd.Series(dtype=float), flow_vph=pd.Series(dtype=float))
    frame = masked.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["link_id"] = frame["link_id"].astype(str)
    frame = frame.sort_values(["timestamp", "link_id"])
    if frame.duplicated(["timestamp", "link_id"]).any():
        raise ValueError("duplicate timestamp/link_id in masked day")
    times = pd.DatetimeIndex(sorted(frame.timestamp.unique()))
    target_times = pd.to_datetime(targets.timestamp, utc=True)
    ti = times.get_indexer(target_times)
    li = pd.Index(link_order).get_indexer(targets.link_id.astype(str))
    if (ti < 0).any() or (li < 0).any():
        raise ValueError("template contains timestamp or link absent from masked day")
    predictions = {}
    quality = pd.to_numeric(frame.get("pct_observed", 100), errors="coerce")
    for field in ("speed_kmh", "flow_vph"):
        visible = pd.to_numeric(frame[field], errors="coerce").where(quality >= min_pct_observed)
        grid = frame.assign(value=visible).pivot(index="timestamp", columns="link_id", values="value")
        grid = grid.reindex(index=times, columns=link_order).to_numpy(dtype=float)
        temporal = interpolate_grid(grid, axis=0)
        field_scale = (spatial_scale or {}).get(field, {})
        scale = np.array([field_scale.get(link, 1.0) for link in link_order], dtype=float)
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(f"invalid spatial scale for {field}")
        spatial = interpolate_grid(grid / scale[None, :], axis=1) * scale[None, :]
        t_values, s_values = temporal[ti, li], spatial[ti, li]
        blended = np.where(
            np.isfinite(t_values) & np.isfinite(s_values),
            weights[field] * t_values + (1 - weights[field]) * s_values,
            np.where(np.isfinite(t_values), t_values, s_values),
        )
        if fallback is not None:
            replacement = np.array([fallback.get(link, {}).get(field, np.nan) for link in targets.link_id.astype(str)])
            blended = np.where(np.isfinite(blended), blended, replacement)
        predictions[field] = np.maximum(blended, 0.0)
    out = targets.copy()
    for field, values in predictions.items():
        out[field] = values
    return out
