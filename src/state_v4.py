"""Same-day low-rank reconstruction using only released visible readings."""
import numpy as np
from src.state import interpolate_grid


def complete_grid(grid, rank=4, iterations=20):
    if grid.ndim != 2 or not np.isfinite(grid).any():
        raise ValueError("grid must be a nonempty 2D array with visible values")
    if rank < 1 or iterations < 1:
        raise ValueError("rank and iterations must be positive")
    observed = np.isfinite(grid)
    temporal = interpolate_grid(grid, 0)
    spatial = interpolate_grid(grid, 1)
    initial = np.where(np.isfinite(temporal), temporal, spatial)
    initial = np.nan_to_num(initial, nan=float(np.nanmean(grid)))
    # Per-sensor scale avoids fitting only the highest-volume detectors.
    scale = np.maximum(np.mean(initial, axis=0), 1.)
    x = initial / scale
    original = np.nan_to_num(grid / scale)
    for _ in range(iterations):
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        estimate = (u[:, :rank] * s[:rank]) @ vt[:rank]
        x = np.where(observed, original, estimate)
    return np.where(observed, grid, np.maximum(x * scale, 0))


def predict_day(masked, targets, link_order, temporal_weight=.8, rules=None):
    import pandas as pd
    from src.state import predict_day as baseline
    result = baseline(masked, targets, link_order, temporal_weight=temporal_weight)
    for field, rule in (rules or {}).items():
        frame = masked.pivot(index='timestamp', columns='link_id', values=field).reindex(columns=link_order)
        ti = frame.index.get_indexer(targets.timestamp)
        li = pd.Index(link_order).get_indexer(targets.link_id)
        if (ti < 0).any() or (li < 0).any():
            raise ValueError('target missing from reconstruction grid')
        reconstructed = complete_grid(frame.to_numpy(float), rank=rule['rank'])
        blend = rule['blend']
        if not 0 <= blend <= 1:
            raise ValueError('blend must be in [0, 1]')
        result[field] = (1-blend)*result[field].to_numpy()+blend*reconstructed[ti,li]
    return result
