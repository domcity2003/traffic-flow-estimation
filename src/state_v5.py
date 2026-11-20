"""Limit the influence of v4's unstable matrix-completion extrapolations."""
import numpy as np


def bounded_blend(base, candidate, cap, blend=.5):
    base=np.asarray(base,dtype=float)
    candidate=np.asarray(candidate,dtype=float)
    if not 0 <= blend <= 1 or np.any(np.asarray(cap)<=0):
        raise ValueError('invalid blend or cap')
    if not np.isfinite(base).all() or not np.isfinite(candidate).all():
        raise ValueError('predictions must be finite')
    return base+blend*np.clip(candidate-base,-np.asarray(cap),np.asarray(cap))
