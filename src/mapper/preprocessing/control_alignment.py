"""Protein-wise control-relative alignment; no treated-data parameter fitting."""

import numpy as np


def available_mean(values, axis=0):
    values = np.asarray(values, dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite intensity")
    count = np.isfinite(values).sum(axis=axis)
    total = np.nansum(values, axis=axis)
    return (
        np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0),
        count,
    )


def fit_reference(control_means, training_groups):
    means = np.asarray(control_means, dtype=float)
    indices = np.asarray(training_groups)
    if (
        means.ndim != 2
        or indices.ndim != 1
        or not len(indices)
        or not np.issubdtype(indices.dtype, np.integer)
    ):
        raise ValueError("Expected group-by-protein means and nonempty integer group indices")
    if (indices < 0).any() or (indices >= len(means)).any():
        raise ValueError("Invalid group index")
    return available_mean(means[np.unique(indices)])[0]


def transform(raw, assigned_means, reference):
    x = np.asarray(raw, dtype=float)
    c = np.asarray(assigned_means, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if x.ndim != 2 or c.shape != x.shape or ref.shape != (x.shape[1],):
        raise ValueError("Shape mismatch")
    if any(np.isinf(a).any() for a in (x, c, ref)):
        raise ValueError("Infinite value")
    support = np.isfinite(c) & np.isfinite(ref)[None, :]
    shifts = np.zeros_like(x)
    np.subtract(ref[None, :], c, out=shifts, where=support)
    observed = np.isfinite(x)
    result = np.zeros_like(x)
    np.add(x, shifts, out=result, where=observed)
    return result, shifts, support
