"""Observed log2 sample-median alignment; training-only reference, no labels."""

import numpy as np


def sample_medians(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or np.isinf(values).any():
        raise ValueError("Expected a nonempty 2D log2 matrix without infinities")
    if np.isnan(values).all(axis=1).any():
        raise ValueError("An empty profile has no observed median")
    return np.nanmedian(values, axis=1)


def fit_reference(training):
    return float(np.median(sample_medians(training)))


def transform(values, reference):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(reference):
        raise ValueError("Reference must be finite")
    medians = sample_medians(values)
    shifts = reference - medians
    adjusted = values + shifts[:, None]
    np.testing.assert_array_equal(np.isnan(adjusted), np.isnan(values))
    np.testing.assert_allclose(np.nanmedian(adjusted, axis=1), reference, atol=1e-12, rtol=0)
    return np.nan_to_num(adjusted, nan=0.0), medians, shifts
