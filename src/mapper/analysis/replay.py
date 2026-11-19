"""Numerical replay checks with exact scientific-decision invariants.

CPU linear-algebra builds can perturb an optimized logistic-regression margin
without changing its predictions. Margins allow 1e-5 relative / 1e-7 absolute
error; calibrated probabilities allow 1e-5 relative / 1e-6 absolute error.
The uncertainty flags and ranking order must still match exactly.
These checks never alter training, calibration, cutoffs or saved results.
"""

import numpy as np


def check_meta_replay(raw, probability, flags, expected_raw, expected_probability, expected_flags):
    raw, probability, expected_raw, expected_probability = [
        np.asarray(x, dtype=float) for x in (raw, probability, expected_raw, expected_probability)
    ]
    if not all(
        np.isfinite(x).all() for x in [raw, probability, expected_raw, expected_probability]
    ):
        raise ValueError("Non-finite meta-model replay score")
    np.testing.assert_allclose(raw, expected_raw, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(probability, expected_probability, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(flags, expected_flags)
    return dict(
        raw_max_absolute_difference=float(np.max(np.abs(raw - expected_raw))),
        probability_max_absolute_difference=float(
            np.max(np.abs(probability - expected_probability))
        ),
        flags_identical=True,
    )


def check_ranking_replay(actual, expected):
    """Require identical order, including stable handling of tied scores."""
    np.testing.assert_array_equal(
        np.argsort(np.asarray(actual), kind="stable"),
        np.argsort(np.asarray(expected), kind="stable"),
    )
    return True
