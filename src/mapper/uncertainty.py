"""The paper's fixed low-maximum-score rule and internal calibration."""

from __future__ import annotations

import numpy as np


def calibrate_score_cutoff(max_scores, correct, false_alarm_target=0.10):
    """Use correctly classified, compound-held-out calibration predictions only.

    Each score must already be aggregated over profiles, descriptions, TabM
    members and seeds. No test compound or external reference label belongs
    in this calibration table. This is the original risk-quantile rule, with
    risk = -max_score and an uncertainty flag at risk >= threshold.
    """
    scores = np.asarray(max_scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if scores.ndim != 1 or correct.shape != scores.shape:
        raise ValueError("Scores and correctness must be aligned one-dimensional arrays")
    if not np.isfinite(scores).all() or correct.sum() < 5:
        raise ValueError("Calibration requires finite scores and at least five correct cases")
    if not 0 < false_alarm_target < 1:
        raise ValueError("False-alarm target must lie strictly between zero and one")
    risk_quantile = np.quantile(-scores[correct], 1 - false_alarm_target, method="higher")
    return float(-np.nextafter(risk_quantile, np.inf))


def flag_uncertain(max_scores, score_cutoff):
    """Scores at or below the frozen, training-specific cutoff are flagged."""
    scores = np.asarray(max_scores, dtype=float)
    if not np.isfinite(scores).all() or not np.isfinite(score_cutoff):
        raise ValueError("Scores and cutoff must be finite")
    return scores <= score_cutoff
