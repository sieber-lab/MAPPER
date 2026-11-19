"""Original 15-feature/C=1 comparator under the shared nested calibration."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from . import meta as nested

FEATURE_SET = "all15"
C = 1.0
apply_fitted = nested.apply_fitted


def fit_fixed(main_table, calibration_pairs):
    """The shared select_and_fit procedure without model/feature selection."""
    reference = pd.concat([test for _, test in calibration_pairs], ignore_index=True)
    if reference.duplicated(["CompoundID", "role"]).any():
        raise ValueError("Duplicate compound-role calibration rows")
    scores, inner_models = [], []
    for train, test in calibration_pairs:
        if set(train.CompoundID) & set(test.CompoundID):
            raise ValueError("Calibration compound entered meta fitting")
        model = nested.fit_raw(train, FEATURE_SET, C)
        scores.append(model.decision_function(nested.matrix(test, FEATURE_SET)))
        inner_models.append(model)
    raw = np.concatenate(scores)
    correct_id = reference.role.eq("id") & reference.correct.eq(1)
    raw_threshold = nested.threshold(raw, correct_id)
    sigmoid = nested.fit_sigmoid(raw, reference.error)
    baseline_thresholds = {
        "low_max_score": nested.threshold(-reference.PredScore, correct_id),
        "low_margin": nested.threshold(-reference.Margin, correct_id),
    }
    model = nested.fit_raw(main_table, FEATURE_SET, C)
    reference = reference.copy()
    reference["cross_fitted_raw_risk"] = raw
    reference["fitted_calibration_probability"] = sigmoid.predict(raw)
    diagnostic = dict(
        feature_set=FEATURE_SET, C=C, inner_ap=float(average_precision_score(reference.error, raw))
    )
    fitted = dict(
        model=model,
        feature_set=FEATURE_SET,
        C=C,
        sigmoid=sigmoid,
        raw_threshold=raw_threshold,
        probability_threshold=float(sigmoid.predict(raw_threshold)),
        baseline_thresholds=baseline_thresholds,
        selection=[diagnostic],
        calibration_table=reference,
    )
    return fitted, inner_models
