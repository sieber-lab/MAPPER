"""Fixed original meta-model architecture, retained as an uncertainty comparator."""

from dataclasses import dataclass
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

META_FEATURES = [
    "PredScore",
    "Margin",
    "PredEntropy",
    "ExpEntropy",
    "MeanPairKL",
    "VarMax",
    "AugVar",
    "VariationRatio",
    "KConsistency",
    "SeedStd",
    "EpistemicMI",
    "BioVar",
    "RepMarginStd",
    "RepMarginIQR",
    "RepAgree",
]
FEATURE_SETS = {"all15": META_FEATURES}


def matrix(table, feature_set):
    d = table[FEATURE_SETS[feature_set]].copy().astype(float)
    for c in ["PredScore", "Margin"]:
        d[c] = -d[c]
    x = d.fillna(0.0).to_numpy()
    assert np.isfinite(x).all()
    return x


def fit_raw(table, feature_set, c):
    assert set(table.error) == {0, 1}
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        LogisticRegression(
            max_iter=1000, C=c, class_weight="balanced", solver="lbfgs", random_state=7
        ),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        model.fit(matrix(table, feature_set), table.error.to_numpy(int))
    return model


@dataclass
class MonotoneSigmoid:
    mean: float
    scale: float
    slope: float
    intercept: float

    def predict(self, raw):
        return expit(self.slope * (np.asarray(raw) - self.mean) / self.scale + self.intercept)


def fit_sigmoid(raw, y):
    raw, y = np.asarray(raw, float), np.asarray(y, int)
    assert np.isfinite(raw).all() and set(y) == {0, 1}
    mean, scale = float(raw.mean()), max(float(raw.std()), 1e-8)
    x = (raw - mean) / scale

    def objective(params):
        a, b = params
        z = a * x + b
        loss = np.mean(np.logaddexp(0.0, z) - y * z)
        residual = expit(z) - y
        return loss, np.array([(residual * x).mean(), residual.mean()])

    initial = [1.0, float(np.log(y.mean() / (1 - y.mean())))]
    fit = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None), (None, None)],
        options={"maxiter": 1000},
    )
    assert fit.success, fit.message
    return MonotoneSigmoid(mean, scale, float(fit.x[0]), float(fit.x[1]))


def threshold(raw, correct_id, target=0.1):
    values = np.asarray(raw, float)[np.asarray(correct_id, bool)]
    assert (
        len(values) >= 5 and np.isfinite(values).all()
    ), "Too few valid correct-ID calibration cases"
    return float(np.nextafter(np.quantile(values, 1 - target, method="higher"), np.inf))


def apply_fitted(fitted, table):
    raw = fitted["model"].decision_function(matrix(table, fitted["feature_set"]))
    result = table.copy()
    result["raw_risk"] = raw
    result["p_error"] = fitted["sigmoid"].predict(raw)
    # Use raw comparison to preserve quantile tie handling even when sigmoid
    # probabilities saturate to exactly 0 or 1 in machine precision.
    result["uncertain"] = raw >= fitted["raw_threshold"]
    result["uncertain_low_max_score"] = (
        -table.PredScore >= fitted["baseline_thresholds"]["low_max_score"]
    )
    result["uncertain_low_margin"] = -table.Margin >= fitted["baseline_thresholds"]["low_margin"]
    return result
