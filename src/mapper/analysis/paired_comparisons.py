"""Corrected paired repeated-CV tests on identical compound folds."""

import numpy as np
import pandas as pd
from scipy.stats import t

MAPPER = "full_tabm_text"
BASELINES = (
    "proteomic_matching",
    "chemical_similarity",
    "proteomics_rf",
    "proteomics_logistic",
    "class_prior",
)
ENDPOINTS = ("top1_accuracy", "macro_f1")


def corrected_test(differences: np.ndarray, ratios: np.ndarray) -> dict:
    """Two-sided corrected t test; ratios are test/train compound counts."""
    differences = np.asarray(differences, dtype=float)
    ratios = np.asarray(ratios, dtype=float)
    if differences.ndim != 1 or ratios.shape != differences.shape or len(differences) < 2:
        raise ValueError("Expected matching one-dimensional difference and ratio vectors")
    if not np.isfinite(differences).all() or not np.isfinite(ratios).all() or (ratios <= 0).any():
        raise ValueError("Differences must be finite and test/train ratios positive")
    n = len(differences)
    mean = float(differences.mean())
    factor = float(1 / n + ratios.mean())
    se = float(np.sqrt(factor * differences.var(ddof=1)))
    if se == 0:
        if mean != 0:
            raise ValueError("Constant nonzero differences: corrected t test is undefined")
        stat, p, half = 0.0, 1.0, 0.0
    else:
        stat = mean / se
        p = float(2 * t.sf(abs(stat), df=n - 1))
        half = float(t.ppf(0.975, df=n - 1) * se)
    return dict(
        difference_fold=mean,
        corrected_se=se,
        ci_low=mean - half,
        ci_high=mean + half,
        t_stat=stat,
        p_raw=p,
        correction_factor=factor,
        df=n - 1,
        n_paired_folds=n,
    )


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values in their original order."""
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or not np.isfinite(pvalues).all() or ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError("Expected finite p-values within [0, 1]")
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.minimum(
        1, np.maximum.accumulate(pvalues[order] * (len(pvalues) - np.arange(len(pvalues))))
    )
    result = np.empty_like(pvalues)
    result[order] = adjusted
    return result


def validate_folds(frame: pd.DataFrame) -> pd.DataFrame:
    """Require identical, complete 5x5 CV fold layouts for all six arms."""
    columns = {"arm", "iteration", "fold", "n_test", "n_train", *ENDPOINTS}
    if not columns.issubset(frame.columns):
        raise ValueError(f"Missing fold columns: {sorted(columns - set(frame.columns))}")
    selected = frame.loc[frame.arm.isin((MAPPER, *BASELINES))].copy()
    expected = {(i, f) for i in range(5) for f in range(5)}
    for arm in (MAPPER, *BASELINES):
        rows = selected.loc[selected.arm.eq(arm)]
        if len(rows) != 25 or set(zip(rows.iteration, rows.fold)) != expected:
            raise ValueError(f"Expected exactly 25 unique CV folds for {arm}")
        if not rows.n_test.isin([10, 11]).all() or not rows.n_train.eq(51 - rows.n_test).all():
            raise ValueError(f"Unexpected train/test compound counts for {arm}")
        if not rows.groupby("iteration").n_test.sum().eq(51).all():
            raise ValueError(f"Incomplete repetition for {arm}")
        values = rows[list(ENDPOINTS)].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Invalid metrics for {arm}")
    return selected


def compare(folds: pd.DataFrame, repeats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the declared ten-test family and keep every paired difference."""
    folds = validate_folds(folds)
    for arm in (MAPPER, *BASELINES):
        rows = repeats.loc[repeats.arm.eq(arm)]
        if len(rows) != 5 or set(rows.iteration) != set(range(5)):
            raise ValueError(f"Expected five complete-repetition metrics for {arm}")
        values = rows[list(ENDPOINTS)].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Invalid repetition metrics for {arm}")
    mapper = folds.loc[folds.arm.eq(MAPPER)]
    results, differences = [], []
    for baseline in BASELINES:
        paired = mapper.merge(
            folds.loc[folds.arm.eq(baseline)],
            on=["iteration", "fold"],
            suffixes=("_mapper", "_baseline"),
            validate="one_to_one",
        ).sort_values(["iteration", "fold"])
        for count in ("n_test", "n_train"):
            if not paired[f"{count}_mapper"].eq(paired[f"{count}_baseline"]).all():
                raise ValueError(f"Unmatched fold sizes: {baseline}")
        for endpoint in ENDPOINTS:
            delta = (
                paired[f"{endpoint}_mapper"].to_numpy() - paired[f"{endpoint}_baseline"].to_numpy()
            )
            result = corrected_test(delta, paired.n_test_mapper / paired.n_train_mapper)
            mean_mapper = repeats.loc[repeats.arm.eq(MAPPER), endpoint].mean()
            mean_baseline = repeats.loc[repeats.arm.eq(baseline), endpoint].mean()
            results.append(
                dict(
                    mapper=MAPPER,
                    baseline=baseline,
                    metric=endpoint,
                    mapper_repetition_mean=mean_mapper,
                    baseline_repetition_mean=mean_baseline,
                    difference_repetition=mean_mapper - mean_baseline,
                    **result,
                )
            )
            for row, value in zip(paired.itertuples(), delta):
                differences.append(
                    dict(
                        baseline=baseline,
                        metric=endpoint,
                        iteration=row.iteration,
                        fold=row.fold,
                        n_test=row.n_test_mapper,
                        n_train=row.n_train_mapper,
                        difference=value,
                    )
                )
    results = pd.DataFrame(results)
    results["p_holm_10"] = holm_adjust(results.p_raw.to_numpy())
    results["significant_holm_0_05"] = results.p_holm_10.lt(0.05)
    return results, pd.DataFrame(differences)
