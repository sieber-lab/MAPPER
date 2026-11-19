"""Class-size and within-class structural associations used in Extended Data 6."""

import itertools
import numpy as np
from scipy.stats import rankdata

N_PERM = 19999
SEED = 20261022


def tanimoto_matrix(x):
    x = np.asarray(x, dtype=np.int64)
    if x.ndim != 2 or not np.isin(x, [0, 1]).all():
        raise ValueError("Expected binary ECFP")
    intersection = x @ x.T
    union = x.sum(axis=1)[:, None] + x.sum(axis=1)[None, :] - intersection
    if not (union > 0).all():
        raise ValueError("Undefined Tanimoto for empty fingerprints")
    return intersection / union


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / norm) if norm > 1e-14 else np.nan


def remove_group_means(values, groups):
    values, groups = np.asarray(values, float), np.asarray(groups)
    result = values.copy()
    for group in np.unique(groups):
        idx = groups == group
        result[idx] -= result[idx].mean()
    return result


def blocked_permutations(groups, count=N_PERM, seed=SEED):
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    indices = np.tile(np.arange(len(groups)), (count, 1))
    for label in np.unique(groups):
        idx = np.flatnonzero(groups == label)
        indices[:, idx] = idx[np.argsort(rng.random((count, len(idx))), axis=1)]
    return indices


def within_class_association(x, flag, groups, permutations):
    ranked = rankdata(x)
    xr = remove_group_means(ranked, groups)
    yr = remove_group_means(flag, groups)
    raw = correlation(ranked, rankdata(flag))
    effect = correlation(xr, yr)
    if not np.isfinite(effect):
        return dict(
            raw_spearman=raw,
            within_class_rank_r=np.nan,
            p_raw=np.nan,
            estimable=False,
            n_variable_flag_classes=0,
        )
    scale = np.linalg.norm(xr) * np.linalg.norm(yr)
    null = (yr[permutations] @ xr) / scale
    p = (1 + np.count_nonzero(np.abs(null) >= abs(effect) - 1e-12)) / (len(null) + 1)
    variable = sum(np.ptp(np.asarray(flag)[np.asarray(groups) == g]) > 0 for g in np.unique(groups))
    return dict(
        raw_spearman=raw,
        within_class_rank_r=effect,
        p_raw=float(p),
        estimable=True,
        n_variable_flag_classes=int(variable),
    )


def exact_class_association(x, fraction):
    x, y = rankdata(x), rankdata(fraction)
    effect = correlation(x, y)
    if not np.isfinite(effect):
        return dict(spearman=np.nan, p_raw=np.nan, estimable=False)
    # Nine class units; repeated rank assignments from tied sizes are equally
    # replicated, so enumerating all permutations gives the exact tied null.
    pidx = np.array(list(itertools.permutations(range(len(x)))), dtype=np.int16)
    xc, yc = x - x.mean(), y - y.mean()
    null = (xc[pidx] @ yc) / (np.linalg.norm(xc) * np.linalg.norm(yc))
    p = np.mean(np.abs(null) >= abs(effect) - 1e-12)
    return dict(spearman=effect, p_raw=float(p), estimable=True)


def holm(pvalues):
    p = np.asarray(pvalues, float)
    result = np.full(len(p), np.nan)
    eligible = np.flatnonzero(np.isfinite(p))
    order = eligible[np.argsort(p[eligible], kind="stable")]
    result[order] = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(len(order), 0, -1)))
    return result
