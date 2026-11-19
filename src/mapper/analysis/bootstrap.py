"""Paired compound bootstrap and weighted metrics used in the study."""

import numpy as np


def divide(numerator, denominator):
    a, b = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b != 0)


def bootstrap_weights(labels, draws, seed, class_blocks=False):
    """Paired weights: columns are compounds, shared across roles and recipes."""
    labels = np.asarray(labels)
    classes = sorted(set(labels))
    rng = np.random.default_rng(seed)
    if class_blocks:
        blocks = rng.multinomial(len(classes), np.full(len(classes), 1 / len(classes)), size=draws)
        return blocks[:, [classes.index(c) for c in labels]]
    weights = np.zeros((draws, len(labels)), dtype=int)
    for cls in classes:
        indices = np.flatnonzero(labels == cls)
        weights[:, indices] = rng.multinomial(
            len(indices), np.full(len(indices), 1 / len(indices)), size=draws
        )
    return weights


def binary_ranking(y, scores, weights):
    """Vectorized weighted ROC AUC and average precision, including score ties."""
    y, scores, weights = np.asarray(y, int), np.asarray(scores, float), np.asarray(weights, float)
    assert weights.ndim == 2 and weights.shape[1] == len(y)
    assert np.isfinite(scores).all() and set(y).issubset({0, 1})
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(y) - 1]
    total_pos, total_neg = weights @ y, weights @ (1 - y)
    tp = np.cumsum(weights[:, order] * y[order], axis=1)[:, ends]
    fp = np.cumsum(weights[:, order] * (1 - y[order]), axis=1)[:, ends]
    tpr = divide(tp, total_pos[:, None])
    fpr = divide(fp, total_neg[:, None])
    prev_tpr, prev_fpr = (
        np.c_[np.zeros(len(weights)), tpr[:, :-1]],
        np.c_[np.zeros(len(weights)), fpr[:, :-1]],
    )
    auc = np.sum((fpr - prev_fpr) * (tpr + prev_tpr) / 2, axis=1)
    precision = np.nan_to_num(divide(tp, tp + fp), nan=0.0)
    ap = np.sum((tpr - prev_tpr) * precision, axis=1)
    auc[(total_pos == 0) | (total_neg == 0)] = np.nan
    ap[(total_pos == 0) | (total_neg == 0)] = np.nan
    return auc, ap


def weighted_rate(values, mask, weights):
    return divide(weights @ (np.asarray(values, float) * mask), weights @ np.asarray(mask, float))


def macro_f1(y, pred, weights, classes):
    values = []
    for cls in classes:
        true, call = np.asarray(y) == cls, np.asarray(pred) == cls
        numerator = 2 * (weights @ (true & call))
        denominator = weights @ true + weights @ call
        values.append(np.nan_to_num(divide(numerator, denominator), nan=0.0))
    return np.mean(values, axis=0)


def interval(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return (
        (float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975)), len(finite))
        if len(finite)
        else (np.nan, np.nan, 0)
    )
