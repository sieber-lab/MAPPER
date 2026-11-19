"""Tie-aware weighted ROC and precision-recall metrics."""

import numpy as np


def ranking_draws(y, risk, weights):
    """Weighted empirical ROC/AP, with ties; tested against sklearn metrics."""
    order = np.argsort(-risk, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(risk[order]) != 0) + 1]
    positive = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
    negative = np.add.reduceat(weights[:, order] * (1 - y[order]), starts, axis=1)
    tp, fp = positive.cumsum(axis=1), negative.cumsum(axis=1)
    if (tp[:, -1] <= 0).any() or (fp[:, -1] <= 0).any():
        raise ValueError("Undefined bootstrap ranking endpoint")
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    ap = ((positive / tp[:, -1, None]) * precision).sum(axis=1)
    tpr, fpr = tp / tp[:, -1, None], fp / fp[:, -1, None]
    previous_tpr = np.c_[np.zeros(len(tpr)), tpr[:, :-1]]
    previous_fpr = np.c_[np.zeros(len(fpr)), fpr[:, :-1]]
    auc = ((fpr - previous_fpr) * (tpr + previous_tpr) / 2).sum(axis=1)
    return {"auroc": auc, "average_precision": ap}
