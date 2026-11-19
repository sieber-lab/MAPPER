"""Compound-level preprocessing metrics with matched empirical bootstrap draws."""

import numpy as np
from sklearn.metrics import f1_score, average_precision_score

N_BOOTSTRAPS = 10000
SEED = 20260918


def metric_values(scores, truth):
    ranks = np.argsort(-scores, axis=1, kind="stable")
    target = np.eye(scores.shape[1], dtype=int)[truth]
    return dict(
        top1_accuracy=float(np.mean(ranks[:, 0] == truth)),
        top2_accuracy=float(np.mean((ranks[:, :2] == truth[:, None]).any(axis=1))),
        macro_f1=float(
            f1_score(
                truth,
                ranks[:, 0],
                labels=np.arange(scores.shape[1]),
                average="macro",
                zero_division=0,
            )
        ),
        pairwise_pr_auc=float(average_precision_score(target.ravel(), scores.ravel())),
    )


def bootstrap_weights(truth, n=N_BOOTSTRAPS, seed=SEED):
    rng = np.random.default_rng(seed)
    weights = np.zeros((n, len(truth)), dtype=np.int16)
    for label in np.unique(truth):
        ids = np.flatnonzero(truth == label)
        draws = rng.choice(ids, size=(n, len(ids)), replace=True)
        np.add.at(weights, (np.arange(n)[:, None], draws), 1)
    assert (weights.sum(axis=1) == len(truth)).all()
    return weights


def weighted_metrics(scores, truth, weights):
    """Exact duplication-equivalent metrics, including tied-score weighted AP."""
    weights = weights.astype(float)
    nclass = scores.shape[1]
    ranks = np.argsort(-scores, axis=1, kind="stable")
    total = weights.sum(axis=1)
    correct = ranks[:, 0] == truth
    top2 = (ranks[:, :2] == truth[:, None]).any(axis=1)
    true_flags = np.eye(nclass)[truth]
    pred_flags = np.eye(nclass)[ranks[:, 0]]
    tp = weights @ (true_flags * correct[:, None])
    denominator = weights @ (true_flags + pred_flags)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0).mean(axis=1)
    # Each original compound's multiplicity is shared across its nine pairs.
    flat_scores = scores.ravel()
    order = np.argsort(-flat_scores, kind="stable")
    ordered_truth = true_flags.ravel()[order]
    ordered_compounds = np.repeat(np.arange(len(truth)), nclass)[order]
    ends = np.r_[np.flatnonzero(np.diff(flat_scores[order]) != 0), len(order) - 1]
    ap = np.zeros(len(weights))
    for start in range(0, len(weights), 1000):
        stop = min(start + 1000, len(weights))
        w = weights[start:stop, ordered_compounds]
        cumulative_weight = np.cumsum(w, axis=1)[:, ends]
        cumulative_tp = np.cumsum(w * ordered_truth, axis=1)[:, ends]
        delta_tp = np.diff(cumulative_tp, axis=1, prepend=0)
        precision = np.divide(
            cumulative_tp,
            cumulative_weight,
            out=np.zeros_like(cumulative_tp),
            where=cumulative_weight > 0,
        )
        ap[start:stop] = (delta_tp * precision).sum(axis=1) / total[start:stop]
    return np.column_stack([weights @ correct / total, weights @ top2 / total, f1, ap])
