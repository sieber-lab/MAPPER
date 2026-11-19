"""Original matched configuration-selection calculations, with explicit inputs."""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd

MODELS = ("pytorch_mlp", "lightgbm", "xgboost", "tabm", "mlp", "rf")


def model_family(name):
    for model in MODELS:
        if f"_{model}_" in name:
            return model
    raise ValueError(f"Unrecognized configuration model: {name}")


def ranked_counts(scores, labels, compound_indices, n_compounds):
    """Exact sufficient statistics for weighted AP, including tied scores.

    Only score groups containing a positive row contribute to average precision.
    Prefix counts retain all negative rows. Compound weights can therefore be
    applied without resorting the original rows for every bootstrap draw.
    """
    order = np.argsort(-np.asarray(scores), kind="stable")
    sorted_scores = np.asarray(scores)[order]
    group = np.cumsum(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]) - 1
    shape = (n_compounds, int(group[-1]) + 1)
    total = np.zeros(shape, dtype=np.int32)
    positive = np.zeros(shape, dtype=np.int32)
    np.add.at(total, (np.asarray(compound_indices)[order], group), 1)
    np.add.at(positive, (np.asarray(compound_indices)[order], group), np.asarray(labels)[order])
    used = positive.sum(axis=0) > 0
    return (
        np.cumsum(total, axis=1)[:, used],
        np.cumsum(positive, axis=1)[:, used],
        positive[:, used],
    )


def weighted_ap(counts, weights):
    """One AP per compound-weight vector; weights expand whole compounds."""
    total, positive, increments = counts
    tp = weights @ positive
    denominator = weights @ total
    precision = np.divide(
        tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0
    )
    numerator = np.sum(precision * (weights @ increments), axis=1)
    positives = weights @ increments.sum(axis=1)
    return np.divide(numerator, positives, out=np.full(len(weights), np.nan), where=positives > 0)


def compound_metrics(scores, truth, weights):
    """Metrics per complete repetition, then average metrics (not scores)."""
    order = np.argsort(-scores, axis=2, kind="stable")
    n_classes = scores.shape[2]
    denominator = weights.sum(axis=1)
    accuracy, top2, f1 = [], [], []
    for iteration in range(scores.shape[0]):
        predicted = order[iteration, :, 0]
        accuracy.append((weights @ (predicted == truth)) / denominator)
        top2.append(
            (weights @ np.any(order[iteration, :, :2] == truth[:, None], axis=1)) / denominator
        )
        cells = np.eye(n_classes * n_classes)[truth * n_classes + predicted]
        confusion = (weights @ cells).reshape(len(weights), n_classes, n_classes)
        support = confusion.sum(axis=2) + confusion.sum(axis=1)
        class_f1 = np.divide(
            2 * np.diagonal(confusion, axis1=1, axis2=2),
            support,
            out=np.zeros_like(support),
            where=support > 0,
        )
        f1.append(class_f1.mean(axis=1))
    return dict(
        top1_accuracy=np.mean(accuracy, axis=0),
        top2_accuracy=np.mean(top2, axis=0),
        macro_f1=np.mean(f1, axis=0),
    )


def feature_tag(name):
    features = name.split("_Text_Embedding", 1)[1].strip("+")
    return "+".join(sorted(features.split("+"))) if features else "(Text only)"


def restrict_search_space(originals, search_space):
    """Keep the same complete 32-feature screen for every model family."""
    if search_space == "all_cached":
        return originals.reset_index(drop=True), originals.iloc[:0].copy()
    if search_space != "matched_features":
        raise ValueError(f"Unknown search space: {search_space}")
    keep = originals.configuration.str.startswith("semantic_BioBERT_")
    selected = originals.loc[keep].reset_index(drop=True)
    excluded = originals.loc[~keep].reset_index(drop=True)
    features = ("Proteomics", "ECFP", "CellPaint", "MIC", "GC")
    expected = {
        "+".join(sorted(subset)) if subset else "(Text only)"
        for length in range(6)
        for subset in itertools.combinations(features, length)
    }
    models = ("tabm", "xgboost", "lightgbm", "pytorch_mlp", "mlp", "rf")
    for model in models:
        names = [name for name in selected.configuration if model_family(name) == model]
        if len(names) != 32 or {feature_tag(name) for name in names} != expected:
            raise ValueError(f"Expected the complete 32 matched feature combinations for {model}")
    if len(selected) != 192 or selected.configuration.duplicated().any():
        raise ValueError("Expected 192 unique matched-feature configurations")
    return selected, excluded


def draw_weights(truth, folds, count, seed):
    """Condition draws on all-class and all-fold coverage for exact equal-fold AP."""
    rng = np.random.default_rng(seed)
    inbag, oob, draw_rows = [], [], []
    members = [np.flatnonzero(truth == label) for label in np.unique(truth)]
    fold_members = [
        np.flatnonzero(folds[iteration] == fold) for iteration in range(5) for fold in range(5)
    ]
    for iteration in range(count):
        attempts = 0
        while True:
            attempts += 1
            sampled = np.concatenate(
                [rng.choice(indices, size=len(indices), replace=True) for indices in members]
            )
            weights = np.bincount(sampled, minlength=len(truth))
            omitted = (weights == 0).astype(int)
            if all(omitted[indices].sum() > 0 for indices in members) and all(
                omitted[indices].sum() > 0 and weights[indices].sum() > 0
                for indices in fold_members
            ):
                break
        inbag.append(weights)
        oob.append(omitted)
        draw_rows.append(
            dict(
                bootstrap_replicate=iteration,
                draw_attempts=attempts,
                n_unique_inbag_compounds=int((weights > 0).sum()),
                n_oob_compounds=int(omitted.sum()),
            )
        )
    return np.asarray(inbag, dtype=float), np.asarray(oob, dtype=float), pd.DataFrame(draw_rows)


def evaluate_cache(cache_path, weights, chunk_size):
    result = np.zeros(len(weights))
    with np.load(cache_path, allow_pickle=False) as cached:
        for iteration in range(5):
            for fold in range(5):
                counts = tuple(
                    cached[f"i{iteration}_f{fold}_{key}"].astype(float)
                    for key in ("total", "positive", "increments")
                )
                for start in range(0, len(weights), chunk_size):
                    result[start : start + chunk_size] += (
                        weighted_ap(counts, weights[start : start + chunk_size]) / 25
                    )
        scores = cached["compound_scores"]
    if not np.isfinite(result).all():
        raise ValueError("Empty evaluation fold despite resample coverage check")
    return result, scores
