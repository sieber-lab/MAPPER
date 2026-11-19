"""Proteomic/chemical nearest-profile baselines and compound aggregation."""

import numpy as np


def pearson_similarity(test, train):
    test = np.asarray(test, dtype=float)
    train = np.asarray(train, dtype=float)
    a = test - test.mean(axis=1, keepdims=True)
    b = train - train.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
    result = np.divide(a @ b.T, denom, out=np.zeros(denom.shape), where=denom > 0)
    return np.clip(result, -1, 1)


def tanimoto(test, train):
    a, b = np.asarray(test, dtype=float), np.asarray(train, dtype=float)
    assert np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all()
    intersection = a @ b.T
    union = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0)


def class_max(similarities, train_labels, n_classes=9):
    assert set(train_labels) == set(range(n_classes))
    return np.column_stack(
        [similarities[:, train_labels == c].max(axis=1) for c in range(n_classes)]
    )


def grouped_means(matrix, compound_ids, order):
    return np.stack([matrix[compound_ids == cid].mean(axis=0) for cid in order])
