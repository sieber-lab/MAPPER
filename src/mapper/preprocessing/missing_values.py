"""Fold-fitted, protein-group-level imputations; no label inputs."""

from __future__ import annotations

import numpy as np

ARMS = ("zero", "low_percentile", "shifted_gaussian")
SEED = 20260918


def fit_parameters(training):
    training = np.asarray(training, dtype=np.float64)
    assert training.ndim == 2 and not np.isinf(training).any()
    observed = training[np.isfinite(training)]
    assert len(observed) and observed.std() > 0
    mean, sd = float(observed.mean()), float(observed.std(ddof=0))
    return dict(
        n_training_profiles=len(training),
        n_observed_values=len(observed),
        supported_groups=np.isfinite(training).any(axis=0),
        low_percentile=float(np.quantile(observed, 0.01, method="linear")),
        observed_mean=mean,
        observed_sd=sd,
        gaussian_mean=mean - 1.8 * sd,
        gaussian_sd=0.3 * sd,
    )


def transform(raw, parameters, arm, seed):
    if arm not in ARMS:
        raise ValueError(arm)
    raw = np.asarray(raw, dtype=np.float64)
    assert raw.ndim == 2 and not np.isinf(raw).any()
    result = raw.copy()
    missing = np.isnan(raw)
    replace = missing & parameters["supported_groups"][None, :]
    result[missing] = 0
    if arm == "low_percentile":
        result[replace] = parameters["low_percentile"]
    elif arm == "shifted_gaussian":
        rng = np.random.default_rng(seed)
        result[replace] = rng.normal(
            parameters["gaussian_mean"], parameters["gaussian_sd"], int(replace.sum())
        )
    assert np.isfinite(result).all()
    assert np.array_equal(result[~missing], raw[~missing])
    return result


def expand_genes(groups, gene_to_group):
    """Fixed original mapping, -1 means no corresponding measured protein group."""
    groups = np.asarray(groups, dtype=np.float64)
    mapping = np.asarray(gene_to_group, dtype=int)
    assert np.all((mapping >= -1) & (mapping < groups.shape[1]))
    result = np.zeros((len(groups), len(mapping)), dtype=np.float64)
    present = mapping >= 0
    result[:, present] = groups[:, mapping[present]]
    return result
