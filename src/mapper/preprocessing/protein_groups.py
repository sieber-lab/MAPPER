"""Conservative multi-gene-group exclusion, preserving fixed input positions."""

import numpy as np


def ambiguous_columns(gene_to_group):
    mapping = np.asarray(gene_to_group, dtype=int)
    if mapping.ndim != 1 or np.any(mapping < -1):
        raise ValueError("Expected group indices with -1 for unavailable features")
    valid = mapping >= 0
    groups, counts = np.unique(mapping[valid], return_counts=True)
    ambiguous_groups = groups[counts > 1]
    return np.isin(mapping, ambiguous_groups), ambiguous_groups


def exclude_multigene_groups(genes, gene_to_group):
    genes = np.asarray(genes, dtype=np.float64)
    if genes.ndim != 2 or genes.shape[1] != len(gene_to_group) or not np.isfinite(genes).all():
        raise ValueError("Expected finite zero-filled fixed-gene input")
    mask, _ = ambiguous_columns(gene_to_group)
    out = genes.copy()
    out[:, mask] = 0
    np.testing.assert_array_equal(out[:, ~mask], genes[:, ~mask])
    return out
