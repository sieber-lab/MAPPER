"""Public modality names without changing historical serialized feature order."""

from __future__ import annotations

import numpy as np

# Never sort blocks after renaming a modality: trained weights are positional.
PUBLIC_TO_STORED = {"InfoAlign": "CellPaint"}
STORED_TO_PUBLIC = {value: key for key, value in PUBLIC_TO_STORED.items()}


def public_name(name):
    return STORED_TO_PUBLIC.get(name, name)


def stored_name(name):
    return PUBLIC_TO_STORED.get(name, name)


def assemble_blocks(blocks, feature_order):
    """Concatenate only in the explicitly stored order, accepting InfoAlign names."""
    normalized = {stored_name(key): np.asarray(value) for key, value in blocks.items()}
    if len(normalized) != len(blocks):
        raise ValueError("Do not provide both the public and stored name for one modality")
    parts = [normalized[stored_name(name)] for name in feature_order]
    if not parts or any(part.ndim != 2 for part in parts):
        raise ValueError("Each feature block must be a two-dimensional array")
    if len({part.shape[0] for part in parts}) != 1:
        raise ValueError("Feature blocks must have the same sample order and row count")
    return np.concatenate(parts, axis=1)
