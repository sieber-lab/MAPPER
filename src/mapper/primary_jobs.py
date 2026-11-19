"""Explicit primary-fit inputs shared by the fixed compound-held-out studies.

The model factory and model training definitions are unchanged. Jobs supply
memberships and candidate classes; this module never invents a new split or
chooses configurations using test outcomes.
"""

from __future__ import annotations

import numpy as np
from .external import pair_inputs


def prepare_job(data, plan, job_key, fields, arm):
    job = plan["jobs"][job_key]
    compounds = np.asarray(data[fields["compound_ids"]]).astype(str)
    profiles = np.asarray(data[fields["profile_ids"]]).astype(str)
    labels = np.asarray(data[fields["labels"]]).astype(str)
    candidates = np.asarray(job.get("candidates", plan["classes"])).astype(str)
    text_classes = np.asarray(data[fields["text_classes"]]).astype(str)
    train_ids, test_ids = set(job["train"]), set(job["predict"])
    if train_ids & test_ids or not (train_ids | test_ids) <= set(compounds):
        raise ValueError("Invalid or overlapping compound memberships")
    train, test = np.isin(compounds, list(train_ids)), np.isin(compounds, list(test_ids))
    if set(labels[train]) != set(candidates):
        raise ValueError("Training labels and candidate classes differ")
    if (
        len(profiles) != len(compounds)
        or len(labels) != len(compounds)
        or len(set(profiles)) != len(profiles)
    ):
        raise ValueError("Profile IDs, compound IDs and labels must be aligned")
    text_indices = np.flatnonzero(np.isin(text_classes, candidates))
    if set(text_classes[text_indices]) != set(candidates):
        raise ValueError("Missing candidate descriptions")
    text = data[fields["text"]][text_indices]
    blocks = [np.asarray(data[key]) for key in fields["feature_blocks"][arm]]
    if any(block.ndim != 2 or len(block) != len(compounds) for block in blocks):
        raise ValueError("Misaligned feature blocks")
    values = np.column_stack(blocks)
    raw_train, raw_test = pair_inputs(values[train], text), pair_inputs(values[test], text)
    target = (labels[train, None] == text_classes[None, text_indices]).astype(int).ravel()
    if not np.isfinite(raw_train).all() or not np.isfinite(raw_test).all():
        raise ValueError("Non-finite model input")
    return dict(
        train=raw_train,
        test=raw_test,
        target=target,
        train_profile_indices=np.flatnonzero(train),
        test_profile_indices=np.flatnonzero(test),
        text_indices=text_indices,
        train_compounds=sorted(train_ids),
        test_compounds=sorted(test_ids),
        candidate_classes=candidates.tolist(),
        feature_width=raw_train.shape[1],
    )
