"""Measured-control disagreement calculations; missing values remain missing."""

import numpy as np
from scipy.stats import rankdata

MIN_SHARED = 50


def observed_mean(values, minimum=1):
    values = np.asarray(values, float)
    count = np.isfinite(values).sum(axis=0)
    result = np.divide(
        np.nansum(values, axis=0),
        count,
        out=np.full(values.shape[1], np.nan),
        where=count >= minimum,
    )
    return result, count


def representatives(mask, group_ids):
    """First eligible gene in frozen order per source protein group."""
    indices = np.flatnonzero(np.asarray(mask, bool) & (np.asarray(group_ids) >= 0))
    _, first = np.unique(np.asarray(group_ids)[indices], return_index=True)
    return indices[np.sort(first)]


def distances(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < MIN_SHARED:
        raise ValueError("Insufficient or mismatched measured protein groups")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Distances must not silently zero-fill missing values")
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        raise ValueError("Constant control-abundance vector")
    rho = float(np.corrcoef(ra, rb)[0, 1])
    difference = a - b
    offset = float(np.median(difference))
    return dict(
        spearman=rho,
        rank_distance=1 - rho,
        median_centered_absolute_difference=float(np.median(np.abs(difference - offset))),
        global_log2_offset=offset,
        protein_groups=len(a),
    )


def leave_group_out(source):
    source = np.asarray(source, float)
    return np.stack(
        [observed_mean(np.delete(source, i, axis=0), minimum=2)[0] for i in range(len(source))]
    )


def score_control(target, source, group_ids, panel):
    source = np.asarray(source, float)
    reference, _ = observed_mean(source, minimum=2)
    use = representatives(panel & np.isfinite(target) & np.isfinite(reference), group_ids)
    if len(use) < MIN_SHARED:
        return None, []
    point = distances(target[use], reference[use])
    point["gene_positions_observed"] = int(
        (panel & np.isfinite(target) & np.isfinite(reference)).sum()
    )
    matched = []
    for i, other_reference in enumerate(leave_group_out(source)):
        idx = representatives(
            panel & np.isfinite(target) & np.isfinite(source[i]) & np.isfinite(other_reference),
            group_ids,
        )
        if len(idx) < MIN_SHARED:
            continue
        # Both distances use exactly the same genes and same source reference.
        foreign = distances(target[idx], other_reference[idx])
        own = distances(source[i, idx], other_reference[idx])
        matched.append(
            dict(
                source_group_index=i,
                protein_groups=len(idx),
                target_rank_distance=foreign["rank_distance"],
                source_rank_distance=own["rank_distance"],
                target_centered_difference=foreign["median_centered_absolute_difference"],
                source_centered_difference=own["median_centered_absolute_difference"],
                rank_excess=foreign["rank_distance"] - own["rank_distance"],
            )
        )
    return point, matched
