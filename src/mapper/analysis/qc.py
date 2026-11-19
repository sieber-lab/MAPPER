"""Protein-level replicate agreement and coefficients of variation."""

from itertools import combinations
import numpy as np
import pandas as pd


def replicate_statistics(values, metadata, key, role):
    rows, pairs, summaries = [], [], []
    for group, meta in metadata.groupby(key, sort=True):
        ids = meta.sample_id.tolist()
        block = values[ids]
        linear = np.exp2(block)
        observed = block.notna().sum(axis=1)
        cv = linear.std(axis=1, ddof=1) / linear.mean(axis=1)
        complete = observed.eq(len(ids))
        valid = observed.ge(3)
        # The primary summary requires all available biological replicates.
        # >=3 observed is supplied separately, never conflated with complete.
        frame = pd.DataFrame(
            {
                "protein_group": values.index,
                "group": group,
                "role": role,
                "n_replicates": len(ids),
                "n_observed": observed.to_numpy(),
                "complete_across_available_replicates": complete.to_numpy(),
                "median_log2_intensity": block.median(axis=1).to_numpy(),
                "cv_linear": cv.where(valid).to_numpy(),
            }
        )
        rows.append(frame.loc[valid.to_numpy()])
        r_values = []
        for a, b in combinations(ids, 2):
            shared = block[a].notna() & block[b].notna()
            r = block.loc[shared, a].corr(block.loc[shared, b])
            r_values.append(r)
            pairs.append(
                dict(
                    group=group,
                    role=role,
                    sample_a=a,
                    sample_b=b,
                    observed_shared=int(shared.sum()),
                    pearson_log2=r,
                    spearman_log2=block.loc[shared, a].corr(
                        block.loc[shared, b], method="spearman"
                    ),
                )
            )
        summaries.append(
            dict(
                group=group,
                role=role,
                n_replicates=len(ids),
                proteins_complete=int(complete.sum()),
                proteins_observed_at_least_three=int(valid.sum()),
                median_cv_complete=float(cv[complete].median()),
                q25_cv_complete=float(cv[complete].quantile(0.25)),
                q75_cv_complete=float(cv[complete].quantile(0.75)),
                median_cv_at_least_three=float(cv[valid].median()),
                fraction_complete_cv_below_20pct=float(cv[complete].lt(0.2).mean()),
                median_pairwise_pearson=float(np.median(r_values)),
                min_pairwise_pearson=float(np.min(r_values)),
            )
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(pairs), pd.DataFrame(summaries)
