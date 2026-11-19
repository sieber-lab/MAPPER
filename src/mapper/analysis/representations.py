"""Protein projection, nominal selection and control-relative input preparation."""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, t


def observed_mean(values, axis=0):
    count = np.isfinite(values).sum(axis=axis)
    mean = np.divide(
        np.nansum(values, axis=axis), count, out=np.full(np.shape(count), np.nan), where=count > 0
    )
    return mean, count


def linear_log2(values):
    values = np.asarray(values, float)
    if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
        raise ValueError("Invalid native linear quantities")
    result = np.full(values.shape, np.nan)
    np.log2(values, out=result, where=np.isfinite(values) & (values > 0))
    return result


def nominal_tests(treated, controls):
    """Original MAPPER nominal test, including its single-treatment fallback."""
    treated, controls = np.asarray(treated, float), np.asarray(controls, float)
    nt, nc = np.isfinite(treated).sum(axis=0), np.isfinite(controls).sum(axis=0)
    p = np.ones(treated.shape[1])
    enough = (nt >= 2) & (nc >= 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p[enough] = ttest_ind(
            treated[:, enough], controls[:, enough], axis=0, equal_var=True, nan_policy="omit"
        ).pvalue
        single = (nt == 1) & (nc >= 3)
        if single.any():
            cm, _ = observed_mean(controls[:, single])
            tm, _ = observed_mean(treated[:, single])
            std = np.nanstd(controls[:, single], axis=0, ddof=1)
            stat = np.divide(
                tm - cm, std * np.sqrt(1 + 1 / nc[single]), out=np.zeros_like(cm), where=std > 0
            )
            p[single] = np.where(std > 0, 2 * t.sf(abs(stat), nc[single] - 1), 1.0)
    return np.where(np.isfinite(p), p, 1.0)


def project(values, mapping, genes, priority=None, aggregate="select"):
    """Return samples x genes, with row selection independent of treatment labels."""
    links = mapping.loc[mapping.mapped_genes.ne(""), ["source_row", "mapped_genes"]].copy()
    links["gene"] = links.mapped_genes.str.split(";")
    links = links.explode("gene").drop_duplicates(["source_row", "gene"])
    result = np.full((values.shape[1], len(genes)), np.nan)
    selected = {}
    positions = {str(g): i for i, g in enumerate(genes)}
    if aggregate == "mean":
        for gene, group in links.groupby("gene", sort=False):
            indices = group.source_row.to_numpy(int)
            result[:, positions[gene]], _ = observed_mean(values[indices], axis=0)
            selected[gene] = indices.tolist()
        return result, selected
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        quality = pd.DataFrame(
            dict(
                source_row=np.arange(len(values)),
                coverage=np.isfinite(values).sum(axis=1),
                median=np.nanmedian(values, axis=1),
            )
        )
    sort_columns = []
    if priority is not None:
        for name in priority:
            quality[name] = pd.to_numeric(priority[name], errors="coerce").fillna(-1).to_numpy()
            sort_columns.append(name)
    sort_columns += ["source_row"] if priority is not None else ["coverage", "median", "source_row"]
    links = links.merge(quality, on="source_row", validate="many_to_one")
    links = links.sort_values(
        ["gene"] + sort_columns,
        ascending=[True] + [False] * (len(sort_columns) - 1) + [True],
        kind="stable",
    ).drop_duplicates("gene")
    for row in links.itertuples():
        result[:, positions[row.gene]] = values[int(row.source_row)]
        selected[row.gene] = [int(row.source_row)]
    return result, selected
