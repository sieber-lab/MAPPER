from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.stats.multitest import multipletests


def _collect_balanced_long(
    metric: str,
    base_outdir: Path,
    model_order: list[str],
    features: list[str],
) -> pd.DataFrame:
    """Load per-cycle metrics across models, keeping only balanced cv cycles."""
    frames = []
    for model in model_order:
        path = base_outdir / model / f"{model}_per_cycle_all_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "var_tag" not in df.columns or metric not in df.columns:
            continue
        df = df[df["var_tag"].isin(features)].copy()
        if df.empty:
            continue
        if "model_type" not in df.columns:
            df["model_type"] = model
        if "cv_cycle" not in df.columns:
            if all(c in df.columns for c in ["cv_iter", "fold_idx", "rep_idx"]):
                df["cv_cycle"] = (
                    df[["cv_iter", "fold_idx", "rep_idx"]]
                    .astype(int)
                    .astype(str)
                    .agg("|".join, axis=1)
                )
            else:
                continue
        df["feature_set"] = df["var_tag"].astype(str)
        df["value"] = pd.to_numeric(df[metric], errors="coerce")
        frames.append(df[["model_type", "feature_set", "cv_cycle", "value"]])

    if not frames:
        return pd.DataFrame(columns=["model_type", "feature_set", "cv_cycle", "value"])

    long = pd.concat(frames, ignore_index=True).dropna(subset=["value"])

    # require all model × feature combinations per cycle
    combos = {(m, f) for m in model_order for f in features}
    def _combo_set(sub):
        return set(zip(sub["model_type"], sub["feature_set"]))

    ok_cycles = []
    for cyc, sub in long.groupby("cv_cycle"):
        if _combo_set(sub) == combos:
            ok_cycles.append(cyc)

    if ok_cycles:
        long = long[long["cv_cycle"].isin(ok_cycles)].copy()
    else:
        long = long.iloc[0:0].copy()

    return long


def _posthoc_conover_friedman_vs_best(wide: pd.DataFrame, alpha: float = 0.05):
    """Return (best_label, means, q-value matrix) using Conover–Friedman (BH)."""
    if wide.empty:
        return None, pd.Series(dtype=float), pd.DataFrame()

    means = wide.mean(axis=0).sort_values(ascending=False)
    best_label = means.index[0]
    groups = list(wide.columns)

    try:
        import scikit_posthocs as sp

        ptab = sp.posthoc_conover_friedman(wide, p_adjust="fdr_bh")
        ptab.index.name = ptab.columns.name = "cond"
        qmat = ptab.copy()
    except Exception:
        pairs = []
        raw_p = []
        for a, b in itertools.combinations(groups, 2):
            ab = wide[[a, b]].dropna()
            if ab.empty:
                p = 1.0
            else:
                p = float(pg.wilcoxon(ab[a], ab[b], alternative="two-sided")["p-val"].iloc[0])
            pairs.append((a, b))
            raw_p.append(p)
        raw_p = np.array(raw_p, float)
        _, p_bh, _, _ = multipletests(raw_p, alpha=alpha, method="fdr_bh")
        qmat = pd.DataFrame(
            np.ones((len(groups), len(groups))),
            index=groups,
            columns=groups,
            dtype=float,
        )
        for (a, b), adj in zip(pairs, p_bh):
            qmat.loc[a, b] = adj
            qmat.loc[b, a] = adj

    return best_label, means, qmat


# Backwards-compatible names used by notebooks after refactor.
def _collect_balanced_long_impl(
    metric: str,
    base_outdir: Path,
    model_order: list[str],
    features: list[str],
) -> pd.DataFrame:
    return _collect_balanced_long(metric, base_outdir, model_order, features)


def _posthoc_conover_friedman_vs_best_impl(wide: pd.DataFrame, alpha: float = 0.05):
    return _posthoc_conover_friedman_vs_best(wide, alpha=alpha)
