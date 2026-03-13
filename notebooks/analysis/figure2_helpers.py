from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, FuncFormatter

from analysis.plot_style import apply_style, style_axes, save_figure

# Fixed 9-color MoA palette (ordered by class frequency mapping)
COLORS = [
    "#264653", "#287271", "#2a9d8f", "#8ab17d",
    "#e9c46a", "#efb366", "#f4a261", "#ee8959", "#e76f51",
]


def short_moa(s: str) -> str:
    mapping = {
        "DNA replication": "DNA repl.",
        "Cell wall biosynthesis": "Cell wall",
        "Protein synthesis": "Protein syn.",
        "Membrane disruption": "Membrane",
        "Folic acid metabolism": "Folic acid",
        "lipopolysaccharide synthesis": "LPS synth.",
        "Fatty acid synthesis": "Fatty acid",
        "RNA Synthesis": "RNA synth.",
        "Nitrofurans": "Nitrofurans",
    }
    return mapping.get(str(s), str(s))


def moa_palette(moa_order=None):
    """
    If a fixed palette from the dendrogram is available (MOA_COLOR_MAP_FIXED),
    use it; otherwise fall back to assigning COLORS in the given order.
    """
    global MOA_COLOR_MAP_FIXED
    if (
        "MOA_COLOR_MAP_FIXED" in globals()
        and isinstance(MOA_COLOR_MAP_FIXED, dict)
        and len(MOA_COLOR_MAP_FIXED)
    ):
        if moa_order is None:
            moa_order = list(MOA_COLOR_MAP_FIXED.keys())
        return {m: MOA_COLOR_MAP_FIXED.get(m, "#bdbdbd") for m in moa_order}

    if moa_order is None:
        moa_order = []
    return {m: COLORS[i % len(COLORS)] for i, m in enumerate(moa_order)}


def compute_mic_from_matrix(mic_df: pd.DataFrame, thresh: float = 0.1) -> pd.Series:
    """
    mic_df: rows = samples/replicates, cols = concentrations (numeric or strings),
    values = density (0..1). Returns a Series of MIC values (first conc with
    density < thresh), NaN if never reached.
    """
    conc = pd.Index([float(c) for c in mic_df.columns], name="concentration")
    df = mic_df.copy()
    df.columns = conc
    df = df.loc[:, sorted(df.columns)]
    df = df.drop_duplicates()

    def row_mic(series: pd.Series, thresh: float) -> Optional[float]:
        s = pd.to_numeric(series, errors="coerce")
        mask = s < thresh
        if mask.any():
            first_idx = np.flatnonzero(mask.values)[0]
            return float(s.index[first_idx])
        return np.nan

    return df.apply(row_mic, axis=1, thresh=thresh)


def _labels_to_series(y_labels: pd.Series | pd.DataFrame, label_col: Optional[str]) -> pd.Series:
    if isinstance(y_labels, pd.DataFrame):
        if label_col is None:
            for cand in ["mode_of_action", "moa", "label", "y"]:
                if cand in y_labels.columns:
                    label_col = cand
                    break
            if label_col is None:
                raise ValueError("Set `label_col` to the MoA column in y_labels.")
        return y_labels[label_col].astype(str)
    return y_labels.astype(str)


def plot_mic_by_moa(
    mic_data: pd.DataFrame,
    y_labels: pd.Series | pd.DataFrame,
    ecfp_unique: pd.DataFrame | None = None,
    label_col: Optional[str] = None,
    thresh: float = 0.1,
    title: Optional[str] = None,
    savepath: Optional[str | Path] = "figure2B_mic_by_moa",
    show_points: str = "outliers",  # "all" | "outliers" | "none"
    whis: float = 1.5,
):
    apply_style()

    mic_series = compute_mic_from_matrix(mic_data, thresh=thresh)
    mic_series.name = "MIC"

    y_series = _labels_to_series(y_labels, label_col)

    idx = mic_series.index.intersection(y_series.index)
    if ecfp_unique is not None:
        idx = idx.intersection(ecfp_unique.index)

    mic_series = mic_series.loc[idx]
    y_series = y_series.loc[idx]

    dfp = pd.DataFrame({"MIC": mic_series, "MoA": y_series})
    dfp["MoA_short"] = dfp["MoA"].map(short_moa)
    dfp = dfp.dropna(subset=["MIC"])

    counts = Counter(dfp["MoA_short"])
    moa_order = [m for m, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)]
    pal = moa_palette(moa_order)

    use_log = False
    if (dfp["MIC"] > 0).any():
        mi = dfp.loc[dfp["MIC"] > 0, "MIC"].min()
        ma = dfp["MIC"].max()
        if (ma / mi) >= 100:
            use_log = True

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sns.boxplot(
        data=dfp,
        x="MoA_short",
        y="MIC",
        order=moa_order,
        whis=whis,
        fliersize=2,
        showfliers=(show_points != "outliers"),
        palette=[pal[m] for m in moa_order],
        ax=ax,
    )

    if show_points == "all":
        sns.stripplot(
            data=dfp,
            x="MoA_short",
            y="MIC",
            order=moa_order,
            color="k",
            alpha=0.25,
            size=2,
            jitter=0.15,
            ax=ax,
        )
    elif show_points == "outliers":
        outs = []
        for m, g in dfp.groupby("MoA_short"):
            q1 = g["MIC"].quantile(0.25)
            q3 = g["MIC"].quantile(0.75)
            iqr = q3 - q1
            lo = q1 - whis * iqr
            hi = q3 + whis * iqr
            og = g[(g["MIC"] < lo) | (g["MIC"] > hi)]
            if not og.empty:
                outs.append(og)
        if outs:
            outs = pd.concat(outs, ignore_index=True)
            sns.stripplot(
                data=outs,
                x="MoA_short",
                y="MIC",
                order=moa_order,
                color="k",
                alpha=0.6,
                size=2.5,
                jitter=0.15,
                ax=ax,
            )

    ax.set_xlabel("")
    ax.set_ylabel("MIC (log scale)" if use_log else "MIC")
    ax.set_title(title or "")
    if use_log:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=60, ha="right")
    style_axes(ax)

    plt.tight_layout()
    if savepath:
        save_figure(fig, savepath)
    return dfp


def plot_ecfp_embedding(
    ecfp_unique: pd.DataFrame,
    y_labels: pd.Series | pd.DataFrame,
    label_col: Optional[str] = None,
    use_umap: bool = True,
    draw_hulls: bool = True,
    show_counts_in_legend: bool = False,
    title: Optional[str] = None,
    savepath: Optional[str | Path] = "figure2C_structural_space",
):
    apply_style()

    y_series = _labels_to_series(y_labels, label_col)
    common_idx = ecfp_unique.index.intersection(y_series.index)
    X = np.asarray(ecfp_unique.loc[common_idx], dtype=float)
    y = y_series.loc[common_idx].map(short_moa)

    if X.size == 0:
        raise ValueError("No overlap between ecfp_unique and y_labels after alignment.")

    method = "UMAP"
    try:
        if use_umap:
            import umap

            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            emb = reducer.fit_transform(X)
        else:
            raise ImportError
    except Exception:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        emb = pca.fit_transform(X)
        method = "PCA"

    plot_df = pd.DataFrame(emb, columns=[f"{method}1", f"{method}2"], index=common_idx)
    plot_df["MoA"] = y.values

    counts = Counter(plot_df["MoA"])
    if "MOA_ORDER_FIXED" in globals() and MOA_ORDER_FIXED:
        moa_order = [m for m in MOA_ORDER_FIXED if m in plot_df["MoA"].unique()]
    else:
        moa_order = [m for m, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)]
    pal = moa_palette(moa_order)

    try:
        from scipy.spatial import ConvexHull

        has_scipy = True
    except Exception:
        has_scipy = False

    def draw_convex_hull(ax, x, y, color, alpha=0.08, lw=0.7):
        if (not has_scipy) or len(x) < 3:
            return
        pts = np.column_stack([x, y])
        try:
            hull = ConvexHull(pts)
            poly = pts[hull.vertices]
            ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, edgecolor="none")
            ax.plot(poly[:, 0], poly[:, 1], color=color, lw=lw, alpha=alpha * 1.5)
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    if draw_hulls:
        for moa in moa_order:
            sub = plot_df.loc[plot_df["MoA"] == moa]
            draw_convex_hull(
                ax,
                sub[f"{method}1"].values,
                sub[f"{method}2"].values,
                pal[moa],
            )

    for moa in moa_order:
        sub = plot_df.loc[plot_df["MoA"] == moa]
        label = f"{moa} (n={len(sub)})" if show_counts_in_legend else moa
        ax.scatter(
            sub[f"{method}1"],
            sub[f"{method}2"],
            s=28,
            c=[pal[moa]],
            edgecolors="white",
            linewidths=0.5,
            alpha=0.95,
            label=label,
        )

    ax.set_xlabel(f"{method}1")
    ax.set_ylabel(f"{method}2")
    ax.set_title(title or "")
    ax.grid(True, linestyle=":", alpha=0.3)
    style_axes(ax)

    ax.margins(x=0.05, y=0.05)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    handles, labels = ax.get_legend_handles_labels()
    ncols = 4 if len(labels) <= 12 else 5
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=ncols, frameon=False, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    if savepath:
        save_figure(fig, savepath)
    plt.subplots_adjust(bottom=0.22)

    return plot_df


def plot_mic_by_moa_violin(
    mic_data: pd.DataFrame,
    y_labels: pd.Series | pd.DataFrame,
    ecfp_unique: pd.DataFrame | None = None,
    label_col: str | None = None,
    thresh: float = 0.1,
    title: str | None = "2B  MIC distribution by MoA",
    savepath: str | Path | None = "figure2B_mic_by_moa_violin",
    bw_adjust: float = 1.15,
    cut: float = 0.2,
):
    apply_style()

    mic_series = compute_mic_from_matrix(mic_data, thresh=thresh).rename("MIC")
    y_series = _labels_to_series(y_labels, label_col)

    idx = mic_series.index.intersection(y_series.index)
    if ecfp_unique is not None:
        idx = idx.intersection(ecfp_unique.index)

    dfp = pd.DataFrame({"MIC": mic_series.loc[idx], "MoA": y_series.loc[idx]}).dropna(subset=["MIC"])
    dfp["MoA_short"] = dfp["MoA"].map(short_moa)

    dfp = dfp[dfp["MIC"] > 0].copy()
    dfp["MIC_log10"] = np.log10(dfp["MIC"])

    counts = Counter(dfp["MoA_short"])
    if "MOA_ORDER_FIXED" in globals() and MOA_ORDER_FIXED:
        moa_order = [m for m in MOA_ORDER_FIXED if m in dfp["MoA_short"].unique()]
    else:
        moa_order = [m for m, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)]
    pal = moa_palette(moa_order)

    ymin = np.floor(dfp["MIC_log10"].min())
    ymax = np.ceil(dfp["MIC_log10"].max())
    major_ticks = np.arange(ymin, ymax + 1)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sns.violinplot(
        data=dfp,
        x="MoA_short",
        y="MIC_log10",
        order=moa_order,
        palette=[pal[m] for m in moa_order],
        inner=None,
        cut=cut,
        scale="width",
        bw_adjust=bw_adjust,
        linewidth=0.8,
        ax=ax,
        zorder=1,
    )

    sns.swarmplot(
        data=dfp,
        x="MoA_short",
        y="MIC_log10",
        order=moa_order,
        dodge=False,
        facecolor="white",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.95,
        size=3.2,
        ax=ax,
        zorder=3,
    )

    ax.set_xlabel("")
    ax.set_ylabel("MIC [µM]")
    ax.set_title(title or "")

    ax.set_yticks(major_ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{int(v):d}}}$"))
    ax.set_ylim(major_ticks.min(), major_ticks.max())
    ax.grid(True, axis="y", alpha=0.3)

    ax.tick_params(axis="x", which="both", labelbottom=False)
    ax.set_xticks([])
    style_axes(ax)

    plt.tight_layout()
    if savepath:
        save_figure(fig, savepath)
    return dfp
