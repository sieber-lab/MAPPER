"""Main Figure 6 and its supporting SI figures, using the 873-protein panel.

Frozen data columns retain their historical gene-identifier keys; display
labels describe protein features and protein coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .style import *
from .layout import left_tick_labels
from .panels import class_score_bars

ARMS = ("proteomics", "ecfp_only", "ecfp_proteomics")
ARM_COLORS = {"proteomics": "#2A9D8F", "ecfp_only": "#A9AFB4", "ecfp_proteomics": "#244A7C"}
ARM_LABELS = {
    "proteomics": "Proteomics",
    "ecfp_only": "ECFP",
    "ecfp_proteomics": "Proteomics + ECFP",
}
MAIN_DATASETS = ("training", "eclipse", "subanovic", "yu")


def labels():
    data = read("external_final", "datasets.csv")
    result = data.drop_duplicates("dataset").set_index("dataset").label.to_dict()
    result.update(st131=result["pxd027164"], ptf2=result["pxd063721"])
    result.update(
        training="In-house training set",
        eclipse="In-house Eclipse",
        inhouse_corramycin="In-house corramycin",
    )
    return result


def coverage(ax, data, column, datasets, *, labels_left=True):
    # The original coverage export used "external" for Subanovic. Normalize
    # its identifier here, without modifying the frozen input table.
    data = data.assign(dataset=data.dataset.replace({"external": "subanovic"}))
    names = labels()
    rng = np.random.default_rng(42)
    for i, key in enumerate(datasets):
        values = (
            pd.to_numeric(data.loc[data.dataset.eq(key), column], errors="coerce")
            .dropna()
            .to_numpy()
        )
        if len(values):
            color = "#9E9E9E" if key in ("training", "inhouse_corramycin") else "#1F77B4"
            box(
                ax,
                [values],
                positions=[i],
                colors=[color],
                horizontal=True,
                points=False,
                width=0.55,
                whis=(0, 100),
            )
            ax.scatter(
                values,
                i + rng.uniform(-0.16, 0.16, len(values)),
                s=10 if key == "ptf2" else 4,
                marker="D" if key == "ptf2" else "o",
                c="#555555",
                alpha=0.35,
                linewidths=0,
                zorder=3,
            )
        else:
            ax.text(
                0.5,
                i,
                "—",
                transform=ax.get_yaxis_transform(),
                ha="center",
                va="center",
                fontsize=8,
            )
    ax.set(
        yticks=range(len(datasets)),
        yticklabels=[names.get(x, x) for x in datasets],
        ylim=(len(datasets) - 0.5, -0.5),
        xlabel="Protein features per profile",
    )
    ax.tick_params(axis="y", length=0, labelleft=labels_left, labelsize=6.5)
    frame(ax, "x")
    return ax


def value_distributions(ax, representation, *, labels_left=False):
    values = read("external_inputs", "matched_value_quantiles.csv")
    medians = read("external_inputs", "matched_profile_medians.csv")
    rng = np.random.default_rng(42)
    for i, key in enumerate(MAIN_DATASETS):
        row = values.loc[values.dataset.eq(key) & values.representation.eq(representation)].iloc[0]
        item = dict(
            med=row["median"], q1=row.q25, q3=row.q75, whislo=row.q05, whishi=row.q95, fliers=[]
        )
        bp = ax.bxp(
            [item],
            positions=[i],
            vert=False,
            widths=0.55,
            patch_artist=True,
            medianprops=dict(color="#222222", lw=0.8),
            boxprops=dict(lw=0.8),
            whiskerprops=dict(lw=0.8),
            capprops=dict(lw=0.8),
            showfliers=False,
        )
        bp["boxes"][0].set_facecolor("#9E9E9E" if key == "training" else "#1F77B4")
        vals = medians.loc[
            medians.dataset.eq(key) & medians.representation.eq(representation), "median"
        ].to_numpy()
        ax.scatter(
            vals,
            i + rng.uniform(-0.14, 0.14, len(vals)),
            color="#555555",
            s=4,
            alpha=0.35,
            zorder=3,
        )
    ax.set(
        yticks=range(4),
        yticklabels=[labels().get(x, x) for x in MAIN_DATASETS],
        ylim=(3.5, -0.5),
        xlabel="Log₂ intensity" if representation == "intensity" else "Log₂ treated / control",
    )
    ax.tick_params(axis="y", labelleft=labels_left, length=0, labelsize=6.5)
    frame(ax, "x")
    if representation == "log2fc":
        ax.axvline(0, color="#AAAAAA", lw=0.8, zorder=1)


def benchmark_calls():
    data = read("external_comparison", "case_comparison_with_similarity.csv")
    assert data.groupby("arm").size().eq(43).all()
    return data


def accuracy_panel(ax, calls):
    groups = [
        ("All cases", np.ones(len(calls), bool)),
        ("Structurally distant", calls.distant_unseen_identity),
        ("Other cases", ~calls.distant_unseen_identity),
    ]
    ax._accuracy_count_labels = []
    for i, (group, mask) in enumerate(groups):
        for j, arm in enumerate(ARMS):
            subset = calls.loc[mask & calls.arm.eq(arm)]
            correct = int(subset.predicted.eq(subset.expected).sum())
            n = len(subset)
            x = i + (j - 1) * 0.22
            accuracy = 100 * correct / n
            ax.scatter(x, accuracy, s=26, marker="D", color=ARM_COLORS[arm], zorder=4)
            # Separate the nearby structure-only and combined counts without
            # moving their data points. Keep low-accuracy labels above the axis.
            below = arm == "proteomics" or (arm == "ecfp_only" and accuracy >= 70)
            label = ax.annotate(
                f"{correct}/{n}",
                xy=(x, accuracy),
                xytext=(0, -7 if below else 6),
                textcoords="offset points",
                color=ARM_COLORS[arm],
                ha="center",
                va="top" if below else "bottom",
                fontsize=5.5,
            )
            ax._accuracy_count_labels.append(label)
    ax.set(
        xticks=range(3),
        xticklabels=["All\n(n=43)", "Structurally\ndistant (n=9)", "Remaining\n(n=34)"],
        ylabel="Top-1 accuracy (%)",
        ylim=(-5, 122),
        yticks=[0, 50, 100],
        xlim=(-0.5, 2.5),
    )
    ax.tick_params(axis="x", labelsize=6)
    frame(ax, "y")


def auxiliary(ax):
    data = read("external_novel", "class_scores.csv")
    wanted = [("hsi6_nominal", "HSI#6 · 30 min"), ("tatrasgap", "TAT-RasGAP")]
    for i, (key, label) in enumerate(wanted):
        part = (
            data.loc[data.arm.eq("ecfp_proteomics") & data.dataset.eq(key)]
            .sort_values("score", ascending=False)
            .head(3)
        )
        if len(part) != 3:
            raise ValueError("Missing unrepresented-MoA example")
        for j, row in enumerate(part.itertuples()):
            candidate = getattr(row, "candidate", getattr(row, "candidate_class", None))
            ax.bar(
                i * 4 + j,
                row.score,
                width=0.7,
                color=MOA_COLORS[candidate],
                edgecolor="white",
                lw=0.3,
            )
        ax.text(
            i * 4 + 1,
            0.96,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.2,
        )
    threshold = float(part.score_cutoff.iloc[0])
    ax.axhline(threshold, color="#444444", ls="--", lw=0.8)
    ticklabels = []
    for key, _ in wanted:
        part = (
            data.loc[data.arm.eq("ecfp_proteomics") & data.dataset.eq(key)]
            .sort_values("score", ascending=False)
            .head(3)
        )
        col = "candidate" if "candidate" in part else "candidate_class"
        ticklabels.extend(SHORT[c] for c in part[col])
    ax.set(
        xticks=[0, 1, 2, 4, 5, 6], xticklabels=ticklabels, ylabel="Matching score", ylim=(0, 0.45)
    )
    ax.tick_params(axis="x", rotation=60, labelsize=5.5)
    frame(ax, "y")
    for t in ax.get_xticklabels():
        t.set_ha("right")


def yu_panel(ax):
    data = read("yu", "yu_paired_predictions.csv")
    cases = sorted(data.case_key.unique())
    for offset, key in zip(np.linspace(-0.10, 0.10, len(cases)), cases):
        part = (
            data.loc[data.case_key.eq(key)]
            .set_index("training_setting")
            .loc[["inhouse_only", "inhouse_plus_yu"]]
        )
        y = part.score_to_cutoff.to_numpy()
        ax.plot([offset, 1 + offset], y, color="#BBBBBB", alpha=0.55, lw=0.5)
        for j, row in enumerate(part.itertuples()):
            ax.scatter(
                j + offset,
                row.score_to_cutoff,
                s=11 if row.correct else 14,
                marker="o" if row.correct else "x",
                color=MOA_COLORS[row.predicted],
                lw=0.7,
                zorder=3,
            )
    ax.axhline(1, color="#444444", ls="--", lw=0.8)
    ax.set(
        xticks=[0, 1],
        xticklabels=["In-house", "+ Yu"],
        xlim=(-0.4, 1.4),
        ylabel="Maximum score / cutoff",
        ylim=(0, 5.8),
        yticks=[0, 1, 2, 3, 4, 5],
    )
    ax.text(
        0, 0.96, "10/19", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5
    )
    ax.text(
        1, 0.96, "13/19", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5
    )
    ax.text(0.5, 0.96, "Correct", transform=ax.transAxes, ha="center", va="top", fontsize=6.5)
    frame(ax, "y")


def study_ranking(calls):
    ranks = read("external_inputs", "source_ranks.csv")
    ranks = ranks.sort_values("display_rank", na_position="last")
    names = labels()
    ranks["label"] = ranks.dataset.map(names).fillna(ranks.label)
    return ranks


def study_distance(fig, rectangle, calls, *, letter="i"):
    x, y, w, h = rectangle
    pieces = [(0.00, 0.205), (0.23, 0.21), (0.48, 0.27), (0.78, 0.22)]
    axes = [fig.add_axes([x + w * left, y, w * width, h]) for left, width in pieces]
    name_ax, dist, acc, counts = axes
    ranks = study_ranking(calls)
    n = len(ranks)
    for ax in axes:
        ax.set_ylim(n - 0.45, -0.55)
    name_ax.set_axis_off()
    counts.set_axis_off()
    ref = ranks.loc[ranks.matched_source_q05.notna()].iloc[0]
    low, high = float(ref.matched_source_q05), float(ref.matched_source_q95)
    all_values = [[], [], []]
    for i, row in enumerate(ranks.itertuples()):
        name_ax.text(1, i, row.label, ha="right", va="center", fontsize=6.5)
        if pd.notna(row.control_distance):
            dist.plot([low, high], [i, i], color="#C8CCCF", lw=3.5, solid_capstyle="round")
            incomplete = pd.isna(row.adjusted_point)
            lo = (
                float(row.adjusted_lower)
                if pd.notna(row.adjusted_lower)
                else float(row.distance_min)
            )
            hi = (
                float(row.adjusted_upper)
                if pd.notna(row.adjusted_upper)
                else float(row.distance_max)
            )
            dist.plot(
                [lo, hi], [i, i], color="#1F77B4", lw=1, ls=(0, (2, 2)) if incomplete else "-"
            )
            dist.scatter(row.ranking_distance, i, color="#1F77B4", s=12, zorder=3)
        else:
            dist.text(0.26, i, "—", ha="center", va="center", fontsize=8)
        for j, (arm, certain) in enumerate(
            [("proteomics", False), ("ecfp_proteomics", False), ("ecfp_proteomics", True)]
        ):
            part = calls.loc[calls.dataset.eq(row.dataset) & calls.arm.eq(arm)]
            if certain:
                part = part.loc[~part.uncertain]
            denom = len(part)
            correct = int(part.predicted.eq(part.expected).sum())
            all_values[j].append(100 * correct / denom if denom else np.nan)
            counts.text(
                j,
                i,
                f"{correct}/{denom}" if denom else "— (0)",
                ha="center",
                va="center",
                fontsize=6.2,
            )
        if i:
            for ax in [dist, acc, counts]:
                ax.axhline(i - 0.5, color="#EEEEEE", lw=0.4, zorder=0)
    for j, (color, style, label) in enumerate(
        [
            ("#2A9D8F", (0, (2, 2)), "Proteomics · all"),
            ("#244A7C", "-", "Proteomics + ECFP · all"),
            ("#E0A458", "-", "Proteomics + ECFP · certain"),
        ]
    ):
        values = np.array(all_values[j])
        valid = np.isfinite(values)
        acc.plot(
            values[valid],
            np.arange(n)[valid],
            color=color,
            ls=style,
            marker="o",
            ms=2.7,
            lw=1,
            label=label,
        )
    dist.set(
        xlim=(-0.015, 0.54), xticks=[0, 0.2, 0.4], yticks=[], xlabel="Experimental disagreement"
    )
    acc.set(xlim=(-4, 104), xticks=[0, 50, 100], yticks=[], xlabel="Top-1 accuracy (%)")
    counts.set(xlim=(-0.5, 2.5), xticks=[])
    for ax in (dist, acc):
        frame(ax, "x")
        ax.tick_params(axis="y", left=False)
    for j, label in enumerate(["All", "All", "Certain"]):
        counts.text(j, n + 0.1, label, ha="center", va="top", fontsize=6)
    counts.text(0, n + 0.85, "Proteomics", ha="center", va="top", fontsize=6)
    counts.text(1.5, n + 0.85, "Proteomics + ECFP", ha="center", va="top", fontsize=6)
    counts.text(1, 1.07, "Correct / total", transform=counts.transAxes, ha="right", fontsize=7)
    fig.text(x, y + h + 0.03, letter, fontsize=11)
    fig.text(x + 0.045, y + h + 0.03, "Prediction across experimental settings", fontsize=9)
    fig.legend(
        *acc.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(x + w * 0.53, y - 0.055),
        ncol=3,
        fontsize=6.5,
        frameon=False,
        columnspacing=1,
    )
    return axes


def main_figure():
    return main_figure_with_representation_comparison()


def main_figure_with_representation_comparison():
    """Approved ten-panel Figure 6, with the matched representation control in f."""
    from .representation_comparison import comparison_panel

    calls = benchmark_calls()
    fig = plt.figure(figsize=(WIDTH, 10.0))
    grid = fig.add_gridspec(
        3,
        6,
        left=0.16,
        right=0.985,
        top=0.945,
        bottom=0.48,
        height_ratios=[0.82, 1, 1.2],
        hspace=0.85,
        wspace=0.85,
    )
    a, b, c = [fig.add_subplot(grid[0, i * 2 : (i + 1) * 2]) for i in range(3)]
    coverage(a, read("external_inputs", "source_coverage.csv"), "retained_genes", MAIN_DATASETS)
    a.set_xticks([0, 1000, 2000])
    a.set_xlabel("Significant protein\nfeatures per profile", fontsize=7)
    panel(a, "a", "Significant-protein coverage")
    training = read("external_inputs", "source_training.csv")
    for y, moa in enumerate(CLASSES):
        vals = training.loc[training.moa.eq(moa), "retained_genes"].to_numpy()
        if len(vals):
            box(
                b,
                [vals],
                positions=[y],
                colors=[MOA_COLORS[moa]],
                horizontal=True,
                width=0.6,
                whis=(0, 100),
            )
    b.set(
        yticks=range(9),
        yticklabels=[SHORT[x] for x in CLASSES],
        ylim=(8.5, -0.5),
        xlabel="Significant protein\nfeatures per profile",
        xticks=[0, 1000, 2000],
    )
    b.xaxis.label.set_fontsize(7)
    b.tick_params(axis="y", labelsize=5.8, length=0)
    frame(b, "x")
    panel(b, "b", "Coverage by MoA")
    panel_data = read("external_inputs", "panel_coverage_profiles.csv")
    coverage(c, panel_data, "observed_genes", MAIN_DATASETS)
    c.set(
        yticklabels=["Training", "Eclipse", "Subanovic\net al.", "Yu et al."],
        xticks=[0, 400, 800],
        xlabel="Protein features\nper profile",
    )
    c.tick_params(axis="y", labelsize=5.8)
    c.xaxis.label.set_fontsize(7)
    panel(c, "c", "Shared 873-protein panel")
    d, e, f = [fig.add_subplot(grid[1, i * 2 : (i + 1) * 2]) for i in range(3)]
    value_distributions(d, "intensity", labels_left=True)
    panel(d, "d", "Measured intensities")
    value_distributions(e, "log2fc")
    panel(e, "e", "Control-relative changes")
    comparison_panel(f)
    panel(f, "f", "Input representation")
    g, h, i = [fig.add_subplot(grid[2, j * 2 : (j + 1) * 2]) for j in range(3)]
    accuracy_panel(g, calls)
    panel(g, "g", "External prediction accuracy")
    yu_panel(h)
    h.set_xticklabels(["In-house", "+ Yu et al."])
    panel(h, "h", "Yu et al. adaptation")
    auxiliary(i)
    panel(i, "i", "Unfamiliar mechanisms")
    for ax in [a, b, c, d, e, f, g, h, i]:
        ax.title.set_fontsize(7.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page_h = fig.bbox.height
    legend_top = min(ax.get_tightbbox(renderer).y0 for ax in [g, h, i]) / page_h - 0.008
    feature_handles = [Patch(facecolor=ARM_COLORS[arm], label=ARM_LABELS[arm]) for arm in ARMS]
    features = fig.legend(
        handles=feature_handles,
        loc="upper center",
        bbox_to_anchor=(g.get_position().x0 + g.get_position().width / 2, legend_top),
        ncol=3,
        frameon=False,
        fontsize=5.6,
        columnspacing=0.8,
        handlelength=1.3,
        handletextpad=0.35,
    )
    fig.legend(
        handles=[
            Line2D([], [], ls="none", marker="o", color="#666666", ms=3, label="Correct"),
            Line2D([], [], ls="none", marker="x", color="#666666", ms=3, label="Incorrect"),
            Line2D([], [], ls="--", color="#666666", label="Cutoff"),
        ],
        loc="upper center",
        bbox_to_anchor=(h.get_position().x0 + h.get_position().width / 2, legend_top),
        ncol=3,
        frameon=False,
        fontsize=5.3,
        columnspacing=0.6,
        handlelength=1.2,
        handletextpad=0.3,
    )
    fig.canvas.draw()
    legend_bottom = features.get_window_extent(renderer).y0 / page_h
    study_top = min(0.35, legend_bottom - 0.055)
    study_distance(fig, [0.025, 0.095, 0.96, study_top - 0.095], calls, letter="j")
    return fig


def representation_figure():
    fig = plt.figure(figsize=(WIDTH, 8.5))
    grid = fig.add_gridspec(
        2,
        6,
        left=0.19,
        right=0.985,
        top=0.93,
        bottom=0.10,
        height_ratios=[1, 2.3],
        hspace=0.45,
        wspace=0.7,
    )
    upper = grid[0, :].subgridspec(1, 2, wspace=0.45)
    a = fig.add_subplot(upper[0])
    b = fig.add_subplot(upper[1])
    draws = read("external_final", "gene_removal_draws.csv")
    caps = [0, 25, 50, 100, 200, 400, 800, -1]
    for key, label, color in [
        ("original_class_retained", "Reference class recovered", "#244A7C"),
        (
            "non_cell_wall_called_cell_wall",
            "Non-CW called cell wall",
            MOA_COLORS["Cell wall biosynthesis"],
        ),
    ]:
        tab = draws.groupby("cap")[key].agg(["mean", "min", "max"]).loc[caps] * 100
        a.plot(range(8), tab["mean"], marker="o", ms=3, color=color, lw=1, label=label)
        a.fill_between(range(8), tab["min"], tab["max"], color=color, alpha=0.15)
    a.set(
        xticks=range(8),
        xticklabels=[str(x) if x >= 0 else "Original" for x in caps],
        ylim=(-3, 103),
        ylabel="Compounds (%)",
        xlabel="Maximum protein features per profile",
    )
    a.tick_params(axis="x", rotation=50, labelsize=6)
    frame(a, "y")
    panel(a, "a", "Reduced protein coverage")
    a.legend(
        frameon=False, loc="upper left", bbox_to_anchor=(0, -0.36), fontsize=6, borderaxespad=0
    )
    intervals = read("external_final", "internal_cv_intervals.csv").set_index("representation")
    keys = ["intensity", "significant_intensity", "fold_change"]
    colors = ["#9E9E9E", "#C45C5C", "#1F77B4"]
    for y, (key, color) in enumerate(zip(keys, colors)):
        row = intervals.loc[key]
        point = row.balanced_accuracy * 100
        b.errorbar(
            point,
            y,
            xerr=[[point - row.ci_lower * 100], [row.ci_upper * 100 - point]],
            fmt="o",
            color=color,
            ms=5,
            capsize=3,
        )
        b.text(min(point + 5, 95), y + 0.15, f"{point:.1f}%", fontsize=6.5, color=color)
    b.set(
        yticks=range(3),
        yticklabels=["4,401 · intensities", "4,401 · significant", "873 · log₂FC"],
        ylim=(2.6, -0.6),
        xlim=(0, 100),
        xlabel="Class-balanced accuracy (%)",
    )
    b.tick_params(axis="y", labelsize=6)
    frame(b, "x")
    panel(b, "b", "Matched in-house CV")
    profiles = read("external_final", "coverage_profiles.csv")
    order = profiles.dataset.drop_duplicates().tolist()
    axes = [fig.add_subplot(grid[1, i * 2 : (i + 1) * 2]) for i in range(3)]
    for i, (ax, column, title) in enumerate(
        zip(
            axes,
            ["detected_reference_genes", "significant_reference_genes", "usable_panel_genes"],
            [
                "All mapped protein features",
                "Significant protein features",
                "Control-supported\n873-protein inputs",
            ],
        )
    ):
        coverage(ax, profiles, column, order, labels_left=i == 0)
        ax.title.set_fontsize(7)
        panel(ax, chr(99 + i), title)
        ax.title.set_fontsize(7.5)
        ax.set_xlim(-30, 4401 if i < 2 else 900)
        ax.set_xticks([0, 2000, 4000] if i < 2 else [0, 400, 800])
    return fig


def class_score_figure():
    scores = read("external_final", "current_class_scores.csv")
    calls = read("external_final", "current_case_predictions.csv")
    order = read("external_final", "heatmap_row_labels.csv")
    cases = order.case_key.tolist()
    if len(cases) != 43:
        raise ValueError("Expected all 43 benchmark cases")
    fig = plt.figure(figsize=(WIDTH, 9.7))
    fig._horizontal_alignment = "left"
    grid = fig.add_gridspec(1, 3, left=0.22, right=0.985, bottom=0.15, top=0.94, wspace=0.12)
    bounds = np.array([0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0])
    cmap = ListedColormap(
        ["#D5E4ED", "#ABCEDC", "#72B1C4", "#3D8FA9", "#20698D", "#17486E", "#102D53"]
    )
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    names = labels()
    groups = order.dataset.tolist()
    display_rows = []
    positions = {}
    headings = []
    for row in order.itertuples():
        if not headings or headings[-1][1] != row.dataset:
            headings.append((len(display_rows), row.dataset))
            display_rows.append(None)
        positions[row.case_key] = len(display_rows)
        display_rows.append(row.case_key)
    nrows = len(display_rows)
    novelty = (
        benchmark_calls()
        .drop_duplicates("case_key")
        .set_index("case_key")
        .distant_unseen_identity.to_dict()
    )
    axes = []
    for j, arm in enumerate(ARMS):
        ax = fig.add_subplot(grid[j])
        axes.append(ax)
        part = scores.loc[scores.arm.eq(arm)]
        table = part.pivot(index="case_key", columns="candidate", values="score").loc[
            cases, list(CLASSES)
        ]
        displayed = np.full((nrows, 9), np.nan)
        for key in cases:
            displayed[positions[key]] = table.loc[key].to_numpy()
        im = ax.imshow(displayed, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        for key in cases:
            y = positions[key]
            row = calls.loc[calls.arm.eq(arm) & calls.case_key.eq(key)].iloc[0]
            reference = list(CLASSES).index(row.expected)
            ax.add_patch(
                Rectangle(
                    (reference - 0.48, y - 0.48), 0.96, 0.96, fill=False, edgecolor="black", lw=0.8
                )
            )
            ranking = part.loc[part.case_key.eq(key)].sort_values("score", ascending=False).head(3)
            for rank, r in enumerate(ranking.itertuples(), 1):
                x = list(CLASSES).index(r.candidate)
                ax.text(
                    x,
                    y,
                    str(rank),
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    fontweight="bold" if rank == 1 else "normal",
                    color="white" if r.score > 0.25 else "#222222",
                )
            if row.uncertain:
                ax.plot(8.9, y, "o", ms=2.2, color="#888888", clip_on=False)
        for y, dataset in headings:
            ax.axhline(y - 0.5, color="#555555", lw=0.65)
            if j == 0:
                from matplotlib.transforms import blended_transform_factory

                ax.text(
                    0.03,
                    y,
                    names[dataset],
                    transform=blended_transform_factory(fig.transFigure, ax.transData),
                    ha="left",
                    va="center",
                    fontsize=6.5,
                    color="#244A7C",
                )
        ax.set(
            xticks=range(9),
            xticklabels=[SHORT[c] for c in CLASSES],
            yticks=[positions[k] for k in cases],
            yticklabels=(
                [
                    label + (" △" if novelty.get(k, False) else "")
                    for k, label in zip(cases, order.display_label)
                ]
                if j == 0
                else []
            ),
            xlim=(-0.5, 9.2),
            ylim=(nrows - 0.5, -0.5),
        )
        ax.tick_params(axis="x", rotation=90, labelsize=6, length=0, pad=4)
        ax.tick_params(axis="y", labelsize=5.5, length=0, pad=5)
        ax.grid(False)
        if j == 0:
            left_tick_labels(ax)
        ax.set_title(ARM_LABELS[arm], fontsize=8, pad=12)
    cax = fig.add_axes([0.35, 0.045, 0.54, 0.012])
    cb = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        boundaries=bounds,
        ticks=[0, 0.1, 0.25, 0.5, 0.75, 1],
        spacing="uniform",
    )
    cb.set_label("Class-matching score", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return fig
