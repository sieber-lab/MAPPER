"""Main Figures 2--6. All numerical layers are generated from archived tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull

from .style import *
from .artwork import reserve
from .panels import three_training_conditions, q_based_boxes


def embedding(ax, data, *, label_column):
    assert {"UMAP1", "UMAP2"}.issubset(data.columns)
    data = data.assign(moa=data[label_column].map(canonical))
    for moa, color in MOA_COLORS.items():
        subset = data.loc[data.moa.eq(moa), ["UMAP1", "UMAP2"]].to_numpy()
        if len(subset) >= 3:
            try:
                hull = ConvexHull(subset)
                ax.fill(*subset[hull.vertices].T, color=color, alpha=0.10, lw=0)
                ax.plot(
                    *np.vstack([subset[hull.vertices], subset[hull.vertices[0]]]).T,
                    color=color,
                    alpha=0.4,
                    lw=0.7,
                )
            except Exception:
                pass  # collinear hulls do not alter point positions
        ax.scatter(*subset.T, color=color, s=13, edgecolor="white", linewidth=0.3, zorder=3)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
    frame(ax, "both")
    ax.margins(0.08)


def figure2():
    import seaborn as sns
    from . import overview_helpers as original

    fig = plt.figure(figsize=(WIDTH, 6.3))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.10,
        right=0.98,
        top=0.87,
        bottom=0.10,
        height_ratios=[1.05, 1],
        hspace=0.50,
        wspace=0.32,
    )
    ax = fig.add_subplot(grid[0, 0])
    mic = read("overview", "mic_profiles.csv")
    order = [original.short_moa(c) for c in CLASSES]
    sns.violinplot(
        data=mic,
        x="MoA_short",
        y="MIC_log10",
        order=order,
        hue="MoA_short",
        palette=dict(zip(order, MOA_COLORS.values())),
        legend=False,
        density_norm="width",
        inner=None,
        cut=0.2,
        bw_adjust=1.15,
        linewidth=0.8,
        ax=ax,
    )
    for i, c in enumerate(order):
        vals = mic.loc[mic.MoA_short.eq(c), "MIC_log10"].to_numpy()
        ax.scatter(
            i + np.random.default_rng(i).uniform(-0.13, 0.13, len(vals)),
            vals,
            s=9,
            c="white",
            edgecolors="#333333",
            linewidths=0.4,
            zorder=4,
        )
    ticks = np.arange(-2, 4)
    ax.set(yticks=ticks, yticklabels=[f"{10.**x:g}" for x in ticks], ylabel="MIC (µM)", xlabel="")
    # Standardize the displayed class name without changing archived labels.
    ax.set_xticks(
        range(len(order)), ["DNA synthesis" if label == "DNA repl." else label for label in order]
    )
    ax.tick_params(axis="x", rotation=60, labelsize=6.5)
    for text in ax.get_xticklabels():
        text.set_ha("right")
    frame(ax, "y")
    panel(ax, "a", "Minimum inhibitory concentrations")
    nested = grid[0, 1].subgridspec(2, 2, hspace=0.55, wspace=0.4)
    axes = np.array([[fig.add_subplot(nested[i, j]) for j in range(2)] for i in range(2)])
    growth = read("overview", "growth_profiles.csv")
    original.plot_growth_facets_axes(
        axes, growth, moas_to_show=["DNA repl.", "Fatty acid", "Protein syn.", "RNA synth."]
    )
    for a, title in zip(
        axes.flat, ["DNA synthesis", "Fatty acid synthesis", "Protein synthesis", "RNA synthesis"]
    ):
        frame(a, "y")
        a.tick_params(labelsize=6)
        a.set_title(title, fontsize=7, pad=5)
        a.xaxis.label.set_fontsize(7)
        a.yaxis.label.set_fontsize(7)
        for annotation in a.texts:
            annotation.set_fontsize(6)
    from .growth_labels import register

    register(fig, axes, growth, ["DNA repl.", "Fatty acid", "Protein syn.", "RNA synth."])
    pos = axes[0, 0].get_position()
    header_y = ax.get_position().y1 + 0.06 * ax.get_position().height
    fig.text(pos.x0 - 0.04, header_y, "b", fontsize=11, va="bottom")
    c = fig.add_subplot(grid[1, 0])
    d = fig.add_subplot(grid[1, 1])
    embedding(c, read("overview", "ecfp_embedding.csv"), label_column="MoA")
    embedding(d, read("overview", "proteomics_embedding.csv"), label_column="MoA")
    panel(c, "c", "Chemical fingerprint space")
    panel(d, "d", "Proteomic response space")
    moa_legend(fig, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, fontsize=6.8)
    return fig


def encoder_panel(ax, metric="pr_auc"):
    data = read("encoders", "rm_anova_input.csv")
    q = read("encoders", f"tukey_{metric}_pvals_matrix.csv", index_col=0)
    values = data.pivot(index="cv_cycle", columns="method", values=metric)
    order = values.mean().sort_values(ascending=False).index.tolist()
    q_based_boxes(ax, values, q, order=order)
    ax.set_xticks(
        range(len(order)),
        [x.replace("PubChemDeBERTa", "PubChem\nDeBERTa").replace("|", "\n") for x in order],
        rotation=50,
        ha="right",
        fontsize=5.5,
    )
    ax.set_ylabel({"pr_auc": "PR AUC", "roc_auc": "ROC AUC", "mcc": "MCC"}[metric])
    ax.set_ylim(min(0.55, float(values.min().min()) - 0.03), 1.02)
    if metric == "pr_auc":
        ax.axhline(values.mean().max(), color="#1F77B4", lw=0.8, ls="--")
    frame(ax, "y")


def figure3():
    fig = plt.figure(figsize=(WIDTH, 5.2))
    # Use the supplied SVG's actual aspect ratio: an oversized reservation
    # adds invisible padding above and below the workflow artwork.
    art_height = 0.98 * WIDTH * (512 / 2115) / fig.get_size_inches()[1]
    art = fig.add_axes([0.01, 0.635, 0.98, art_height])
    reserve(art, "figure3/figure3_ab.svg")
    fig.text(0.015, 0.988, "a", fontsize=11, va="top")
    fig.text(0.755, 0.988, "b", fontsize=11, va="top")
    grid = fig.add_gridspec(1, 2, left=0.10, right=0.98, bottom=0.19, top=0.53, wspace=0.40)
    c = fig.add_subplot(grid[0])
    encoder_panel(c)
    panel(c, "c", "Text encoder and augmentation")
    d = fig.add_subplot(grid[1])
    three_training_conditions(
        d,
        read("figure3", "metric_source.csv"),
        read("figure3", "pairwise_tests.csv"),
        "macro_f1",
        "Macro-F1",
    )
    panel(d, "d", "Text-based task formulation")
    fig.legend(
        handles=[Patch(facecolor=color, label=arm) for arm, color in CONDITIONS.items()],
        loc="lower center",
        bbox_to_anchor=(0.72, 0.035),
        ncol=1,
        frameon=False,
        fontsize=7,
    )
    fig.legend(
        handles=[
            Patch(facecolor=color, label=label)
            for label, color in [
                ("Highest mean", STAT_COLORS["best"]),
                ("Not significantly different", STAT_COLORS["not different"]),
                ("Significantly lower", STAT_COLORS["worse"]),
            ]
        ],
        loc="center",
        bbox_to_anchor=(0.50, 0.60),
        ncol=3,
        frameon=False,
        fontsize=6,
    )
    return fig


def feature_panel(ax, dots, metric="pr_auc"):
    data = read("model_screen", "tabm/tabm_per_cycle_all_metrics.csv")
    values = data.pivot(index="cv_cycle", columns="var_tag", values=metric)
    order = values.mean().sort_values(ascending=False).index.tolist()
    ax.figure._condition_keys = getattr(ax.figure, "_condition_keys", []) + [
        dict(
            comparison="TabM feature sets",
            metric=metric,
            index=i + 1,
            condition=tag.replace("CellPaint", "InfoAlign"),
        )
        for i, tag in enumerate(order)
    ]
    q = read("model_screen", f"tabm/{metric}/tabm_pairwise_{metric}_pvals_BH.csv", index_col=0)
    q_based_boxes(ax, values, q, order=order)
    ax.axhline(values.mean().max(), color="#1F77B4", ls="--", lw=0.8)
    ax.set(
        ylabel={"pr_auc": "PR AUC", "roc_auc": "ROC AUC", "mcc": "MCC"}[metric],
        xticks=[],
        xlim=(-0.6, len(order) - 0.4),
        ylim=(float(values.min().min()) - 0.025, 1.02),
    )
    frame(ax, "y")
    if dots is not None:
        names = ["Proteomics", "ECFP", "CellPaint", "GC", "MIC"]
        for x, tag in enumerate(order):
            for y, atom in enumerate(names):
                dots.scatter(x, y, s=7, color="#222222" if atom in tag.split("+") else "#EBEBEB")
        dots.set(
            xlim=ax.get_xlim(),
            ylim=(4.7, -0.7),
            yticks=range(5),
            yticklabels=["Proteomics", "ECFP", "InfoAlign", "GC", "MIC"],
            xticks=[],
        )
        dots.tick_params(length=0, labelsize=6)
        dots.grid(False)
        for s in dots.spines.values():
            s.set_visible(False)
        dots._unboxed = True
    return order


def model_panel(ax, metric="pr_auc", legend=False):
    values = read("model_screen", f"cross_model/fold_{metric}.csv", index_col=0)
    q = read("model_screen", f"cross_model/pairwise_{metric}_pvals_BH.csv", index_col=0)
    models = ["rf", "lightgbm", "xgboost", "mlp", "pytorch_mlp", "tabm"]
    # Read the frozen exact condition names rather than infer them from labels.
    ordered = []
    positions = []
    for i, model in enumerate(models):
        cols = [c for c in values.columns if c.startswith(model + "|")]
        if len(cols) != 3:
            raise ValueError("Expected three matched conditions per model")
        for j, c in enumerate(cols):
            ordered.append(c)
            positions.append(i + (j - 1) * 0.25)
    result = q_based_boxes(ax, values, q, order=ordered, offsets=positions, width=0.22)
    ax.figure._condition_keys = getattr(ax.figure, "_condition_keys", []) + [
        dict(
            comparison="Model / feature sets",
            metric=metric,
            index=i + 1,
            condition=tag.replace("CellPaint", "InfoAlign"),
        )
        for i, tag in enumerate(ordered)
    ]
    for i, patch in enumerate(result["boxes"]):
        patch.set_hatch(MODEL_HATCHES[i % 3])
    ax.set(
        xticks=range(6),
        xticklabels=["RF", "LightGBM", "XGBoost", "sklearn\nMLP", "PyTorch\nMLP", "TabM"],
        ylabel={"pr_auc": "PR AUC", "roc_auc": "ROC AUC", "mcc": "MCC"}[metric],
        ylim=(min(0.60, float(values.min().min()) - 0.03), 1.02),
        xlim=(-0.6, 5.6),
    )
    ax.tick_params(axis="x", labelsize=6)
    frame(ax, "y")
    if legend:
        ax.legend(
            handles=[
                Patch(facecolor="white", edgecolor="#333333", hatch=hatch, label=label)
                for hatch, label in zip(
                    MODEL_HATCHES,
                    [
                        "All five modalities",
                        "Proteomics + ECFP + InfoAlign + MIC",
                        "Proteomics + ECFP + InfoAlign",
                    ],
                )
            ],
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0, -0.25),
            fontsize=5.5,
            handlelength=2.5,
            handleheight=1.4,
            labelspacing=0.3,
        )
    return ordered


def modality_scores(ax, identity):
    # These are the original LOO modality predictions, not the newer repeated-CV
    # structural-coverage analysis. Keep their aggregation/SEM unchanged.
    if identity in {"JH27", "JH32", "JH42", "JH61"}:
        data = read(
            "modalities", "extended_same_class_pairs_binary_loo_ablation_moa_grouped_stats.csv"
        )
        data = data.loc[data.CompoundID.eq(identity)]
    elif identity == "JH14":
        data = read("modalities", "JH14_binary_loo_ablation_moa_grouped_stats.csv")
    else:
        data = read("modalities", identity + "_binary_loo_ablation_moa_grouped_stats.csv")
    names = {
        "ecfp_cp": "ECFP +\nInfoAlign",
        "ecfp_cp_prot": "+ Proteomics",
        "all": "All features",
        "all_modalities": "All features",
    }
    keys = data.ConfigKey.drop_duplicates().tolist()
    if len(keys) != 3:
        raise ValueError("Expected three modality arms")
    centers = np.array([0.0, 0.55, 1.10])
    offsets = -0.4 / 2 + (np.arange(9) + 0.5) * (0.4 / 9)
    for j, key in enumerate(keys):
        part = data.loc[data.ConfigKey.eq(key)].copy()
        part["class"] = part.MoA.map(canonical)
        for i, c in enumerate(CLASSES):
            row = part.loc[part["class"].eq(c)].iloc[0]
            x = centers[j] + offsets[i]
            ax.bar(x, row["mean"], width=0.4 / 9, color=MOA_COLORS[c], edgecolor="none", zorder=2)
            ax.errorbar(x, row["mean"], yerr=row["sem"], color="#333333", lw=0.6, capsize=1)
            ax.plot(
                x,
                0.025 * (1 + 0.02 * j),
                marker="o",
                ms=2.5,
                color=MOA_COLORS[c],
                ls="none",
                zorder=5,
            )
    # True-class gains use the exact means shown in the bars.
    truth = canonical(
        read("metadata", "dataset_compounds.csv")
        .set_index("compound_id")
        .loc[identity, "moa_reporting_normalized"]
    )
    true_means = [
        float(data.loc[data.ConfigKey.eq(key) & data.MoA.map(canonical).eq(truth), "mean"].iloc[0])
        for key in keys
    ]
    true_sems = [
        float(data.loc[data.ConfigKey.eq(key) & data.MoA.map(canonical).eq(truth), "sem"].iloc[0])
        for key in keys
    ]
    true_index = list(CLASSES).index(truth)
    true_x = centers + offsets[true_index]
    # Original up-across-down connectors, anchored to the true-class bar tops.
    ax.set_ylim(0, 1.2)
    span = 1.2
    required = 1.2
    for j in range(2):
        left = max(true_means[j] + true_sems[j] + 0.05 * span, 0.045) + (0.08 * span if j else 0)
        right = true_means[j + 1] + true_sems[j + 1] + 0.05 * span
        high = max(left, right) + 0.05 * span
        ax.plot(
            [true_x[j], true_x[j], true_x[j + 1], true_x[j + 1]],
            [left, high, high, right],
            color="black",
            lw=0.9,
            zorder=6,
        )
        ax.plot(true_x[j + 1], right, marker="v", ms=3.8, color="black", clip_on=False, zorder=7)
        delta = true_means[j + 1] - true_means[j]
        ax.text(
            (true_x[j] + true_x[j + 1]) / 2,
            high - 0.014 * span,
            f"Δ{j+1}: {delta:+.2f}",
            ha="center",
            va="top",
            fontsize=6.5,
            color="#2E7D32" if delta >= 0 else "#C62828",
            zorder=7,
        )
        required = max(required, high + 0.10 * span)
    ax.set(
        xticks=centers,
        xticklabels=[names.get(k, k) for k in keys],
        xlim=(-0.35, 1.45),
        ylim=(0, required),
        yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
        ylabel="Mean matching score",
    )
    ax.tick_params(axis="x", labelsize=6)
    frame(ax, None)


def artwork_grid(fig, spec, names, folder, columns=2):
    grid = spec.subgridspec(int(np.ceil(len(names) / columns)), columns, hspace=0.12, wspace=0.12)
    for i, name in enumerate(names):
        inner = grid[i // columns, i % columns].subgridspec(2, 1, height_ratios=[5, 1], hspace=0)
        ax = fig.add_subplot(inner[0])
        reserve(ax, f"{folder}/{name}.svg", chemical_role="reference")
        label = fig.add_subplot(inner[1])
        label.set_axis_off()
        display = {"gentamycin": "Gentamicin", "tobramycine": "Tobramycin"}.get(
            name, name.replace("_", " ").capitalize()
        )
        label.text(0.5, 0.7, display, ha="center", va="top", fontsize=5.5)


def molecular_example(fig, spec, identity, filename, letter, folder="figure4"):
    inner = spec.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.02)
    ax = fig.add_subplot(inner[0])
    reserve(ax, f"{folder}/{filename}.svg", chemical_role="example")
    ax.text(0, 1.04, letter, transform=ax.transAxes, fontsize=11, va="bottom")
    label = fig.add_subplot(inner[1])
    label.set_axis_off()
    statistics = read("similarities", "calculated_annotations.csv")
    row = statistics.loc[statistics.compound_id.eq(identity)].iloc[0]
    label.text(0.5, 0.83, row["name"], ha="center", va="top", fontsize=7)
    label.text(
        0.5,
        0.43,
        f"Max. Tanimoto: {row.tanimoto:.2f}\nMax. MCS: {row.mcs:.2f}",
        ha="center",
        va="top",
        fontsize=6.5,
    )


def figure4():
    from .original_artwork import structure_panel, adjacent_plot, adjacent_header

    fig = plt.figure(figsize=(WIDTH, 9.7))
    grid = fig.add_gridspec(
        3,
        2,
        left=0.10,
        right=0.98,
        bottom=0.04,
        top=0.91,
        height_ratios=[1.9, 1.25, 3.6],
        hspace=0.24,
        wspace=0.32,
    )
    top = grid[0, :].subgridspec(2, 1, height_ratios=[3, 1.2], hspace=0.03)
    a = fig.add_subplot(top[0])
    dots = fig.add_subplot(top[1])
    feature_panel(a, dots)
    panel(a, "a", "Contribution of input modalities")
    next(t for t in a.texts if t.get_text() == "a").set_x(-0.045)
    b = fig.add_subplot(grid[1, 0])
    model_panel(b)
    panel(b, "b", "Model comparison")
    c = fig.add_subplot(grid[1, 1])
    shapley = read("model_screen", "verified_tabm_shapley.csv")
    # Column naming is checked against the verified table during preparation.
    c.barh(
        shapley.feature.str.replace("CellPaint", "InfoAlign"),
        shapley.shapley,
        color=["#264653", "#287271", "#2A9D8F", "#7FB48A", "#E9C46A"][: len(shapley)],
    )
    c.set(xlabel="Shapley contribution")
    c.invert_yaxis()
    frame(c, "x")
    panel(c, "c", "Feature contributions")
    pos = grid[2, :].get_position(fig)
    page_w = fig.get_size_inches()[0] * 72
    art = structure_panel(fig, "main04", [3 / page_w, pos.y0, 299 / page_w, pos.height])
    for identity, letter, name, rectangle, header_y in [
        ("JH14", "f", "Tigecycline", [338.61, 402.20, 503.90, 514.15], 392.37),
        ("JH13", "h", "Chloramphenicol", [338.61, 553.34, 503.90, 665.30], 543.50),
    ]:
        ax = adjacent_plot(fig, art, rectangle)
        modality_scores(ax, identity)
        adjacent_header(fig, art, ax, letter, name, header_y)
    moa_legend(
        fig,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=9,
        fontsize=6,
        handlelength=0.8,
        columnspacing=0.7,
        handletextpad=0.3,
    )
    fig.legend(
        handles=[
            Patch(facecolor=color, label=label)
            for label, color in [
                ("Highest mean", STAT_COLORS["best"]),
                ("Not significantly different", STAT_COLORS["not different"]),
                ("Significantly lower", STAT_COLORS["worse"]),
            ]
        ],
        loc="upper center",
        bbox_to_anchor=(0.55, 0.995),
        ncol=3,
        frameon=False,
        fontsize=6,
    )
    fig.legend(
        handles=[
            Patch(facecolor="white", edgecolor="#333333", hatch=hatch, label=label)
            for hatch, label in zip(
                MODEL_HATCHES,
                [
                    "All five modalities",
                    "Proteomics + ECFP + InfoAlign + MIC",
                    "Proteomics + ECFP + InfoAlign",
                ],
            )
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.55, dots.get_position().y0 - 0.003),
        ncol=3,
        fontsize=5.3,
        handlelength=2.8,
        handleheight=1.5,
        columnspacing=1.4,
    )
    return fig


def figure5():
    from .uncertainty import main_figure

    return main_figure()


def figure6():
    from .external import main_figure

    return main_figure()


def build(number):
    return {2: figure2, 3: figure3, 4: figure4, 5: figure5, 6: figure6}[number]()
