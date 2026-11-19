"""Supporting figures. Numerical definitions and saved test families are fixed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .style import *
from .panels import three_training_conditions, significance_matrix
from .layout import plain_colorbar, left_tick_labels


def qc():
    profiles = read("qc", "development_profiles.csv")
    summaries = read("qc", "replicate_summaries.csv")
    scores = read("qc", "pca_scores.csv")
    variance = read("qc", "pca_variance.csv")
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 6.8))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.94, bottom=0.13, hspace=0.42, wspace=0.42)
    operators = ["Operator 1", "Operator 2"]
    column = "operator" if "operator" in profiles else "culture_treatment_operator"
    profiles[column] = profiles[column].replace({"LE": "Operator 1", "DS": "Operator 2"})
    for ax, key, title, letter in [
        (axes[0, 0], "observed_protein_groups", "Observed protein groups", "a"),
        (axes[0, 1], "median_log2_intensity", "Median observed log₂ intensity", "b"),
    ]:
        arrays = [profiles.loc[profiles[column].eq(op), key] for op in operators]
        box(ax, arrays, colors=list(OPERATORS.values()), points=True, labels=operators)
        ax.set_ylabel(title)
        frame(ax, "y")
        panel(ax, letter)
    for ax, key, title, letter, scale in [
        (axes[1, 0], "median_pairwise_pearson", "Within-group Pearson\ncorrelation", "c", 1),
        (axes[1, 1], "median_cv_complete", "Protein coefficient\nof variation (%)", "d", 100),
    ]:
        arrays = [
            summaries.loc[summaries.group_type.eq(group), key] * scale
            for group in ["Antibiotic", "Solvent control"]
        ]
        box(
            ax,
            arrays,
            colors=["#1F77B4", "#9E9E9E"],
            points=True,
            labels=["Antibiotics", "Solvent controls"],
        )
        ax.set_ylabel(title)
        frame(ax, "y")
        panel(ax, letter)
    pc1 = (
        float(
            variance.loc[
                variance.component.astype(str).isin(["PC1", "1"]), "explained_variance_fraction"
            ].iloc[0]
        )
        * 100
    )
    pc2 = (
        float(
            variance.loc[
                variance.component.astype(str).isin(["PC2", "2"]), "explained_variance_fraction"
            ].iloc[0]
        )
        * 100
    )
    for ax, key, palette, letter, title in [
        (axes[2, 0], "model_label", MOA_COLORS, "e", "PCA by MoA"),
        (axes[2, 1], "operator", OPERATORS, "f", "PCA by operator"),
    ]:
        if key not in scores:
            scores[key] = scores.culture_treatment_operator.replace(
                {"LE": "Operator 1", "DS": "Operator 2"}
            )
        for name, color in palette.items():
            part = scores.loc[scores[key].eq(name)]
            ax.scatter(
                part.PC1,
                part.PC2,
                s=12,
                color=color,
                edgecolors="white",
                lw=0.3,
                label=SHORT.get(name, name),
            )
        ax.set(xlabel=f"PC1 ({pc1:.1f}%)", ylabel=f"PC2 ({pc2:.1f}%)")
        frame(ax, "both")
        panel(ax, letter, title)
        if key == "operator":
            ax.legend(frameon=False, loc="best", fontsize=6.5)
    moa_legend(fig, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=5, fontsize=6.5)
    return fig


def augmentation():
    from .main import encoder_panel

    fig = plt.figure(figsize=(WIDTH, 6.8))
    grid = fig.add_gridspec(
        2, 6, left=0.13, right=0.98, bottom=0.19, top=0.92, hspace=0.70, wspace=0.85
    )
    for i, (metric, letter, title) in enumerate(
        [
            ("roc_auc", "a", "Encoder / augmentation: ROC AUC"),
            ("mcc", "b", "Encoder / augmentation: MCC"),
        ]
    ):
        ax = fig.add_subplot(grid[0, i * 3 : (i + 1) * 3])
        encoder_panel(ax, metric)
        # Neither metric passed the corrected omnibus test. Show the saved
        # distributions without the post-hoc significance colour encoding.
        for patch in ax.patches:
            patch.set_facecolor("#AEC7E8")
        panel(ax, letter, title)
        ax.title.set_fontsize(8)
    q = read("encoders", "tukey_pr_auc_pvals_matrix.csv", index_col=0)
    means = (
        read("encoders", "tukey_pr_auc_means.csv")
        .set_index("method")
        .pr_auc.sort_values(ascending=False)
    )
    c = fig.add_subplot(grid[1, :2])
    matrix = q.loc[means.index, means.index]
    image = significance_matrix(c, matrix, labels=list(means.index))
    c.set_aspect("auto")
    c.tick_params(axis="both", labelsize=4.8)
    lower_axes = [(c, "c", "Tukey-adjusted PR AUC")]
    values = read("figure3", "metric_source.csv")
    tests = read("figure3", "pairwise_tests.csv")
    for index, (metric, title, letter) in enumerate(
        [("macro_ap_ovr", "Macro PR AUC (OvR)", "d"), ("macro_roc_ovr", "Macro ROC AUC (OvR)", "e")]
    ):
        ax = fig.add_subplot(grid[1, 2 + index * 2 : 4 + index * 2])
        three_training_conditions(ax, values, tests, metric, title, zoom=True)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35, labelsize=6)
        for t in ax.get_xticklabels():
            t.set_ha("right")
        lower_axes.append((ax, letter, title))
    for ax, letter, title in lower_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, pos.height - 0.09])
        fig.text(pos.x0 - 0.025, pos.y1 + 0.018, letter, fontsize=11, ha="right", va="bottom")
        fig.text(pos.x0, pos.y1 + 0.018, title, fontsize=7.5, va="bottom")
    cb = fig.colorbar(
        image,
        cax=fig.add_axes([0.15, 0.055, 0.20, 0.012]),
        orientation="horizontal",
        ticks=[0.05, 0.5, 1],
        spacing="proportional",
    )
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Adjusted p-value", fontsize=7)
    plain_colorbar(cb)
    fig.legend(
        handles=[Patch(facecolor=color, label=arm) for arm, color in CONDITIONS.items()],
        loc="lower center",
        bbox_to_anchor=(0.72, 0.035),
        frameon=False,
        ncol=1,
        fontsize=6.5,
    )
    return fig


def text_controls():
    data = read("text_controls", "figure_source.csv")
    models = ["TabM", "LightGBM"]
    fig, ax = plt.subplots(figsize=(WIDTH, 5.6))
    fig.subplots_adjust(left=0.30, right=0.98, top=0.91, bottom=0.13)
    fig._horizontal_alignment = "left"
    for j, model in enumerate(models):
        part = data.loc[data.model.eq(model)]
        color = ["#244A7C", "#E0A458"][j]
        base = part.loc[part.arm.eq("semantic_aug") & part.view.eq("standard")].iloc[0]
        if model == "TabM":
            ax.axvline(base.estimate * 100, color=color, ls="--", lw=0.8, alpha=0.65)
        for row in part.itertuples():
            fixed = row.training_schedule == "fixed_100_epochs"
            alpha = 0.3 if fixed else 1
            x, lo, hi = np.array([row.estimate, row.ci_low, row.ci_high]) * 100
            ax.errorbar(
                x,
                row.y + (j - 0.5) * 0.18,
                xerr=[[x - lo], [hi - x]],
                fmt="D" if fixed else "o",
                color=color,
                ms=4,
                lw=1,
                capsize=2,
                alpha=alpha,
            )
    reference = data.loc[data.model.eq("TabM")].drop_duplicates("y")
    ax.set(
        yticks=reference.y,
        yticklabels=reference.label,
        ylim=(reference.y.max() + 0.8, -0.8),
        xlim=(0, 100),
        xticks=[0, 20, 40, 60, 80, 100],
        xlabel="Compound-level top-1 accuracy (%)",
    )
    ax.tick_params(axis="y", length=0, labelsize=6.5)
    frame(ax, "x")
    left_tick_labels(ax)
    ax.set_title("Controls for text-based class matching", loc="left", pad=12)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", color=color, label=model)
            for model, color in zip(models, ["#244A7C", "#E0A458"])
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.65, 0.015),
        ncol=2,
        fontsize=7,
    )
    return fig


def feature_statistics():
    from .main import feature_panel, model_panel

    fig = plt.figure(figsize=(WIDTH, 10.1))
    grid = fig.add_gridspec(
        4,
        6,
        left=0.10,
        right=0.98,
        top=0.95,
        bottom=0.08,
        height_ratios=[1, 1.4, 1, 1.3],
        hspace=0.7,
        wspace=0.8,
    )
    for i, (metric, letter) in enumerate([("roc_auc", "a"), ("mcc", "b")]):
        ax = fig.add_subplot(grid[0, i * 3 : (i + 1) * 3])
        order = feature_panel(ax, None, metric)
        ax.set_xticks([0, 7, 15, 23, 31], [1, 8, 16, 24, 32])
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Feature-set index (ordered by mean performance)", fontsize=6.5)
        panel(ax, letter, f"TabM · {metric.replace('_',' ').upper()}")
    for i, (metric, letter) in enumerate([("pr_auc", "c"), ("roc_auc", "d"), ("mcc", "e")]):
        ax = fig.add_subplot(grid[1, 2 * i : 2 * i + 2])
        source = read("model_screen", "tabm/tabm_per_cycle_all_metrics.csv")
        order = source.groupby("var_tag")[metric].mean().sort_values(ascending=False).index
        fig._condition_keys = getattr(fig, "_condition_keys", []) + [
            dict(
                comparison="TabM feature sets",
                metric=metric,
                index=j + 1,
                condition=tag.replace("CellPaint", "InfoAlign"),
            )
            for j, tag in enumerate(order)
        ]
        q = read(
            "model_screen", f"tabm/{metric}/tabm_pairwise_{metric}_pvals_BH.csv", index_col=0
        ).loc[order, order]
        image = significance_matrix(ax, q, ticks=4)
        panel(ax, letter, f"TabM · {metric.replace('_',' ').upper()}")
        ax.title.set_fontsize(7.5)
        ax.set_xlabel("Feature-set index", fontsize=6)
    for i, (metric, letter) in enumerate([("roc_auc", "f"), ("mcc", "g")]):
        ax = fig.add_subplot(grid[2, i * 3 : (i + 1) * 3])
        model_panel(ax, metric)
        panel(ax, letter, f"Models · {metric.replace('_',' ').upper()}")
        ax.title.set_fontsize(8)
    for i, (metric, letter) in enumerate([("pr_auc", "h"), ("roc_auc", "i"), ("mcc", "j")]):
        ax = fig.add_subplot(grid[3, 2 * i : 2 * i + 2])
        q = read("model_screen", f"cross_model/pairwise_{metric}_pvals_BH.csv", index_col=0)
        ordered = [
            tag
            for model in ["rf", "lightgbm", "xgboost", "mlp", "pytorch_mlp", "tabm"]
            for tag in q.index
            if tag.startswith(model + "|")
        ]
        q = q.loc[ordered, ordered]
        fig._condition_keys = getattr(fig, "_condition_keys", []) + [
            dict(
                comparison="Model / feature sets",
                metric=metric,
                index=j + 1,
                condition=tag.replace("CellPaint", "InfoAlign"),
            )
            for j, tag in enumerate(ordered)
        ]
        image = significance_matrix(ax, q, ticks=3)
        panel(ax, letter, f"Models · {metric.replace('_',' ').upper()}")
        ax.title.set_fontsize(7.5)
        ax.set_xlabel("Model–feature condition", fontsize=6)
    cb = fig.colorbar(
        image,
        cax=fig.add_axes([0.37, 0.04, 0.35, 0.009]),
        orientation="horizontal",
        ticks=[0.05, 0.25, 0.5, 0.75, 1],
        spacing="proportional",
    )
    cb.set_label("BH-adjusted q-value", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    plain_colorbar(cb)
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
        bbox_to_anchor=(0.55, 0.995),
        ncol=3,
        fontsize=5.5,
        handlelength=2.8,
        handleheight=1.5,
        columnspacing=1.4,
    )
    # Feature maps share each boxplot's exact, metric-specific column order.
    # Reuse the existing inter-row space without moving the numerical panels.
    features = [
        ("Proteomics", "Proteomics"),
        ("ECFP", "ECFP"),
        ("CellPaint", "InfoAlign"),
        ("GC", "GC"),
        ("MIC", "MIC"),
    ]
    fig._feature_maps = []
    for ax, metric in zip(fig.axes[:2], ["roc_auc", "mcc"]):
        values = source.pivot(index="cv_cycle", columns="var_tag", values=metric)
        order = values.mean().sort_values(ascending=False).index.tolist()
        assert (
            order
            == source.groupby("var_tag")[metric].mean().sort_values(ascending=False).index.tolist()
        )
        pos = ax.get_position()
        dots = fig.add_axes([pos.x0, pos.y0 - 0.058, pos.width, 0.049], sharex=ax)
        for x, tag in enumerate(order):
            included = set(tag.split("+"))
            for y, (key, name) in enumerate(features):
                dots.scatter(x, y, s=7, color="#222222" if key in included else "#EBEBEB")
        dots.set(
            ylim=(4.7, -0.7),
            yticks=range(5),
            yticklabels=[name for _, name in features],
            xticks=[0, 7, 15, 23, 31],
            xticklabels=[1, 8, 16, 24, 32],
        )
        dots.tick_params(axis="y", length=0, labelsize=6, pad=3)
        dots.tick_params(axis="x", length=2, width=0.8, labelsize=6, pad=2)
        dots.grid(False)
        for spine in dots.spines.values():
            spine.set_visible(False)
        dots._unboxed = True
        ax.set_xlabel("")
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        fig._feature_maps.append(dict(ax=ax, dots=dots, metric=metric, order=order))
    return fig


MODEL_ARMS = [
    ("deployed_mapper", "MAPPER"),
    ("selected_lightgbm", "LightGBM"),
    ("selected_pytorch_mlp", "PyTorch MLP"),
    ("selected_xgboost", "XGBoost"),
    ("selected_rf", "Random forest"),
    ("selected_mlp", "sklearn MLP"),
]


def confusion():
    data = read("baselines", "main_model_confusion_source.csv")
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 9.1))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.94, bottom=0.13, hspace=0.40, wspace=0.27)
    for i, (ax, (arm, label)) in enumerate(zip(axes.flat, MODEL_ARMS)):
        matrix = (
            data.loc[data.arm.eq(arm)]
            .pivot(index="Reference_MoA", columns="Predicted_MoA", values="percentage")
            .reindex(index=CLASSES, columns=CLASSES)
            .fillna(0)
        )
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
        for (y, x), v in np.ndenumerate(matrix.to_numpy()):
            if v:
                ax.text(
                    x,
                    y,
                    f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if v > 55 else "#222222",
                )
        ax.set(
            xticks=range(9),
            xticklabels=[SHORT[c] for c in CLASSES],
            yticks=range(9),
            yticklabels=[SHORT[c] for c in CLASSES] if i % 2 == 0 else [],
        )
        ax.tick_params(axis="x", rotation=90, labelsize=6, length=0)
        ax.tick_params(axis="y", labelsize=6, length=0)
        ax.grid(False)
        panel(ax, chr(97 + i), label)
    cb = fig.colorbar(
        image,
        cax=fig.add_axes([0.25, 0.05, 0.5, 0.012]),
        orientation="horizontal",
        ticks=[0, 50, 100],
    )
    cb.set_label("Reference-class recall (%)", fontsize=8)
    return fig


def calibration():
    data = read("baselines", "calibration_bins.csv")
    views = data.view.drop_duplicates().tolist()
    view = next((v for v in views if "top1" in v or "top_1" in v), views[-1])
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 7.7))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.08, hspace=0.5, wspace=0.3)
    colors = ["#244A7C", "#168A85", "#8054A3", "#D47719", "#C44E52", "#626A73"]
    for i, (ax, (arm, label), color) in enumerate(zip(axes.flat, MODEL_ARMS, colors)):
        part = data.loc[data.arm.eq(arm) & data.view.eq(view)].sort_values("mean_score")
        ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8)
        ax.plot(part.mean_score, part.observed_event_frequency, color=color, lw=1)
        for row in part.itertuples():
            ax.scatter(
                row.mean_score,
                row.observed_event_frequency,
                s=22,
                color=color if not row.sparse else "white",
                edgecolor=color,
                lw=0.8,
                zorder=3,
            )
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1.03),
            xticks=[0, 0.5, 1],
            yticks=[0, 0.5, 1],
            xlabel="Mean highest matching score",
            ylabel="Observed accuracy",
        )
        frame(ax, "both")
        panel(ax, chr(97 + i), label)
    return fig


def representatives():
    scores = read("representatives", "all_compound_mean_scores.csv")
    truths = (
        read("representatives", "nine_compound_predictions.csv")
        .drop_duplicates("CompoundID")
        .set_index("CompoundID")
        .TrueMoA
    )
    ids = ["JH6", "JH13", "JH24", "JH27", "JH31", "JH38", "JH42", "JH44", "JH48"]
    names = [
        "Imipenem",
        "Chloramphenicol",
        "Tetroxoprim",
        "Temafloxacin",
        "Nitrofurantoin",
        "Rifampicin",
        "MUT056399",
        "Polymyxin B",
        "PF-5081090",
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.8))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.81, bottom=0.31)
    for x, cid in enumerate(ids):
        part = (
            scores.loc[scores.arm.eq("all_modalities") & scores.CompoundID.eq(cid)]
            .sort_values("score", ascending=False)
            .head(3)
        )
        for j, row in enumerate(part.itertuples()):
            ax.bar(
                x + (j - 1) * 0.23,
                row.score,
                width=0.21,
                color=MOA_COLORS[row.CandidateMoA],
                edgecolor=(
                    "black" if row.CandidateMoA == truths[cid] else MOA_COLORS[row.CandidateMoA]
                ),
                lw=1 if row.CandidateMoA == truths[cid] else 0.5,
            )
    ax.set(xticks=range(9), xticklabels=names, ylim=(0, 1.03), ylabel="Mean matching score")
    ax.tick_params(axis="x", rotation=45, labelsize=6.5)
    for t in ax.get_xticklabels():
        t.set_ha("right")
    frame(ax, "y")
    ax.set_title("Full-feature predictions for pathway-enrichment representatives", loc="left")
    moa_legend(fig, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=5, fontsize=6.5)
    return fig


def operators():
    from .operator_bootstrap import plot_values

    counts = read("operator_combined", "operator_counts.csv", index_col=0)
    comparisons = plot_values().rename(columns={"mean": "mean_top1"})
    assert comparisons.interval_type.eq("conditional_compound_bootstrap_95_CI").all()
    context = read("operator", "classwise_plot_data.csv")
    if not context.target_class.eq("Overall").any():
        totals = {
            key: float(np.average(context[key], weights=context.n_unique_test_compounds))
            for key in ["top1_DS_only_percent", "top1_LE_only_percent", "top1_DS_plus_LE_percent"]
        }
        context = pd.concat(
            [context, pd.DataFrame([dict(target_class="Overall", **totals)])], ignore_index=True
        )
    fig = plt.figure(figsize=(WIDTH, 8.7))
    grid = fig.add_gridspec(
        3,
        2,
        left=0.16,
        right=0.96,
        top=0.94,
        bottom=0.13,
        height_ratios=[1, 1.1, 1.1],
        hspace=0.65,
        wspace=0.35,
    )
    a = fig.add_subplot(grid[0, :])
    y = np.arange(len(counts))
    for op, key, offset in [("Operator 1", "LE", -0.17), ("Operator 2", "DS", 0.17)]:
        a.barh(y + offset, counts[key], height=0.30, color=OPERATORS[op], label=op)
        for yy, value in zip(y + offset, counts[key]):
            a.text(value + 0.10, yy, str(value), va="center", fontsize=6.5)
    a.set(
        yticks=y,
        yticklabels=[SHORT.get(c, c) for c in counts.index],
        xlim=(0, 9.2),
        xlabel="Compounds per operator",
    )
    a.invert_yaxis()
    a.tick_params(axis="y", length=0)
    frame(a, "x")
    panel(a, "a", "Class-by-operator distribution")
    a.legend(loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2, frameon=False, fontsize=7)
    for i, direction in enumerate(["LE_to_DS", "DS_to_LE"]):
        ax = fig.add_subplot(grid[1, i])
        for j, config in enumerate(["proteomics_text", "all_modalities_text"]):
            for k, validation in enumerate(
                ["operator_transfer", "count_matched", "structure_matched"]
            ):
                row = comparisons.loc[
                    comparisons.direction.eq(direction)
                    & comparisons.configuration.eq(config)
                    & comparisons.validation.eq(validation)
                ].iloc[0]
                value = row.mean_top1
                lo = row.low
                hi = row.high
                color = ["#676E56", "#E0A458", "#9BC53D"][k]
                ax.errorbar(
                    value,
                    j * 4 + k,
                    xerr=[[value - lo], [hi - value]],
                    fmt="o",
                    color=color,
                    ms=4,
                    lw=0.8,
                    ls="none",
                    capsize=2,
                )
        ax.set(
            xlim=(0, 100),
            xticks=[0, 50, 100],
            ylim=(6.6, -1.1),
            yticks=[0, 1, 2, 4, 5, 6],
            yticklabels=(
                ["Operator held out", "Count matched", "Structure matched"] * 2 if i == 0 else []
            ),
            xlabel="Top-1 accuracy (%)",
        )
        ax.tick_params(axis="y", length=0, labelsize=6)
        ax.set_title("Operator 1 → 2" if i == 0 else "Operator 2 → 1", fontsize=8)
        ax.text(0, -0.65, "Proteomics + text", fontsize=6, va="bottom")
        ax.text(0, 3.45, "All modalities + text", fontsize=6, va="bottom")
        frame(ax, "x")
        if i == 0:
            panel(ax, "b")
    c = fig.add_subplot(grid[2, :])
    for j, (col, label, color) in enumerate(
        [
            ("top1_DS_only_percent", "Operator 2 only", "#676E56"),
            ("top1_LE_only_percent", "Operator 1 references only", "#E0A458"),
            ("top1_DS_plus_LE_percent", "Operator 2 + same references", "#9BC53D"),
        ]
    ):
        c.scatter(
            context[col],
            np.arange(len(context)) + (j - 1) * 0.10,
            color=color,
            s=20,
            label=label,
            zorder=3,
        )
    c.set(
        yticks=range(len(context)),
        yticklabels=[SHORT.get(x, x) for x in context.target_class],
        xlabel="Compound-level top-1 accuracy (%)",
        xlim=(-3, 105),
    )
    c.invert_yaxis()
    c.tick_params(axis="y", length=0, labelsize=7)
    frame(c, "x")
    panel(c, "c", "Classwise operator coverage")
    c.legend(
        loc="upper center",
        bbox_to_anchor=(0.4, -0.23),
        ncol=1,
        frameon=False,
        fontsize=6.5,
        title="Training compounds in the tested MoA class",
        title_fontsize=7,
    )
    return fig


def batches():
    performance = read("batches", "performance_data.csv")
    composition = read("batches", "composition_data.csv")
    fig = plt.figure(figsize=(WIDTH, 6.0))
    grid = fig.add_gridspec(
        2, 1, left=0.17, right=0.98, top=0.93, bottom=0.12, height_ratios=[1, 1.7], hspace=0.30
    )
    a = fig.add_subplot(grid[0])
    for j, (config, color, label) in enumerate(
        [
            ("proteomics_text", "#2A9D8F", "Proteomics + text"),
            ("all_modalities_text", "#244A7C", "All modalities + text"),
        ]
    ):
        rows = performance.loc[performance.configuration.eq(config)].sort_values("campaign")
        x = np.arange(5) + (j - 0.5) * 0.3
        a.bar(x, rows.top1_accuracy * 100, width=0.28, color=color, label=label)
        a.scatter(x, rows.top2_accuracy * 100, s=20, facecolors="white", edgecolors=color, zorder=3)
    n_reference = len(read("overview", "structures.csv"))
    counts = (
        performance.loc[performance.configuration.eq("proteomics_text")]
        .sort_values("campaign")
        .n_compounds
    )
    a.set(
        xticks=range(5),
        xticklabels=[f"B{i+1}\n{n_reference-n} / {n}" for i, n in enumerate(counts)],
        ylim=(0, 112),
        yticks=[0, 50, 100],
        ylabel="Accuracy (%)",
    )
    frame(a, "y")
    panel(a, "a", "Measurement-batch holdouts")
    fig.legend(
        *a.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.55, 0.995),
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    b = fig.add_subplot(grid[1])
    classes = list(CLASSES)
    for x, campaign in enumerate(sorted(composition.campaign.unique())):
        for y, moa in enumerate(classes):
            row = composition.loc[
                composition.campaign.eq(campaign) & composition.moa_class.eq(moa)
            ].iloc[0]
            if not row.tested:
                b.add_patch(Rectangle((x - 0.48, y - 0.46), 0.96, 0.92, color="#EEEEEE", lw=0))
                continue
            for field, offset in [("train", -0.19), ("test", 0.19)]:
                start = x - 0.43
                for op, key in [("Operator 1", "LE"), ("Operator 2", "DS")]:
                    n = float(row[f"{field}_{key}_compounds"])
                    width = n / 9 * 0.84
                    if width:
                        b.add_patch(
                            Rectangle(
                                (start, y + offset - 0.12),
                                width,
                                0.24,
                                facecolor=OPERATORS[op],
                                edgecolor="white",
                                lw=0.3,
                            )
                        )
                        if n >= 1:
                            b.text(
                                start + width / 2,
                                y + offset,
                                f"{n:g}",
                                ha="center",
                                va="center",
                                fontsize=5.5,
                            )
                    start += width
            if campaign == "C4" and moa == "RNA Synthesis":
                train_total = float(row.train_LE_compounds + row.train_DS_compounds)
                b.text(
                    x - 0.43 + train_total / 9 * 0.84 + 0.05,
                    y - 0.19,
                    "*",
                    ha="left",
                    va="center",
                    fontsize=8,
                )
    b.set(
        xlim=(-0.5, 4.5),
        ylim=(8.5, -0.5),
        xticks=range(5),
        xticklabels=["B1", "B2", "B3", "B4", "B5"],
        yticks=range(9),
        yticklabels=[SHORT[c] for c in classes],
    )
    b.tick_params(length=0, labelsize=7)
    b.grid(False)
    panel(b, "b", "Training and test reference coverage")
    fig.legend(
        handles=[Patch(facecolor=color, label=op) for op, color in OPERATORS.items()],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    return fig


def build(number):
    if number in (10, 11):
        from .external import representation_figure, class_score_figure

        return representation_figure() if number == 10 else class_score_figure()
    return {
        1: qc,
        2: augmentation,
        3: text_controls,
        4: feature_statistics,
        5: confusion,
        6: calibration,
        7: representatives,
        8: operators,
        9: batches,
    }[number]()
