"""Extended Data figures, including programmatically assembled artwork."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .style import *
from .artwork import reserve
from .main import artwork_grid, molecular_example, modality_scores


def growth():
    from . import overview_helpers as original

    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 7.5))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.94, bottom=0.12, hspace=0.55, wspace=0.30)
    data = read("overview", "growth_profiles.csv")
    names = ["Cell wall", "Folic acid", "Membrane", "Nitrofurans", "LPS synth."]
    original.plot_growth_facets_axes(axes, data, moas_to_show=names)
    for i, ax in enumerate(axes.flat):
        if i < 5:
            frame(ax, "y")
            panel(ax, chr(97 + i), names[i])
            ax.title.set_fontsize(9)
            ax.xaxis.label.set_fontsize(8)
            ax.yaxis.label.set_fontsize(8)
        else:
            ax.set_axis_off()
    from .growth_labels import register

    register(fig, axes, data, names, font_size=6.5)
    return fig


def molecular_comparisons():
    from .original_artwork import structure_panel, adjacent_plot, adjacent_header

    height = 667.846
    fig = plt.figure(figsize=(WIDTH, height / 72))
    page_w = WIDTH * 72
    for key, top, bottom, examples in [
        (
            "extended02_dna",
            27.5,
            318,
            [
                ("JH27", "c", "Temafloxacin", [342.95, 36.37, 504.46, 155.73], 28.65),
                ("JH32", "e", "Gepotidacin", [342.95, 197.51, 504.46, 316.88], 182.83),
            ],
        ),
        (
            "extended02_fatty",
            363.5,
            648,
            [
                ("JH42", "h", "MUT056399", [342.95, 367.27, 504.46, 486.64], 365.22),
                ("JH61", "j", "Debio-1452-NH₂", [342.95, 524.30, 504.46, 643.67], 513.85),
            ],
        ),
    ]:
        art = structure_panel(
            fig,
            key,
            [3 / page_w, (height - bottom) / height, 299 / page_w, (bottom - top) / height],
        )
        for identity, letter, name, rectangle, header_y in examples:
            ax = adjacent_plot(fig, art, rectangle)
            modality_scores(ax, identity)
            adjacent_header(fig, art, ax, letter, name, header_y)
    moa_legend(
        fig,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.997),
        ncol=9,
        fontsize=6,
        handlelength=0.8,
        columnspacing=0.7,
        handletextpad=0.3,
    )
    return fig


def volcano(ax, filename, label):
    USED_INPUTS.add("volcano/" + filename)
    data = pd.read_csv(INPUTS / "volcano" / filename, sep="\t", low_memory=False)
    difference = next(c for c in data if "Student's T-test Difference" in c)
    pvalue = next(c for c in data if "-Log Student's T-test p-value" in c)
    x = pd.to_numeric(data[difference], errors="coerce").to_numpy()
    y = pd.to_numeric(data[pvalue], errors="coerce").to_numpy()
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    threshold = -np.log10(0.05)
    up = (x > 1) & (y > threshold)
    down = (x < -1) & (y > threshold)
    ax.scatter(x[~(up | down)], y[~(up | down)], s=3, color="#BBBBBB", alpha=0.6, linewidths=0)
    ax.scatter(x[down], y[down], s=4, color="#346E9B", alpha=0.8, linewidths=0)
    ax.scatter(x[up], y[up], s=4, color="#C45C5C", alpha=0.8, linewidths=0)
    for boundary in [-1, 1]:
        ax.axvline(boundary, color="#333333", lw=0.8)
    ax.axhline(threshold, color="#333333", lw=0.8)
    ax.set(xlabel="Log₂ treated / control", ylabel="−log₁₀ p-value")
    ax.set_title(label, loc="left", fontsize=9, pad=8)
    frame(ax, None)


def enrichment_and_volcano():
    # STRING panels are the declared fixed web-interface exports, not newly
    # calculated enrichments. Their original result tables are also bundled.
    import fitz
    import textwrap
    import xml.etree.ElementTree as ET

    USED_INPUTS.add("string/string_analysis.pdf")
    source = fitz.open(INPUTS / "string/string_analysis.pdf")
    blocks = source[0].get_text("blocks")
    titles = sorted(
        [
            b
            for b in blocks
            if b[0] < 310 and "Biological Process (Gene Ontology) enrichment" in b[4]
        ],
        key=lambda b: b[1],
    )
    assert len(titles) == 5
    count_labels = sorted(
        [
            span
            for block in source[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span["text"] == "Gene count"
        ],
        key=lambda span: span["bbox"][1],
    )
    assert len(count_labels) == 5
    fig = plt.figure(figsize=(WIDTH, 11.4))
    grid = fig.add_gridspec(
        5,
        2,
        left=0.035,
        right=0.985,
        top=0.975,
        bottom=0.045,
        width_ratios=[2.05, 1],
        hspace=0.23,
        wspace=0.28,
    )
    examples = [
        ("jh27_h.txt", "Temafloxacin"),
        ("jh13_d.txt", "Chloramphenicol"),
        ("jh24_dc.txt", "Tetroxoprim"),
        ("jh31_df.txt", "Nitrofurantoin"),
        ("jh44_hh.txt", "Polymyxin B"),
    ]
    for i, (filename, label) in enumerate(examples):
        left = fig.add_subplot(grid[i, 0])
        left.set_axis_off()
        pos = left.get_position()
        title = titles[i]
        next_y = titles[i + 1][1] if i < 4 else source[0].rect.height
        signal = next(
            b for b in blocks if b[0] < 310 and title[1] < b[1] < next_y and "Signal" in b[4]
        )
        fragments = sorted(
            [b for b in blocks if b[2] < 95 and title[3] < b[1] and b[3] < signal[1]],
            key=lambda b: b[1],
        )
        terms = []
        for fragment in fragments:
            if terms and fragment[1] - terms[-1][3] < 1.5:
                old = terms.pop()
                terms.append(
                    (
                        min(old[0], fragment[0]),
                        old[1],
                        max(old[2], fragment[2]),
                        fragment[3],
                        old[4] + " " + fragment[4],
                    )
                )
            else:
                terms.append(fragment)
        assert len(terms) == (7 if i == 0 else 10)
        path = f"string/enrichment_{i+1}.svg"
        root = ET.parse(INPUTS / "artwork" / path).getroot()
        _, _, native_w, native_h = map(float, root.get("viewBox").split())
        crop_x = max(b[2] for b in terms) - 9 + 2
        crop_y = 0.0
        crop = [crop_x, crop_y, native_w - crop_x, native_h - crop_y]
        picture = fig.add_axes([pos.x0 + pos.width * 0.36, pos.y0, pos.width * 0.64, pos.height])
        count = count_labels[i]
        clip_y = title[1] - 1
        assert title[1] < count["bbox"][1] < next_y
        patch = dict(
            original="Gene count",
            replacement="Protein count",
            bbox=[
                count["bbox"][0] - 9,
                count["bbox"][1] - clip_y,
                count["bbox"][2] - 9,
                count["bbox"][3] - clip_y,
            ],
            x=count["origin"][0] - 9,
            y=count["origin"][1] - clip_y,
            font_size=count["size"],
        )
        reserve(picture, path, view_box=crop, text_patches=[patch])
        p = picture.get_position()
        page_w, page_h = fig.get_size_inches() * 72
        scale = min(p.width * page_w / crop[2], p.height * page_h / crop[3])
        actual_left = p.x0 + (p.width - crop[2] * scale / page_w) / 2
        actual_top = p.y0 + (p.height + crop[3] * scale / page_h) / 2
        # Preserve the original STRING plot, Signal values and grouping as
        # artwork; reflow exact process labels and relabel the count heading.
        for term in terms:
            name = " ".join(term[4].split())
            center = (term[1] + term[3]) / 2 - (title[1] - 1) - crop_y
            fig.text(
                actual_left - 4 / page_w,
                actual_top - center * scale / page_h,
                textwrap.fill(name, 34, break_long_words=False, break_on_hyphens=False),
                ha="right",
                va="center",
                fontsize=5.5,
                linespacing=1.0,
            )
        fig.text(pos.x0 - 0.02, pos.y1 + 0.01, chr(97 + 2 * i), fontsize=11)
        fig.text(pos.x0 + 0.01, pos.y1 + 0.009, label + " · STRING", fontsize=7, va="bottom")
        right = fig.add_subplot(grid[i, 1])
        volcano(right, filename, label)
        panel(right, chr(98 + 2 * i))
        if i < 4:
            right.set_xlabel("")
        right.tick_params(labelsize=6)
        right.xaxis.label.set_fontsize(7)
        right.yaxis.label.set_fontsize(7)
    source.close()
    return fig


def remaining_volcano():
    fig = plt.figure(figsize=(WIDTH, 7.6))
    grid = fig.add_gridspec(
        3,
        2,
        left=0.12,
        right=0.98,
        top=0.94,
        bottom=0.16,
        height_ratios=[1, 1, 1.15],
        hspace=0.60,
        wspace=0.32,
    )
    entries = [
        ("jh6_dh.txt", "Imipenem"),
        ("jh48_db.txt", "PF-5081090"),
        ("jh38_dc.txt", "Rifampicin"),
        ("jh42_da.txt", "MUT056399"),
    ]
    for i, (file, name) in enumerate(entries):
        ax = fig.add_subplot(grid[i // 2, i % 2])
        volcano(ax, file, name)
        panel(ax, chr(97 + i))
    ax = fig.add_subplot(grid[2, :])
    scores = read("representatives", "all_compound_mean_scores.csv")
    truth = (
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
    for x, cid in enumerate(ids):
        rows = (
            scores.loc[scores.arm.eq("proteomics_only") & scores.CompoundID.eq(cid)]
            .sort_values("score", ascending=False)
            .head(3)
        )
        for j, row in enumerate(rows.itertuples()):
            ax.bar(
                x + (j - 1) * 0.23,
                row.score,
                width=0.21,
                color=MOA_COLORS[row.CandidateMoA],
                edgecolor=(
                    "black" if row.CandidateMoA == truth[cid] else MOA_COLORS[row.CandidateMoA]
                ),
                lw=1 if row.CandidateMoA == truth[cid] else 0.4,
            )
    ax.set(xticks=range(9), xticklabels=names, ylabel="Mean matching score", ylim=(0, 1.03))
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    for t in ax.get_xticklabels():
        t.set_ha("right")
    frame(ax, "y")
    panel(ax, "e", "Proteomics-only MoA matching")
    moa_legend(fig, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=5, fontsize=6.5)
    return fig


def structural_coverage():
    mean = read("structural_cases", "compound_mean_scores.csv")
    summary = read("structural_cases", "compound_top3_and_margins.csv")
    similarities = read("structural_detail", "fold_specific_similarity.csv")
    metrics = read("structural_cases", "performance_by_cv.csv")
    ids = [
        ("JH10", "Fosfomycin"),
        ("JH13", "Chloramphenicol"),
        ("JH32", "Gepotidacin"),
        ("JH61", "Debio-1452-NH₂"),
        ("JH45", "Squalamine"),
        ("JH47", "Lolamicin"),
    ]
    arms = [
        ("ecfp_only", "ECFP"),
        ("proteomics_only", "Proteomics"),
        ("proteomics_ecfp", "Proteomics + ECFP"),
    ]
    feature_colors = ["#A9AFB4", "#2A9D8F", "#244A7C"]
    fig = plt.figure(figsize=(WIDTH, 9.7))
    grid = fig.add_gridspec(
        7,
        3,
        left=0.15,
        right=0.98,
        bottom=0.16,
        top=0.94,
        height_ratios=[1, 1, 1, 1, 1, 1, 2.6],
        hspace=1.05,
        wspace=0.5,
    )
    for i, (cid, name) in enumerate(ids):
        truth = summary.loc[summary.CompoundID.eq(cid), "true_class"].iloc[0]
        sims = similarities.loc[similarities.CompoundID.eq(cid), "max_same_class_tanimoto"]
        pos = grid[i, :].get_position(fig)
        fig.text(0.025, pos.y1 + 0.012, chr(97 + i), fontsize=11)
        similarity_label = f"{sims.mean():.2f}"
        if f"{sims.min():.2f}" != f"{sims.max():.2f}":
            similarity_label += f" ({sims.min():.2f}–{sims.max():.2f})"
        fig.text(
            0.065,
            pos.y1 + 0.012,
            f"{name} | True: {SHORT[truth]} | Same-class Tanimoto: {similarity_label}",
            fontsize=7,
        )
        for j, (arm, label) in enumerate(arms):
            ax = fig.add_subplot(grid[i, j])
            rows = (
                mean.loc[mean.CompoundID.eq(cid) & mean.arm.eq(arm)]
                .sort_values("score", ascending=False)
                .head(3)
            )
            for y, row in enumerate(rows.itertuples()):
                ax.barh(
                    y,
                    row.score,
                    height=0.7,
                    color=MOA_COLORS[row.CandidateMoA],
                    edgecolor=(
                        "black" if row.CandidateMoA == truth else MOA_COLORS[row.CandidateMoA]
                    ),
                    lw=0.8,
                )
                ax.text(
                    min(row.score + 0.02, 1),
                    y,
                    f"{row.score:.3f}",
                    ha="left" if row.score < 0.75 else "right",
                    va="center",
                    fontsize=5.5,
                )
            ax.set(
                xlim=(0, 1.06),
                ylim=(2.6, -0.6),
                yticks=range(3),
                yticklabels=[SHORT[c] + (" *" if c == truth else "") for c in rows.CandidateMoA],
                xticks=[0, 0.5, 1],
            )
            ax.tick_params(axis="y", length=0, labelsize=5.8)
            ax.tick_params(axis="x", labelsize=5.5)
            frame(ax, "x")
            outcome = summary.loc[summary.CompoundID.eq(cid) & summary.arm.eq(arm)].iloc[0]
            ax.text(
                1,
                -0.57,
                f"Margin: {outcome.top1_top2_margin:.2f} | Correct: {int(outcome.correct_cv_repetitions)}/5",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=5.4,
            )
            if i == 0:
                ax.set_title(label, fontsize=8, pad=24)
    groups = ["six_examples", "low", "middle", "high"]
    for j, (metric, title) in enumerate(
        [
            ("top1_accuracy", "Top-1 accuracy (%)"),
            ("top2_accuracy", "Top-2 accuracy (%)"),
            ("macro_average_precision", "Macro average precision"),
        ]
    ):
        ax = fig.add_subplot(grid[6, j])
        for x, group in enumerate(groups):
            for k, (arm, _) in enumerate(arms):
                values = metrics.loc[
                    metrics.group.eq(group) & metrics.arm.eq(arm), metric
                ].to_numpy()
                if metric != "macro_average_precision":
                    values = values * 100
                box(
                    ax,
                    [values],
                    positions=[x + (k - 1) * 0.23],
                    colors=[feature_colors[k]],
                    width=0.21,
                    points=True,
                    whis=1.5,
                )
                ax.scatter(
                    x + (k - 1) * 0.23,
                    np.median(values),
                    s=18,
                    marker=["o", "^", "D"][k],
                    color=feature_colors[k],
                    edgecolor="white",
                    lw=0.4,
                    zorder=4,
                )
        ax.set(
            xticks=range(4),
            xticklabels=["Six\nexamples", "Bottom\n25%", "Middle\n50%", "Top\n25%"],
            ylim=(-5, 105) if metric != "macro_average_precision" else (-0.05, 1.05),
            ylabel=title,
        )
        ax.tick_params(axis="x", labelsize=6)
        frame(ax, "y")
        panel(ax, chr(103 + j))
    fig.legend(
        handles=[
            Line2D([], [], marker=marker, color=color, label=label, ls="none")
            for marker, color, (_, label) in zip(["o", "^", "D"], feature_colors, arms)
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085),
        ncol=3,
        frameon=False,
        fontsize=7,
    )
    return fig


def build(number):
    if number in (6, 7):
        from .uncertainty import classwise_figure, examples_figure

        return classwise_figure() if number == 6 else examples_figure()
    return {
        1: growth,
        2: molecular_comparisons,
        3: enrichment_and_volcano,
        4: remaining_volcano,
        5: structural_coverage,
    }[number]()
