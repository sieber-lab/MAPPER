"""Reusable numerical panels with explicit data and colour semantics."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from .style import box, frame, panel, MOA_COLORS, SHORT, STAT_COLORS


def significance_brackets(ax, data, metric, top=1.04, step=0.11, above=False):
    """Display saved post-hoc decisions; never recompute a test when plotting."""
    conditions = ["Native no-text", "Matched no-text", "With text"]
    offsets = dict(zip(conditions, [-0.25, 0, 0.25]))
    heights = {
        ("Native no-text", "Matched no-text"): top,
        ("Matched no-text", "With text"): top + step,
        ("Native no-text", "With text"): top + 2 * step,
    }
    for i, model in enumerate(["lightgbm", "pytorch_mlp"]):
        for row in data.loc[data.model.eq(model) & data.metric.eq(metric)].itertuples():
            y = heights[row.arm_a, row.arm_b]
            left, right = i + offsets[row.arm_a], i + offsets[row.arm_b]
            transform = ax.get_xaxis_transform() if above else ax.transData
            ax.plot(
                [left, left, right, right],
                [y, y + 0.015, y + 0.015, y],
                color="#222222",
                lw=0.8,
                clip_on=False,
                transform=transform,
            )
            ax.text(
                (left + right) / 2,
                y + 0.035,
                row.annotation,
                ha="center",
                va="bottom",
                fontsize=7,
                transform=transform,
            )


def three_training_conditions(ax, values, tests, metric, ylabel, zoom=False):
    conditions = ["No text", "Repeated samples, no text", "With text"]
    from .style import CONDITIONS

    for i, model in enumerate(["LightGBM", "PyTorch-MLP"]):
        arrays = [
            values.loc[
                values.Model.eq(model) & values.Method.eq(arm) & values.metric_key.eq(metric),
                "value",
            ]
            for arm in conditions
        ]
        assert all(len(x) == 25 for x in arrays)
        box(ax, arrays, [i - 0.25, i, i + 0.25], list(CONDITIONS.values()), width=0.23)
    significance_brackets(ax, tests, metric, above=zoom)
    ax.set(
        xticks=[0, 1],
        xticklabels=["LightGBM", "PyTorch MLP"],
        ylabel=ylabel,
        ylim=(0, 1.40),
        yticks=[0, 0.25, 0.5, 0.75, 1],
    )
    frame(ax)
    if zoom:
        ax.set(ylim=(0.5, 1.01), yticks=[0.5, 0.75, 1.0])


def significance_matrix(ax, matrix, labels=None, ticks=1):
    """Blue below .05, original OrRd gradient otherwise; diagonal omitted."""
    n = len(matrix)
    mask = np.triu(np.ones((n, n), bool), 0)
    colors = np.vstack(
        [plt.matplotlib.colors.to_rgba("#1F77B4"), plt.get_cmap("OrRd")(np.linspace(0, 1, 243))]
    )
    norm = BoundaryNorm(np.r_[0, np.linspace(0.05, 1, 244)], 244)
    mesh = ax.pcolormesh(
        np.ma.masked_where(mask, np.asarray(matrix, float)),
        cmap=ListedColormap(colors),
        norm=norm,
        edgecolors="white",
        linewidth=0.2,
    )
    indices = np.arange(0, n, ticks)
    strings = [str(i + 1) for i in indices] if labels is None else [labels[i] for i in indices]
    ax.set(xlim=(0, n), ylim=(n, 0), aspect="equal")
    ax.set_xticks(
        indices + 0.5,
        strings,
        rotation=60 if labels is not None else 0,
        ha="right" if labels is not None else "center",
    )
    ax.set_yticks(indices + 0.5, strings)
    ax.tick_params(length=0, labelsize=5.5, pad=2)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def class_score_bars(
    ax,
    scores,
    *,
    candidate="CandidateMoA",
    value="score",
    truth=None,
    maximum=3,
    horizontal=False,
    label_scores=False,
):
    selected = scores.sort_values(value, ascending=False, kind="stable").head(maximum)
    for i, row in enumerate(selected.to_dict("records")):
        cls, height = row[candidate], float(row[value])
        color = MOA_COLORS[cls]
        edge = "black" if cls == truth else color
        if horizontal:
            ax.barh(
                i,
                height,
                height=0.65,
                color=color,
                edgecolor=edge,
                linewidth=1 if cls == truth else 0.5,
            )
        else:
            ax.bar(
                i,
                height,
                width=0.65,
                color=color,
                edgecolor=edge,
                linewidth=1 if cls == truth else 0.5,
            )
        if label_scores:
            text = "<0.001" if 0 < height < 0.0005 else f"{height:.3f}"
            if horizontal:
                ax.text(
                    min(height + 0.02, 0.98),
                    i,
                    text,
                    fontsize=6.5,
                    va="center",
                    ha="left" if height < 0.8 else "right",
                )
            else:
                ax.text(i, height + 0.025, text, fontsize=6.5, ha="center", va="bottom")
    labels = [SHORT[x] for x in selected[candidate]]
    if horizontal:
        ax.set(
            yticks=range(len(selected)),
            yticklabels=labels,
            xlim=(0, 1.03),
            ylim=(len(selected) - 0.5, -0.5),
        )
        ax.set_xticks([0, 0.5, 1])
        frame(ax, "x")
    else:
        ax.set(xticks=range(len(selected)), xticklabels=labels, ylim=(0, 1.12))
        ax.set_yticks([0, 0.5, 1])
        frame(ax, "y")
    return selected


def q_based_boxes(ax, values, q, *, order, offsets=None, width=0.65):
    means = values.mean()
    best = means.idxmax()
    if offsets is None:
        offsets = np.arange(len(order))
    colors = [
        STAT_COLORS[
            "best" if tag == best else "worse" if q.loc[best, tag] < 0.05 else "not different"
        ]
        for tag in order
    ]
    return box(
        ax,
        [values[tag].dropna().to_numpy() for tag in order],
        positions=offsets,
        colors=colors,
        width=width,
    )
