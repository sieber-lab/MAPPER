"""Final uncertainty figures from the matched, original-architecture analysis.

All six methods use the same saved primary predictions. The displayed meta
model is the original fixed architecture, not the superseded revision model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch

from .style import *
from .artwork import reserve


DECISION_COLOURS = {True: "#346E9B", False: "#B84A59"}


def decision_colour(row):
    """Retrospective outcome of a saved compound-level uncertainty decision.

    Omitted classes and incorrect known-class assignments should be flagged;
    correct known-class assignments should pass. No score or flag is changed.
    """
    should_flag = row.role == "ood" or not bool(row.correct)
    return DECISION_COLOURS[bool(row.flag) == should_flag]


def methods_tables():
    calls = read("uncertainty", "six_method_calls.csv")
    metrics = read("uncertainty", "metrics.csv")
    curves = read("uncertainty", "curves.csv")
    # Original fixed architecture replaces only the meta-model contribution.
    for table in (calls, metrics, curves):
        table.loc[table.method.eq("fixed_original_meta"), "method"] = "meta_model"
    return calls, metrics, curves


def method_rows(table, method):
    part = table.loc[table.method.eq(method)]
    if part.empty:
        source = read(
            "uncertainty_baseline", "metrics.csv" if "auroc" in table.columns else "curves.csv"
        )
        part = source.loc[source.method.eq(method)]
    return part


def pr_points(rows, average_precision):
    positive = (
        rows.loc[rows.x.gt(0)]
        .sort_values(["x", "y"], ascending=[False, False])
        .drop_duplicates("x", keep="first")
    )
    anchor = pd.DataFrame({"x": [0.0], "y": [float(positive.iloc[-1].y)]})
    points = pd.concat([positive, anchor], ignore_index=True).sort_values("x")
    np.testing.assert_allclose(
        np.sum(np.diff(points.x) * points.y.to_numpy()[1:]), average_precision, atol=1e-12
    )
    return points


def curve(ax, metrics, curves, *, kind, letter):
    for method, label, color in METHODS:
        row = method_rows(metrics, method).iloc[0]
        data = method_rows(curves, method)
        data = data.loc[data.curve.eq(kind)]
        area = float(row.auroc if kind == "roc" else row.average_precision)
        if kind == "pr":
            points = pr_points(data, area)
            ax.step(
                points.x, points.y, where="pre", color=color, label=f"{label}: {area:.2f}", lw=1
            )
        else:
            ax.plot(data.x, data.y, color=color, label=f"{label}: {area:.2f}", lw=1)
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8)
    else:
        ax.axhline(0.5, color="#BBBBBB", lw=0.8)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.03),
        xticks=[0, 0.5, 1],
        yticks=[0, 0.5, 1],
        xlabel="False-positive rate" if kind == "roc" else "Recall",
        ylabel="True-positive rate" if kind == "roc" else "Precision",
    )
    frame(ax, "both")
    panel(
        ax, letter, "Held-out class: ROC" if kind == "roc" else "Held-out class:\nprecision–recall"
    )
    ax.title.set_fontsize(8)
    ax.legend(
        loc="lower right",
        ncol=1,
        frameon=False,
        fontsize=5.7,
        labelspacing=0.2,
        handlelength=1.0,
        handletextpad=0.35,
    )


def operating(ax, metrics, *, false_alarm, letter):
    prefix = "correct_id" if false_alarm else "ood"
    metric = "correct_id_false_alarm" if false_alarm else "ood_detection"
    for i, (method, label, color) in enumerate(METHODS):
        row = method_rows(metrics, method).iloc[0]
        value = float(row[metric])
        low = float(row[metric + "_ci_low"])
        high = float(row[metric + "_ci_high"])
        ax.errorbar(
            value,
            i,
            xerr=[[value - low], [high - value]],
            fmt="o",
            color=color,
            ms=4.5,
            lw=0.9,
            capsize=2,
        )
        ax.text(
            min(high + 0.03, 0.99),
            i,
            f"{int(row[prefix+'_count'])}/{int(row[prefix+'_n'])}",
            fontsize=6.5,
            va="center",
            ha="left" if high < 0.9 else "right",
            color=color,
        )
    ax.set(
        yticks=range(6),
        yticklabels=[
            label.replace("Predictive entropy", "Predictive\nentropy").replace(
                "Expected entropy", "Expected\nentropy"
            )
            for _, label, _ in METHODS
        ],
        ylim=(5.6, -0.6),
        xlim=(-0.02, 0.43) if false_alarm else (-0.02, 1.10),
        xticks=[0, 0.1, 0.2, 0.3, 0.4] if false_alarm else [0, 0.25, 0.5, 0.75, 1],
        xlabel="False-alarm fraction" if false_alarm else "Detection fraction",
    )
    ax.tick_params(axis="y", length=0, labelsize=6.5)
    frame(ax, "both")
    panel(
        ax,
        letter,
        "Correct known calls\nfalsely flagged" if false_alarm else "Held-out class flagged",
    )
    ax.title.set_fontsize(8)


def application(ax, predictor, letter):
    data = read("figure5_application", "full_vs_proteomics_all_class_scores.csv")
    import json

    USED_INPUTS.add("figure5_application/full_vs_proteomics_top3_manifest.json")
    manifest = json.loads(
        (INPUTS / "figure5_application/full_vs_proteomics_top3_manifest.json").read_text()
    )
    # Exact cutoffs are stored with the frozen application scores.
    threshold = float(
        manifest[
            "full_score_cutoff" if predictor == "Full multimodal" else "proteomics_score_cutoff"
        ]
    )
    ids = ["JH68", "JH53", "JH67", "JH51", "JH66"]
    names = ["Cystobactamid", "Nitroxoline", "Myxovirescin", "Xanthocillin", "Darobactin D22"]
    for x, cid in enumerate(ids):
        part = (
            data.loc[data.predictor.eq(predictor) & data.CompoundID.eq(cid)]
            .sort_values("score", ascending=False)
            .head(3)
        )
        assert len(part) == 3
        for j, row in enumerate(part.itertuples()):
            ax.bar(
                x + (j - 1) * 0.23,
                row.score,
                width=0.21,
                color=MOA_COLORS[row.CandidateMoA],
                edgecolor="white",
                linewidth=0.3,
            )
    ax.axhline(threshold, color="#444444", ls=(0, (4, 3)), lw=0.9)
    ax.set(
        xticks=range(5),
        xticklabels=names,
        ylim=(0, 1.04),
        yticks=[0, 0.25, 0.5, 0.75, 1],
        ylabel="Mean matching score",
    )
    ax.tick_params(axis="x", rotation=45, labelsize=6.5)
    for t in ax.get_xticklabels():
        t.set_ha("right")
    frame(ax, "y")
    panel(ax, letter, "Full modalities" if predictor == "Full multimodal" else "Proteomics only")


def main_figure():
    _, metrics, curves = methods_tables()
    fig = plt.figure(figsize=(WIDTH, 8.2))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.10,
        right=0.985,
        top=0.95,
        bottom=0.405,
        width_ratios=[1, 1, 1.18],
        hspace=0.42,
        wspace=0.50,
    )
    curve(fig.add_subplot(grid[0, 0]), metrics, curves, kind="roc", letter="a")
    curve(fig.add_subplot(grid[0, 1]), metrics, curves, kind="pr", letter="b")
    operating(fig.add_subplot(grid[1, 0]), metrics, false_alarm=False, letter="c")
    operating(fig.add_subplot(grid[1, 1]), metrics, false_alarm=True, letter="d")
    # Reclaim the unused gap to the right of the statistical panels without
    # shrinking those panels. Every structure shares the resulting bond scale.
    drawings = fig.add_gridspec(
        3, 1, left=0.685, right=0.985, top=0.95, bottom=0.405, height_ratios=[1, 1, 1], hspace=0.10
    )
    first = drawings[0].subgridspec(1, 2, width_ratios=[0.40, 1], wspace=0.08)
    second = drawings[1].subgridspec(1, 2, width_ratios=[0.65, 1], wspace=0.08)
    items = [
        ("e", "nitroxoline", "Nitroxoline", "Metal chelator", first[0]),
        ("f", "darobactin", "Darobactin D22", "BamA inhibitor", first[1]),
        ("g", "xanthocillin", "Xanthocillin", "Heme dysregulation", second[0]),
        ("h", "myxovirescin", "Myxovirescin", "Signal peptidase\ninhibitor", second[1]),
        ("i", "cystobactamid", "Cystobactamid", "DNA synthesis inhibitor", drawings[2]),
    ]
    for letter, file, name, moa, spec in items:
        pos = spec.get_position(fig)
        a = fig.add_axes([pos.x0, pos.y0 + 0.047, pos.width, pos.height - 0.047])
        reserve(a, f"figure5/{file}.svg")
        fig.text(pos.x0 - 0.015, pos.y1, letter, fontsize=11, va="top")
        name_text = fig.text(
            pos.x0 + pos.width / 2, pos.y0 + 0.034, name, ha="center", va="top", fontsize=6
        )
        moa_text = fig.text(
            pos.x0 + pos.width / 2, pos.y0 + 0.018, moa, ha="center", va="top", fontsize=5.5
        )
        fig._artwork[-1]["label_texts"] = [name_text, moa_text]
    lower = fig.add_gridspec(1, 2, left=0.10, right=0.985, bottom=0.17, top=0.315, wspace=0.22)
    application(fig.add_subplot(lower[0]), "Full multimodal", "j")
    right = fig.add_subplot(lower[1])
    application(right, "Proteomics only", "k")
    right.set_ylabel("")
    right.tick_params(labelleft=False)
    moa_legend(fig, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=5, fontsize=6.6)
    fig.legend(
        handles=[
            Line2D([], [], color="#444444", ls=(0, (4, 3)), label="Internally calibrated cutoff")
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        frameon=False,
        fontsize=7,
    )
    return fig


def final_calls():
    data = read("uncertainty", "six_method_calls.csv")
    data.loc[data.method.eq("fixed_original_meta"), "method"] = "meta_model"
    wanted = {method for method, _, _ in METHODS}
    if not wanted.issubset(data.method):
        baseline = read("uncertainty_baseline", "outer_calls.csv")
        fixed = read("uncertainty", "fixed_outer_calls.csv")
        fixed["method"] = "meta_model"
        data = pd.concat(
            [baseline.loc[~baseline.method.eq("meta_model")], fixed], ignore_index=True
        )
    data = data.loc[data.method.isin(wanted)]
    assert len(data) == 612 and not data.duplicated(["CompoundID", "role", "method"]).any()
    return data


def flag_heatmap(ax, calls, *, held_out, show_labels):
    role = "ood" if held_out else "id"
    data = calls.loc[calls.role.eq(role)]
    if not held_out:
        data = data.loc[data.correct.eq(1)]
    for x, (method, _, color) in enumerate(METHODS):
        rgb = np.array(to_rgb(color))
        for y, moa in enumerate([*CLASSES, "All compounds"]):
            sub = data.loc[data.method.eq(method)]
            if moa != "All compounds":
                sub = sub.loc[sub.TrueMoA.eq(moa)]
            n = len(sub)
            k = int(sub.flag.sum())
            frac = k / n if n else 0
            ax.add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    facecolor=(1 - frac * 0.85) + frac * 0.85 * rgb,
                    edgecolor="white",
                    lw=0.8,
                )
            )
            ax.text(
                x,
                y,
                f"{k}/{n}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if frac > 0.75 else "#222222",
            )
    ax.set(xlim=(-0.5, 5.5), ylim=(9.5, -0.5))
    ax.set_xticks(
        range(6),
        ["Low\nscore", "Pred.\nentropy", "Exp.\nentropy", "Global\nZ", "Class\nZ", "Meta"],
        fontsize=6,
    )
    ax.set_yticks(range(10), [*[SHORT[x] for x in CLASSES], "All"], fontsize=7)
    ax.tick_params(length=0, labelleft=show_labels)
    ax.grid(False)


def classwise_figure():
    calls = final_calls()
    fig = plt.figure(figsize=(WIDTH, 7.5))
    grid = fig.add_gridspec(
        3,
        2,
        left=0.13,
        right=0.98,
        top=0.94,
        bottom=0.09,
        height_ratios=[2.6, 0.9, 1.9],
        hspace=0.34,
        wspace=0.22,
    )
    a = fig.add_subplot(grid[0, 0])
    flag_heatmap(a, calls, held_out=True, show_labels=True)
    panel(a, "a", "Held-out class flagged")
    b = fig.add_subplot(grid[0, 1])
    flag_heatmap(b, calls, held_out=False, show_labels=False)
    panel(b, "b", "Correct known-class false alarms")
    a.title.set_fontsize(8)
    b.title.set_fontsize(8)
    c = fig.add_subplot(grid[1, :])
    c.set_axis_off()
    examples = [("JH10", "Fosfomycin"), ("JH47", "Lolamicin"), ("JH49", "BB78485")]
    # Use actual identities in the saved three-error table, not name guesses.
    errors = calls.loc[calls.role.eq("id") & calls.correct.eq(0)]
    identities = errors.CompoundID.drop_duplicates().tolist()
    names = (
        read("metadata", "dataset_compounds.csv").set_index("compound_id").compound_name.to_dict()
    )
    for y, cid in enumerate(identities):
        c.text(
            -0.64, 2 - y, names[cid].replace(" (calcium)", ""), ha="right", va="center", fontsize=7
        )
        for x, (method, label, color) in enumerate(METHODS):
            row = errors.loc[errors.CompoundID.eq(cid) & errors.method.eq(method)].iloc[0]
            colour = decision_colour(row)
            c.add_patch(
                Rectangle(
                    (x - 0.45, 1.65 - y), 0.9, 0.7, facecolor=colour, edgecolor=colour, lw=0.8
                )
            )
            c.text(
                x,
                2 - y,
                "Flagged" if row.flag else "Missed",
                ha="center",
                va="center",
                fontsize=6,
                color="white",
            )
    c.set(xlim=(-0.55, 5.55), ylim=(-0.65, 3.1))
    for x, (_, label, _) in enumerate(METHODS):
        c.text(x, 2.65, label.replace(" ", "\n"), ha="center", va="bottom", fontsize=6)
    panel(c, "c", "Known-class prediction errors")
    bottom = grid[2, :].subgridspec(1, 3, width_ratios=[1.05, 1.45, 0.85], wspace=0.80)
    d = fig.add_subplot(bottom[0])
    e = fig.add_subplot(bottom[1])
    f = fig.add_subplot(bottom[2])
    cls = read("difficulty", "class_summary.csv")
    low = read("difficulty", "compound_calls.csv")
    draw_difficulty(d, e, f, cls, low)
    return fig


def draw_difficulty(ax_size, ax_structure, ax_pair, classes, calls):
    # The displayed low-score rows and paired outcomes are archived separately.
    pair = read("difficulty", "paired_role_calls.csv")
    if "method" in calls:
        calls = calls.loc[calls.method.eq("low_max_score")]
    if "role" in calls:
        calls = calls.loc[calls.role.eq("ood")]
    size = classes.copy()
    if "method" in size:
        size = size.loc[size.method.eq("low_max_score")]
    if "scope" in size:
        size = size.loc[size.scope.eq("ood")]
    offsets = {}
    for _, group in size.groupby("class_size"):
        positions = np.linspace(-0.16, 0.16, len(group)) if len(group) > 1 else [0]
        offsets.update(dict(zip(group.TrueMoA, positions)))
    for row in size.itertuples():
        name = getattr(row, "TrueMoA", getattr(row, "class_name", None))
        n = getattr(row, "n", getattr(row, "class_size", 0))
        frac = getattr(row, "flag_fraction", getattr(row, "flag_rate", 0))
        x = n + offsets[name]
        shift = {"DNA replication": -12}.get(name, 4)
        ax_size.scatter(x, frac * 100, color="#244A7C", s=20)
        right_aligned = n == 9
        position = (-3 if right_aligned else 3, shift)
        label_size = 5.5
        if name == "Nitrofurans":
            # Keep the longer label inside the panel and clear of the 40% tick.
            position = (4, 10)
            label_size = 6
        ax_size.annotate(
            SHORT.get(name, name),
            (x, frac * 100),
            xytext=position,
            ha="right" if right_aligned else "left",
            textcoords="offset points",
            fontsize=label_size,
        )
    ax_size.set(
        xlabel="Compounds in MoA class",
        ylabel="LOCO flagged (%)",
        ylim=(-5, 108),
        xticks=[3, 4, 5, 9],
        xlim=(2.7, 9.4),
    )
    panel(ax_size, "d", "Class size")
    frame(ax_size, "both")
    similarity = "reference_same_class_similarity"
    for y, moa in enumerate(CLASSES):
        part = calls.loc[calls.TrueMoA.eq(moa)]
        for offset, row in zip(np.linspace(-0.16, 0.16, len(part)), part.itertuples()):
            flag = bool(row.flag)
            ax_structure.scatter(
                getattr(row, similarity),
                y + offset,
                s=17,
                facecolor="#244A7C" if flag else "white",
                edgecolor="#244A7C" if flag else "#777777",
                lw=0.7,
            )
    ax_structure.set(
        yticks=range(9),
        yticklabels=[SHORT[x] for x in CLASSES],
        xlim=(-0.02, 0.92),
        xlabel="Same-class ECFP similarity",
        xticks=[0, 0.25, 0.5, 0.75],
    )
    ax_structure.invert_yaxis()
    ax_structure.tick_params(axis="y", length=0, labelsize=6)
    panel(ax_structure, "e", "Structural isolation")
    frame(ax_structure, "x")
    ax_structure.legend(
        handles=[
            Line2D([], [], marker="o", color="#244A7C", ls="none", label="Flagged"),
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="white",
                color="#777777",
                ls="none",
                label="Not flagged",
            ),
        ],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
        fontsize=6,
        handletextpad=0.3,
    )
    selected = pair.loc[pair.method.eq("low_max_score")]
    matrix = (
        pd.crosstab(selected.id_flag, selected.ood_flag)
        .reindex(index=[False, True], columns=[False, True], fill_value=0)
        .to_numpy()
    )
    assert matrix.sum() == 51
    ax_pair.imshow(matrix, cmap="Blues", vmin=0, vmax=33)
    for (y, x), v in np.ndenumerate(matrix):
        ax_pair.text(
            x,
            y,
            str(v),
            ha="center",
            va="center",
            fontsize=12,
            color="white" if v > 24 else "#222222",
        )
    ax_pair.set(
        xticks=[0, 1],
        xticklabels=["No", "Yes"],
        yticks=[0, 1],
        yticklabels=["No", "Yes"],
        xlabel="Class omitted:\nflagged?",
        ylabel="Class present:\nflagged?",
    )
    ax_pair.set_anchor("N")
    ax_pair.grid(False)
    panel(ax_pair, "f", "Paired flags")


def examples_figure():
    calls = final_calls()
    members = read("uncertainty_examples", "figure5_si_member_scores.csv.gz")
    cases = [
        ("JH32", "ood", "Gepotidacin — DNA class withheld"),
        ("JH47", "id", "Lolamicin"),
        ("JH61", "id", "Debio-1452-NH₂"),
        ("JH40", "id", "Triclosan"),
    ]
    fig = plt.figure(figsize=(WIDTH, 7.0))
    grid = fig.add_gridspec(
        2, 2, left=0.15, right=0.98, bottom=0.06, top=0.94, hspace=0.20, wspace=0.4
    )
    for i, (cid, role, title) in enumerate(cases):
        sub = grid[i // 2, i % 2].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.18)
        ax = fig.add_subplot(sub[0])
        part = members.loc[members.CompoundID.eq(cid) & members.role.eq(role)]
        decisions = calls.loc[calls.CompoundID.eq(cid) & calls.role.eq(role)]
        assert decisions.PredictedMoA.nunique() == decisions.TrueMoA.nunique() == 1
        predicted = decisions.PredictedMoA.iloc[0]
        truth = decisions.TrueMoA.iloc[0]
        colour = MOA_COLORS[predicted]
        cutoff = -float(decisions.loc[decisions.method.eq("low_max_score"), "threshold"].iloc[0])
        for j, sample in enumerate(sorted(part.SampleID.unique())):
            values = part.loc[part.SampleID.eq(sample), "match_score"].to_numpy(float)
            q = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
            result = ax.bxp(
                [dict(q1=q[1], med=q[2], q3=q[3], whislo=q[0], whishi=q[4], fliers=[])],
                positions=[j],
                vert=False,
                widths=0.32,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=to_rgba(colour, 0.65), edgecolor="#333333", lw=0.8),
                medianprops=dict(color="#222222", lw=0.8),
                whiskerprops=dict(color="#333333", lw=0.8),
                capprops=dict(color="#333333", lw=0.8),
            )
            ax.scatter(values.mean(), j, s=20, facecolor="white", edgecolor=colour, zorder=4)
        ax.text(
            0.02,
            0.99,
            f"True: {SHORT[truth]} | Predicted: {SHORT[predicted]}",
            transform=ax.transAxes,
            fontsize=6.2,
            color="#333333",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
            zorder=5,
        )
        ax.scatter(part.match_score.mean(), 4, s=27, marker="D", color="#222222")
        ax.axhline(3.5, color="#BBBBBB", lw=0.8)
        ax.axvline(cutoff, color="#444444", ls="--", lw=0.9)
        ax.set(
            yticks=range(5),
            yticklabels=["Replicate 1", "Replicate 2", "Replicate 3", "Replicate 4", "Mean"],
            ylim=(4.5, -0.6),
            xlim=(-0.02, 1.02),
            xticks=[0, 0.25, 0.5, 0.75, 1],
            xlabel="MoA matching score",
        )
        frame(ax, "x")
        panel(ax, chr(97 + i), title)
        ax.title.set_fontsize(8)
        matrix = fig.add_subplot(sub[1])
        matrix.set_axis_off()
        for j, (method, label, color) in enumerate(METHODS):
            row = decisions.loc[decisions.method.eq(method)].iloc[0]
            flag = bool(row.flag)
            decision = decision_colour(row)
            matrix.add_patch(
                Rectangle(
                    (j - 0.45, -0.2), 0.9, 0.65, facecolor=decision, edgecolor=decision, lw=0.8
                )
            )
            matrix.text(
                j,
                0.125,
                "Flag" if flag else "Pass",
                ha="center",
                va="center",
                fontsize=6,
                color="white",
            )
            matrix.text(j, 0.65, label.replace(" ", "\n"), ha="center", va="bottom", fontsize=5.8)
        matrix.set(xlim=(-0.55, 5.55), ylim=(-0.3, 1.4))
    return fig
