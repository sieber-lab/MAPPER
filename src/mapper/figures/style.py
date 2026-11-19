"""One presentation standard, expressed at final publication dimensions.

The MoA palette and statistical colour semantics follow the paper's original
notebooks. Dense matrix labels may use 5.5--6.5 pt; normal labels use 8 pt.
No renderer is allowed to change score aggregation or significance definitions.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[3]
INPUTS = PACKAGE / "data/figure_inputs"
OUTPUTS = PACKAGE / "figures"
USED_INPUTS = set()
WIDTH = 180 / 25.4  # full-width figures, in inches
MOA_COLORS = {
    "Fatty acid synthesis": "#264653",
    "Nitrofurans": "#287271",
    "Folic acid metabolism": "#2A9D8F",
    "Membrane disruption": "#7FB48A",
    "RNA Synthesis": "#E9C46A",
    "lipopolysaccharide synthesis": "#C6902E",
    "DNA replication": "#F28E2B",
    "Cell wall biosynthesis": "#E76F51",
    "Protein synthesis": "#D1495B",
}
CLASSES = tuple(MOA_COLORS)
SHORT = dict(
    zip(
        CLASSES,
        [
            "Fatty acid",
            "Nitrofurans",
            "Folate",
            "Membrane",
            "RNA",
            "LPS",
            "DNA",
            "Cell wall",
            "Protein",
        ],
    )
)
FEATURE_COLORS = {"proteomics": "#2A9D8F", "ecfp": "#A9AFB4", "ecfp_proteomics": "#244A7C"}
METHODS = (
    ("low_max_score", "Low score", "#244A7C"),
    ("high_predictive_entropy", "Predictive entropy", "#346E9B"),
    ("high_expected_entropy", "Expected entropy", "#39949A"),
    ("z_gate_global", "Global Z-gate", "#E2A777"),
    ("z_gate_class", "Class Z-gate", "#D77868"),
    ("meta_model", "Meta-model", "#B84A59"),
)
OPERATORS = {"Operator 1": "#AEC7E8", "Operator 2": "#1F77B4"}
CONDITIONS = {"No text": "#676E56", "Repeated samples, no text": "#E0A458", "With text": "#9BC53D"}
STAT_COLORS = {"best": "#1F77B4", "not different": "#9E9E9E", "worse": "#C45C5C"}
MODEL_HATCHES = ("", "/////", "xxxx")


def apply_style():
    """Apply the same normal-weight fonts, outlines and white-grid style."""
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Arimo", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1,
            "lines.markersize": 4,
            "patch.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "hatch.linewidth": 0.45,
            "text.usetex": False,
            "figure.dpi": 130,
            "savefig.dpi": 600,
        }
    )


def read(group, name, **kwargs):
    """Read an archived plotting table without lossy float reformatting."""
    path = INPUTS / group / name
    USED_INPUTS.add(str(path.relative_to(INPUTS)))
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def canonical(label):
    variants = {
        "Fatty acid": "Fatty acid synthesis",
        "Folic acid": "Folic acid metabolism",
        "Membrane": "Membrane disruption",
        "RNA synth.": "RNA Synthesis",
        "RNA synthesis": "RNA Synthesis",
        "LPS synth.": "lipopolysaccharide synthesis",
        "LPS synthesis": "lipopolysaccharide synthesis",
        "DNA repl.": "DNA replication",
        "DNA synthesis": "DNA replication",
        "Cell wall": "Cell wall biosynthesis",
        "Protein syn.": "Protein synthesis",
    }
    return variants.get(str(label), str(label))


def frame(ax, grid="y"):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    ax.tick_params(direction="out", length=3, width=0.8, top=False, right=False)
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid, color="#D9D9D9", linewidth=0.6)


def panel(ax, letter, title=None):
    ax.text(
        -0.12,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        ha="left",
        va="bottom",
        fontweight="normal",
        clip_on=False,
    )
    if title:
        ax.set_title(title, loc="left", fontsize=9, pad=9)


def moa_legend(fig, **kwargs):
    handles = [Patch(facecolor=color, label=SHORT[label]) for label, color in MOA_COLORS.items()]
    return fig.legend(handles=handles, frameon=False, **kwargs)


def box(
    ax,
    arrays,
    positions=None,
    colors=None,
    width=0.6,
    points=False,
    horizontal=False,
    labels=None,
    whis=None,
):
    if positions is None:
        positions = np.arange(len(arrays))
    result = ax.boxplot(
        arrays,
        positions=positions,
        widths=width,
        patch_artist=True,
        vert=not horizontal,
        whis=((0, 100) if points else 1.5) if whis is None else whis,
        showfliers=not points,
        medianprops=dict(color="#222222", linewidth=0.8),
        boxprops=dict(edgecolor="#333333", linewidth=0.8),
        whiskerprops=dict(color="#333333", linewidth=0.8),
        capprops=dict(color="#333333", linewidth=0.8),
        flierprops=dict(marker="o", markersize=2, markeredgecolor="#333333", alpha=0.55),
    )
    for i, patch in enumerate(result["boxes"]):
        patch.set_facecolor(colors[i] if colors is not None else "#AEC7E8")
    if points:
        rng = np.random.default_rng(20261105)
        for position, values in zip(positions, arrays):
            values = np.asarray(values, float)
            jitter = position + rng.uniform(-width * 0.28, width * 0.28, len(values))
            ax.scatter(
                values if horizontal else jitter,
                jitter if horizontal else values,
                s=8,
                color="#555555",
                alpha=0.6,
                linewidths=0,
                zorder=3,
            )
    if labels is not None:
        if horizontal:
            ax.set_yticks(positions, labels)
        else:
            ax.set_xticks(positions, labels)
    return result


def validate_style(fig):
    """Standardize data-axis frames but preserve unboxed artwork/table axes."""
    from matplotlib.text import Text

    font = next(
        (
            name
            for name in ["Arial", "Arimo", "Helvetica", "DejaVu Sans"]
            if any(x.name == name for x in font_manager.fontManager.ttflist)
        ),
        "DejaVu Sans",
    )
    for text in fig.findobj(match=Text):
        text.set_fontfamily(font)
    for ax in fig.axes:
        if ax.axison and not getattr(ax, "_unboxed", False):
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#333333")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    hidden_tick_labels = set()
    for ax in fig.axes:
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lo, hi = sorted(limits)
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if not lo <= tick.get_loc() <= hi:
                    hidden_tick_labels.update([id(tick.label1), id(tick.label2)])
    outside = []
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text() or id(text) in hidden_tick_labels:
            continue
        bbox = text.get_window_extent(renderer)
        if (
            bbox.x0 < page.x0 - 1
            or bbox.y0 < page.y0 - 1
            or bbox.x1 > page.x1 + 1
            or bbox.y1 > page.y1 + 1
        ):
            outside.append(dict(text=text.get_text(), bbox_pixels=[float(v) for v in bbox.extents]))
    return dict(
        font_family=font,
        axes_linewidth_pt=0.8,
        axis_label_pt=8,
        normal_tick_pt=7,
        normal_title_pt=9,
        panel_letter_pt=11,
        text_outside_page=outside,
    )


def provenance(figure_id, destination, metadata):
    """Record exact input/code identities alongside each final figure."""
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    record = dict(
        figure=figure_id,
        style=metadata,
        source_manifest_sha256=sha(INPUTS / "manifest.json"),
        source_files=len(manifest["files"]),
        new_model_fits=0,
        new_statistical_tests=0,
        used_tables={name: sha(INPUTS / name) for name in sorted(USED_INPUTS)},
        code_sha256={
            str(p.relative_to(PACKAGE)): sha(p)
            for p in sorted((PACKAGE / "src/mapper/figures").glob("*.py"))
        },
    )
    (destination / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
