from __future__ import annotations

from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

SAVE_DIR: Path | None = None

LABEL_SIZE = 8
TITLE_SIZE = 10
TICK_SIZE = 8
LEGEND_SIZE = 7
LEGEND_TITLE_SIZE = 7

def apply_style() -> None:
    """Apply consistent paper styling across all plots."""
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.0,
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "0.2",
            "grid.color": "0.85",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "boxplot.flierprops.markersize": 3,
            "boxplot.boxprops.linewidth": 0.8,
            "boxplot.whiskerprops.linewidth": 0.8,
            "boxplot.capprops.linewidth": 0.8,
            "boxplot.medianprops.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
        },
    )

    plt.rcParams.update({
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def style_axes(ax) -> None:
    """Force boxed axes and consistent ticks."""
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=3,
        width=0.8,
        color="0.25",
        bottom=True,
        top=False,
        left=True,
        right=False,
    )
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("0.2")

def normalize_figure(fig) -> None:
    """Force consistent font sizes and boxed axes before saving."""
    for ax in fig.axes:
        ax.title.set_fontsize(TITLE_SIZE)
        ax.title.set_fontweight("normal")
        ax.xaxis.label.set_fontsize(LABEL_SIZE)
        ax.yaxis.label.set_fontsize(LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        if not getattr(ax, "_skip_style_axes", False):
            style_axes(ax)

        legend = ax.get_legend()
        if legend is not None:
            legend.set_title(legend.get_title().get_text(), prop={"size": LEGEND_TITLE_SIZE})
            for txt in legend.get_texts():
                txt.set_fontsize(LEGEND_SIZE)

    for legend in fig.legends:
        legend.set_title(legend.get_title().get_text(), prop={"size": LEGEND_TITLE_SIZE})
        for txt in legend.get_texts():
            txt.set_fontsize(LEGEND_SIZE)


def set_save_dir(repo_root: Path, figure_id: str) -> Path:
    """Set the default output directory for figures in a notebook."""
    global SAVE_DIR
    root = Path(repo_root).resolve()
    # Support both invocation styles:
    # 1) repo_root points directly to code_repository
    # 2) repo_root points to parent dir containing code_repository
    if (root / "images").exists() and (root / "notebooks").exists():
        base = root
    elif (root / "code_repository").exists():
        base = root / "code_repository"
    else:
        base = root

    save_dir = base / "images" / figure_id
    save_dir.mkdir(parents=True, exist_ok=True)
    SAVE_DIR = save_dir
    return save_dir


def _normalize_path(path: str | Path) -> Path:
    base = Path(path)
    if base.suffix:
        base = base.with_suffix("")
    if SAVE_DIR is not None:
        base = SAVE_DIR / base.name
    return base



def save_figure(fig, path: str | Path, dpi: int = 600, **kwargs) -> None:
    base = _normalize_path(path)
    kwargs.pop("dpi", None)
    kwargs.pop("format", None)
    kwargs["bbox_inches"] = "tight"

    normalize_figure(fig)

    dpi = 600

    with mpl.rc_context({
        "svg.fonttype": "none",
        "text.usetex": False,          # key if style enables LaTeX
        "mathtext.fontset": "dejavusans",
        # optional but helps consistency:
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    }):
        fig.savefig(base.with_suffix(".png"), dpi=dpi, **kwargs)
        fig.savefig(base.with_suffix(".pdf"), **kwargs)
        fig.savefig(base.with_suffix(".svg"), **kwargs)
