"""Dot version of the class-mean growth curves in Fig. 2b and Extended Data Fig. 1."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .growth_labels import register
from .overview_helpers import plot_growth_facets_axes, short_moa
from .style import WIDTH, frame, read


# Panel order and means match the two published figures. Here the visual key
# distinguishes doses within a panel; the panel titles already name the class.
CLASSES = (
    ("DNA repl.", "DNA synthesis"), ("Fatty acid", "Fatty acid synthesis"),
    ("Protein syn.", "Protein synthesis"), ("RNA synth.", "RNA synthesis"),
    ("Cell wall", "Cell wall biosynthesis"), ("Folic acid", "Folate metabolism"),
    ("Membrane", "Membrane disruption"), ("Nitrofurans", "Nitrofurans"),
    ("LPS synth.", "LPS synthesis"),
)
DOSE_STYLES = {
    1: ("#205B96", "o"), 5: ("#CA5100", "^"), 10: ("#007F68", "s"),
    15: ("#82549A", "D"), 20: ("#AA365A", "v"), 50: ("#906500", "P"),
    100: ("#343434", "X"),
}
DOSES = tuple(DOSE_STYLES)
CONTROL_COLOUR = "#747474"


def observations(ax, rows, dose, centre, half_width):
    # Fixed offsets keep a given antibiotic in the same slot at every time.
    ids = sorted(rows.compound_id.unique())
    offsets = np.linspace(-half_width, half_width, len(ids)) if len(ids) > 1 else [0.]
    x = rows.hour + centre + rows.compound_id.map(dict(zip(ids, offsets)))
    colour, marker = DOSE_STYLES[dose] if dose else (CONTROL_COLOUR, "o")
    ax.scatter(x, rows.od, s=13 if dose else 6, c=[colour], marker=marker,
               edgecolors="white", linewidths=.26 if dose else .16,
               alpha=.90 if dose else .60, zorder=3.5,
               gid=f"observations-dose-{dose}")


def individual_growth():
    data = read("overview", "growth_profiles.csv")
    keys = ["compound_id", "bucket", "hour"]
    if data.duplicated(keys).any() or data[keys + ["od"]].isna().any().any():
        raise ValueError("Expected one nonmissing OD per compound/dose/time after averaging duplicates.")
    data = data.assign(moa=data.mode_of_action.map(short_moa))
    names = [key for key, _ in CLASSES]
    if set(data.moa) != set(names) or set(data.bucket) != {0, *DOSES}:
        raise ValueError("Unexpected class or MIC multiple.")
    fig, axes = plt.subplots(5, 2, figsize=(WIDTH, 8.6))
    fig.subplots_adjust(left=.09, right=.985, top=.935, bottom=.058, wspace=.25, hspace=.46)
    # The extra axis is reserved for the repeated pooled control, not a class.
    plot_growth_facets_axes(axes, data, names, buckets=DOSES, ymax=4.2, mean_only=True)
    for i, (key, title) in enumerate(CLASSES):
        ax = axes.flat[i]
        curves = data.loc[(data.moa == key) & (data.bucket != 0)]
        groups = curves.groupby("bucket", sort=True)
        centres = np.linspace(-.36, .36, len(groups)) if len(groups) > 1 else [0.]
        # Do not let adjacent doses swap order or a point approach the next hour.
        half_width = min(.075, .30 / max(1, len(groups) - 1))
        for centre, (dose, rows) in zip(centres, groups):
            colour, _ = DOSE_STYLES[dose]
            line, = [line for line in ax.lines if line.get_gid() == f"mean-dose-{dose}"]
            line.set(color=colour, linewidth=1.0, linestyle="-", alpha=.90)
            label, = [t for t in ax.texts if t.get_text() == str(dose)]
            label.set_color(colour)
            observations(ax, rows, dose, centre, half_width)
        ax.set_title(title, fontsize=8, pad=4)
    register(fig, axes, data, names, font_size=6.5)
    ctrl = axes.flat[-1]
    controls = data.loc[data.bucket == 0]
    means = controls.groupby("hour", sort=True).od.mean()
    ctrl.plot(means.index, means, color="0.35", lw=1.6, alpha=.95, gid="mean-control")
    observations(ctrl, controls, 0, 0., .34)
    ctrl.set_title("Pooled solvent controls", fontsize=8, pad=4)
    for i, ax in enumerate(axes.flat):
        frame(ax, "y")
        control_line, = [line for line in ax.lines if line.get_gid() == "mean-control"]
        control_line.set(color=CONTROL_COLOUR, linewidth=1.0)
        ax.set(xlim=(-.75, 26.5), ylim=(-.10, 4.2), yticks=[0, 1, 2, 3, 4],
               xticks=np.arange(0, 25, 4), xlabel="", ylabel="")
        ax.tick_params(labelsize=6.5, labelbottom=i >= 8, labelleft=i % 2 == 0)
    handles = [Line2D([], [], color=c, marker=m, markersize=4.6, lw=1.,
                      label=f"{dose}×") for dose, (c, m) in DOSE_STYLES.items()]
    handles.append(Line2D([], [], color=CONTROL_COLOUR, lw=1., label="Solvent control"))
    fig.legend(handles=handles, title="Treatment concentration (× MIC)",
               loc="upper center", bbox_to_anchor=(.52, .998), ncol=len(handles),
               frameon=False, fontsize=7, title_fontsize=7.5, columnspacing=1.25,
               handlelength=1.55, handletextpad=.45)
    fig.supxlabel("Time (h)", fontsize=8.5, y=.014)
    fig.supylabel("Optical density (600 nm)", fontsize=8.5, x=.013)
    return fig
