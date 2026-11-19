"""Small layout controls shared by the publication figures."""


def left_tick_labels(ax, figure_x=0.03):
    """Align the label column to one page position, not to the plot border."""
    target = ax.figure.transFigure.transform((figure_x, 0))[0]
    for label in ax.get_yticklabels():
        x = label.get_transform().inverted().transform((target, 0))[0]
        label.set_x(x)
        label.set_ha("left")
        label.set_multialignment("left")


def plain_colorbar(colorbar):
    """Show only labelled ticks; dense BoundaryNorm minor ticks form a stripe."""
    colorbar.minorticks_off()
    colorbar.ax.tick_params(
        which="minor", length=0, bottom=False, top=False, left=False, right=False
    )
    colorbar.outline.set_visible(False)
