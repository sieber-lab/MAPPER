"""Collision-free growth-curve end labels, without modifying trajectories."""

import numpy as np
from sklearn.isotonic import IsotonicRegression


def register(fig, axes, data, classes, font_size=6):
    from .overview_helpers import short_moa

    frame = data.assign(_class=data.mode_of_action.map(short_moa))
    panels = []
    for ax, moa in zip(np.asarray(axes).flat, classes):
        labels = []
        for text in ax.texts:
            value = text.get_text()
            if not value.isdigit() or text.get_position()[0] < 20:
                continue
            subset = frame.loc[frame._class.eq(moa) & frame.bucket.eq(int(value))]
            means = subset.groupby("hour").od.mean().dropna()
            assert len(means)
            text.set_fontsize(font_size)
            text.set_ha("left")
            text.set_va("center")
            labels.append(dict(text=text, x=float(means.index[-1]), y=float(means.iloc[-1])))
        panels.append(dict(ax=ax, labels=labels))
    fig._growth_panels = getattr(fig, "_growth_panels", []) + panels


def arrange(fig):
    panels = getattr(fig, "_growth_panels", [])
    if not panels:
        return []
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    reports = []
    for panel in panels:
        ax = panel["ax"]
        labels = panel["labels"]
        if not labels:
            continue
        dimensions = [item["text"].get_window_extent(renderer) for item in labels]
        widest = max(box.width for box in dimensions)
        # Reserve a real column after the measured 24-h endpoint. Ticks remain
        # at measured time points; connector lines identify displaced labels.
        reserve = widest + 8 * fig.dpi / 72
        fraction = reserve / ax.bbox.width
        assert fraction < 0.45, "Growth panel too narrow for readable end labels"
        xmax = max(26.5, max(item["x"] for item in labels) / (1 - fraction))
        ax.set_xlim(0, xmax)
        desired = np.array([ax.transData.transform((item["x"], item["y"]))[1] for item in labels])
        order = np.argsort(desired, kind="stable")
        items = [labels[i] for i in order]
        heights = np.array([dimensions[i].height for i in order])
        gaps = (heights[:-1] + heights[1:]) / 2 + 1.8 * fig.dpi / 72
        cumulative = np.r_[0, np.cumsum(gaps)]
        lower = ax.bbox.y0 + heights[0] / 2 + 2
        upper = ax.bbox.y1 - heights[-1] / 2 - 2
        assert upper - lower >= cumulative[-1], "Growth panel too short for readable end labels"
        fit = IsotonicRegression(increasing=True).fit_transform(
            np.arange(len(items)), desired[order] - cumulative
        )
        positions = np.clip(fit, lower, upper - cumulative[-1]) + cumulative
        label_x = ax.bbox.x1 - widest - 2
        for item, target in zip(items, positions):
            x, y = ax.transData.inverted().transform((label_x, target))
            item["text"].set_position((x, y))
            if abs(target - ax.transData.transform((item["x"], item["y"]))[1]) > 1:
                ax.plot(
                    [item["x"] + 0.12, x - 0.15],
                    [item["y"], y],
                    color=item["text"].get_color(),
                    lw=0.4,
                    alpha=0.7,
                    zorder=3,
                    clip_on=True,
                )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [item["text"].get_window_extent(renderer) for item in items]
        overlaps = sum(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
        assert overlaps == 0, "Overlapping growth-curve labels remain"
        reports.append(dict(labels=len(items), overlaps=int(overlaps), xmax=float(xmax)))
    return reports
