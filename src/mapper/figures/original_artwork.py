"""Place the author's original compound composition at its native aspect."""

import json
from .style import INPUTS, USED_INPUTS, read
from .artwork import reserve


def structure_panel(fig, key, rect):
    USED_INPUTS.add("original_structure_layout.json")
    item = json.loads((INPUTS / "original_structure_layout.json").read_text())[key]
    x, y, w, h = rect
    page_w, page_h = fig.get_size_inches() * 72
    scale = min(w * page_w / item["width"], h * page_h / item["height"])
    actual_w = item["width"] * scale / page_w
    actual_h = item["height"] * scale / page_h
    x = x + (w - actual_w) / 2
    y = y + (h - actual_h) / 2
    ax = fig.add_axes([x, y, actual_w, actual_h])
    reserve(ax, item["artwork"])

    def position(px, py):
        return x + px * scale / page_w, y + actual_h - py * scale / page_h

    for letter, (px, py) in item["letters"].items():
        fig.text(*position(px, py), letter, ha="left", va="top", fontsize=11)
    annotations = read("similarities", "calculated_annotations.csv").set_index("compound_id")
    for entry in item["annotations"]:
        row = annotations.loc[entry["compound_id"]]
        annotation_x, annotation_y = position(entry["x"], entry["y"])
        name = fig.text(
            annotation_x,
            annotation_y + 2 / page_h,
            row["name"],
            ha="left",
            va="bottom",
            fontsize=6.2,
        )
        statistics = fig.text(
            annotation_x,
            annotation_y,
            f"Max. Tanimoto Similarity: {row.tanimoto:.2f}\nMax. MCS: {row.mcs:.2f}",
            ha="left",
            va="top",
            fontsize=6.2,
            linespacing=1.12,
        )
        statistics._focal_compound_name = name
    return dict(
        x=x,
        y=y,
        width=actual_w,
        height=actual_h,
        scale=scale,
        source_width=item["width"],
        source_height=item["height"],
        original_crop=item["original_crop"],
    )


def adjacent_plot(fig, panel, rectangle):
    """Map an original adjacent data-panel rectangle to the same composition."""
    left, top, right, bottom = rectangle
    page_w, page_h = fig.get_size_inches() * 72
    x = panel["x"] + (left - panel["original_crop"][0]) * panel["scale"] / page_w
    y = (
        panel["y"]
        + panel["height"]
        - (bottom - panel["original_crop"][1]) * panel["scale"] / page_h
    )
    return fig.add_axes(
        [x, y, (right - left) * panel["scale"] / page_w, (bottom - top) * panel["scale"] / page_h]
    )


def adjacent_header(fig, art, ax, letter, name, source_y):
    page_w, page_h = fig.get_size_inches() * 72
    y = art["y"] + art["height"] - (source_y - art["original_crop"][1]) * art["scale"] / page_h
    fig.text(ax.get_position().x0 - 14 / page_w, y, letter, ha="left", va="top", fontsize=11)
    title = fig.text(ax.get_position().x0, y, name, ha="left", va="top", fontsize=8)
    # Keep the original header alignment while allowing enough room above the
    # numerical axes for an 8-pt title. One original panel has only a 2-pt gap.
    position = ax.get_position()
    top = min(position.y1, y - 10 / page_h)
    ax.set_position([position.x0, position.y0, position.width, top - position.y0])
    ax._comparison_header = title
