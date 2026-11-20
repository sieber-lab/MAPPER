"""One explicit entry point per final manuscript figure."""

from __future__ import annotations

from .style import apply_style, USED_INPUTS
from .artwork import export

FIGURES = {
    **{f"main{n:02d}": f"Figure {n}" for n in range(2, 7)},
    **{f"extended{n:02d}": f"Extended Data Figure {n}" for n in range(1, 8)},
    **{f"supplementary{n:02d}": f"Supplementary Figure S{n}" for n in range(1, 13)},
}


def build_figure(figure_id):
    if figure_id not in FIGURES:
        raise KeyError(figure_id)
    USED_INPUTS.clear()
    apply_style()
    if figure_id.startswith("main"):
        from .main import build
    elif figure_id.startswith("extended"):
        from .extended import build
    else:
        from .supplementary import build
    figure = build(int(figure_id[-2:]))
    return export(figure, figure_id)
