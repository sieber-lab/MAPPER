"""Generate complete paper figures from checked numerical tables and artwork."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
from mapper.figures.registry import FIGURES, build_figure
from mapper.figures.artwork import render_svg
from mapper.figures.style import INPUTS, OUTPUTS, WIDTH, sha


def artwork_figure():
    source = INPUTS / "artwork/figure1/figure1.svg"
    destination = OUTPUTS / "main01"
    destination.mkdir(parents=True, exist_ok=True)
    svg = destination / "main01.svg"
    shutil.copy2(source, svg)
    render_svg(svg, destination / "main01.pdf", width_pt=WIDTH * 72)
    render_svg(svg, destination / "main01.png", png=True, dpi=600, width_pt=WIDTH * 72)
    render_svg(svg, destination / "main01_preview.png", png=True, dpi=150, width_pt=WIDTH * 72)
    (destination / "provenance.json").write_text(
        json.dumps(
            dict(artwork_only=True, source=str(source.relative_to(ROOT)), sha256=sha(source)),
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("figures", nargs="*", choices=["all", "main01", *FIGURES])
    selected = parser.parse_args().figures or ["all"]
    if "all" in selected:
        selected = ["main01", *FIGURES]
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        if sha(INPUTS / name) != record["sha256"]:
            raise ValueError("Changed figure input: " + name)
    report = []
    for figure_id in selected:
        start = time.monotonic()
        artwork_figure() if figure_id == "main01" else build_figure(figure_id)
        report.append(
            dict(figure=figure_id, status="ok", seconds=round(time.monotonic() - start, 2))
        )
        print(json.dumps(report[-1]), flush=True)
    destination = ROOT / "results/verification"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "figure_build.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
