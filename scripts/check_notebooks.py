"""Run each final figure notebook in a separate fresh Python kernel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("figures", nargs="*")
    selected = parser.parse_args().figures
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if selected:
        notebooks = [path for path in notebooks if path.stem in selected]
    if not notebooks:
        raise ValueError("No notebooks selected")
    output = ROOT / "executed_notebooks"
    output.mkdir(exist_ok=True)
    report = ROOT / "results/verification/notebooks.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in notebooks:
        started = time.monotonic()
        notebook = nbformat.read(source, as_version=4)
        try:
            client = NotebookClient(
                notebook,
                timeout=600,
                kernel_name="python3",
                resources={"metadata": {"path": str(ROOT / "notebooks")}},
            )
            client.execute()
            nbformat.write(notebook, output / source.name)
            row = dict(
                figure=source.stem, status="passed", seconds=round(time.monotonic() - started, 2)
            )
        except Exception:
            row = dict(figure=source.stem, status="failed", traceback=traceback.format_exc())
        rows.append(row)
        report.write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if row["status"] == "failed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
