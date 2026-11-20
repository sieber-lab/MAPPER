"""Portable plotting inputs for the matched three-representation control.

The new archive supplements, rather than modifies, the original input manifest.
No models, thresholds or significance masks are fitted by this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .style import INPUTS, USED_INPUTS, read, sha, frame

GROUP = "external_representation"
DATASETS = ("eclipse", "subanovic", "yu")
REPRESENTATIONS = ("intensity", "significant_intensity", "fold_change")
STYLE = (
    ("intensity", "All intensities", "#9E9E9E", "o"),
    ("significant_intensity", "Significant intensities", "#C45C5C", "s"),
    ("fold_change", "873-protein log₂FC", "#1F77B4", "D"),
)


def check_tables(calls, metrics):
    """Recompute each displayed metric from treatment-level predictions."""
    if len(calls) != 78 or set(calls.representation) != set(REPRESENTATIONS):
        raise ValueError("Expected 26 treatments in each of three representations")
    if calls.duplicated(["representation", "case_key"]).any():
        raise ValueError("Duplicated treatment predictions")
    if set(calls.dataset) != set(DATASETS):
        raise ValueError("Only the three requested datasets may enter this comparison")
    if not calls.compound_excluded.all() or not calls.seeds.eq(5).all():
        raise ValueError("Antibiotic exclusions and five-seed aggregation must be matched")
    for column in ["expected", "context", "biological_profiles", "train_compounds"]:
        if not calls.groupby("case_key")[column].nunique().eq(1).all():
            raise ValueError("Representations have different evaluation sets")
    if metrics.duplicated(["representation", "dataset"]).any():
        raise ValueError("Duplicated metric summary")
    summary = metrics.set_index(["representation", "dataset"])
    for representation in REPRESENTATIONS:
        base = calls.loc[calls.representation.eq(representation)]
        for dataset in (*DATASETS, "ALL"):
            part = base if dataset == "ALL" else base.loc[base.dataset.eq(dataset)]
            correct = part.predicted.eq(part.expected)
            expected = dict(
                treatments=len(part),
                correct_top1=int(correct.sum()),
                top1_accuracy=float(correct.mean()),
                balanced_top1=float(correct.groupby(part.expected).mean().mean()),
                cell_wall_assignments=int(part.predicted.eq("Cell wall biosynthesis").sum()),
            )
            row = summary.loc[(representation, dataset)]
            for name, value in expected.items():
                np.testing.assert_allclose(float(row[name]), value, atol=1e-12, rtol=0)
    return summary


def archive_run(source):
    """Freeze the small completed reports, not weights or historical run folders."""
    source = Path(source)
    verified = json.loads((source / "verification.json").read_text())
    if verified["new_fits_verified"] != 90 or verified["treatments_per_representation"] != 26:
        raise ValueError("The matched run has not completed verification")
    manifest = json.loads((source / "manifest.json").read_text())
    for name in ("case_predictions.csv", "metrics.csv"):
        if sha(source / name) != manifest["outputs"][name]:
            raise ValueError("Changed source report: " + name)
    check_tables(pd.read_csv(source / "case_predictions.csv"), pd.read_csv(source / "metrics.csv"))
    destination = INPUTS / GROUP
    names = ("case_predictions.csv", "metrics.csv", "verification.json", "manifest.json")
    if (destination / "manifest.json").exists():
        old = verify_archive()
        if any(sha(source / name) != old["source_files"][name] for name in names):
            raise ValueError("Do not silently replace a frozen comparison")
        return destination
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Inspect the partial comparison archive before resuming")
    destination.mkdir(parents=True)
    for name in names:
        shutil.copy2(
            source / name, destination / ("run_manifest.json" if name == "manifest.json" else name)
        )
    archive = dict(
        description="Matched input representations for Eclipse, Subanovic and Yu; accuracy only",
        original_input_manifest_sha256=sha(INPUTS / "manifest.json"),
        source_folder=str(source),
        source_files={name: sha(source / name) for name in names},
        files={
            path.name: dict(sha256=sha(path), bytes=path.stat().st_size)
            for path in destination.iterdir()
            if path.is_file()
        },
        new_model_fits_during_plotting=0,
        uncertainty_recalibrated=False,
    )
    (destination / "manifest.json").write_text(json.dumps(archive, indent=2) + "\n")
    verify_archive()
    return destination


def verify_archive():
    folder = INPUTS / GROUP
    manifest = json.loads((folder / "manifest.json").read_text())
    expected = manifest.get("release_input_manifest_sha256", manifest["original_input_manifest_sha256"])
    if expected != sha(INPUTS / "manifest.json"):
        raise ValueError("Figure input manifest does not match the recorded checksum")
    for name, record in manifest["files"].items():
        if sha(folder / name) != record["sha256"]:
            raise ValueError("Frozen comparison table changed: " + name)
        USED_INPUTS.add(GROUP + "/" + name)
    USED_INPUTS.add(GROUP + "/manifest.json")
    return manifest


def comparison_panel(ax):
    verify_archive()
    calls = read(GROUP, "case_predictions.csv")
    metrics = read(GROUP, "metrics.csv")
    summary = check_tables(calls, metrics)
    ax._representation_comparison_values = []
    for j, (representation, label, color, marker) in enumerate(STYLE):
        for i, dataset in enumerate(DATASETS):
            row = summary.loc[(representation, dataset)]
            value = 100 * float(row.balanced_top1)
            x = i + (j - 1) * 0.20
            ax.scatter(x, value, s=20, marker=marker, color=color, zorder=4)
            ax._representation_comparison_values.append(
                dict(
                    representation=representation,
                    dataset=dataset,
                    balanced_accuracy=value,
                    treatments=int(row.treatments),
                )
            )
    handles = [
        Line2D([], [], marker=marker, ls="none", color=color, ms=3.5, label=label)
        for _, label, color, marker in STYLE
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        frameon=False,
        fontsize=5.2,
        handletextpad=0.4,
        labelspacing=0.25,
        borderaxespad=0,
        handlelength=1.0,
    )
    ax.set(
        xticks=range(3),
        xticklabels=["Eclipse\n(n=3)", "Subanovic et al.\n(n=4)", "Yu et al.\n(n=19)"],
        ylabel="Class-balanced accuracy (%)",
        xlim=(-0.5, 2.5),
        ylim=(-6, 111),
        yticks=[0, 50, 100],
    )
    ax.tick_params(axis="x", labelsize=5.5)
    frame(ax, "y")
    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-run", type=Path, required=True, help="Completed run's report directory"
    )
    print(archive_run(parser.parse_args().archive_run))
