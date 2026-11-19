"""Conditional compound-bootstrap intervals for fixed operator comparisons.

The statistic is mean top-1 accuracy across the saved test splits. Resampling
is stratified by MoA over unique test compounds. A compound receives one
multiplicity per draw, reused in every split where it was tested. We then
recalculate each split's accuracy and average across splits. Thus overlapping
splits, biological profiles and model seeds are not independent bootstrap units.
Models, splits and predictions remain fixed. No confidence interval for the
selection of training compounds or for model refitting is estimated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .style import INPUTS, USED_INPUTS, read, sha


def compound_weights(labels, classes, repeats=2000, seed=20260917):
    """Empirical class-stratified draws, with one weight per compound ID."""
    labels = np.asarray(labels)
    assert set(labels) == set(classes)
    rng = np.random.default_rng(seed)
    weights = np.zeros((repeats, len(labels)), dtype=np.int16)
    for name in classes:
        members = np.flatnonzero(labels == name)
        selected = rng.choice(members, size=(repeats, len(members)), replace=True)
        np.add.at(weights, (np.arange(repeats)[:, None], selected), 1)
    assert np.all(weights.sum(axis=1) == len(labels))
    return weights


def bootstrap_accuracy(table, classes, repeats=2000, seed=20260917):
    """Return the unchanged split-mean accuracy and clustered bootstrap draws.

    Stratification applies to the union of the fixed test sets. Consequently
    the individual splits' bootstrap sizes can vary. Every saved appearance
    of a compound carries the same draw weight; no test case is refitted.
    """
    assert not table.duplicated(["split", "compound_id"]).any()
    assert table.groupby("compound_id").true_moa.nunique().eq(1).all()
    assert table.top1_correct.dtype == bool
    assert np.array_equal(table.top1_correct, table.top1_moa.eq(table.true_moa))
    ids = sorted(table.compound_id.unique())
    splits = sorted(table.split.unique())
    labels = table.groupby("compound_id").true_moa.first().reindex(ids).to_numpy()
    grid = table.pivot(index="split", columns="compound_id", values="top1_correct").reindex(
        index=splits, columns=ids
    )
    observed = grid.notna().to_numpy(dtype=float)
    correct = grid.eq(True).to_numpy(dtype=float)
    point = float((correct.sum(axis=1) / observed.sum(axis=1)).mean())
    weights = compound_weights(labels, classes, repeats, seed)
    denominators = weights @ observed.T
    # This is checked for the archived data; do not silently discard draws.
    assert np.all(denominators > 0), "A resample has an empty test split."
    samples = ((weights @ correct.T) / denominators).mean(axis=1)
    assert np.all((0 <= samples) & (samples <= 1))
    return (
        point,
        samples,
        dict(
            n_splits=len(splits),
            n_unique_compounds=len(ids),
            n_compound_split_records=len(table),
            empty_draws=0,
        ),
    )


def calculate(predictions, baseline, settings):
    """Generate 12 plotted rows, preserving all means and original holdout CIs."""
    rows, draws = [], []
    paired = {}
    for row in baseline.itertuples(index=False):
        group = predictions.loc[
            predictions.template.eq(row.direction)
            & predictions.configuration.eq(row.configuration)
            & predictions.validation.eq(row.validation)
        ].copy()
        assert group.split.nunique() == (1 if row.validation == "operator_transfer" else 5)
        counts = group.groupby(["split", "true_moa"]).size().unstack(fill_value=0)
        assert counts.eq(counts.iloc[0]).all().all()
        layout = (
            group[["split", "compound_id", "true_moa"]]
            .sort_values(["split", "compound_id"])
            .reset_index(drop=True)
        )
        key = (row.direction, row.validation)
        if key in paired:
            pd.testing.assert_frame_equal(layout, paired[key])
        else:
            paired[key] = layout
        point, samples, info = bootstrap_accuracy(
            group, settings["classes"], settings["repeats"], settings["seed"]
        )
        np.testing.assert_allclose(100 * point, row.mean, atol=1e-10, rtol=0)
        low, high = 100 * np.quantile(samples, [0.025, 0.975])
        if row.validation == "operator_transfer":
            np.testing.assert_allclose([low, high], [row.low, row.high], atol=1e-10, rtol=0)
            low, high = row.low, row.high  # Retain the original exported precision.
        rows.append(
            dict(
                direction=row.direction,
                configuration=row.configuration,
                validation=row.validation,
                mean=row.mean,
                low=low,
                high=high,
                interval_type="conditional_compound_bootstrap_95_CI",
                **info,
            )
        )
        draws.extend(
            dict(
                direction=row.direction,
                configuration=row.configuration,
                validation=row.validation,
                draw=i,
                accuracy=float(value),
            )
            for i, value in enumerate(samples)
        )
    return pd.DataFrame(rows), pd.DataFrame(draws)


def verify_archive():
    folder = INPUTS / "operator_bootstrap"
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        assert sha(folder / name) == expected["sha256"], name
        USED_INPUTS.add("operator_bootstrap/" + name)
    USED_INPUTS.add("operator_bootstrap/manifest.json")
    return manifest


def plot_values():
    verify_archive()
    return read("operator_bootstrap", "plot_values.csv")


def reproduce(output):
    """Recalculate from the portable snapshots, without historical run folders."""
    verify_archive()
    settings = json.loads((INPUTS / "operator_bootstrap/settings.json").read_text())
    predictions = read("operator_bootstrap", "compound_predictions.csv")
    baseline = read("operator_combined", "transfer_plot_values.csv")
    result, samples = calculate(predictions, baseline, settings)
    expected = read("operator_bootstrap", "plot_values.csv")
    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=1e-12, rtol=0)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "plot_values.csv", index=False)
    samples.to_csv(output / "bootstrap_draws.csv", index=False)
    print("Reproduced all 12 means/intervals; original holdout intervals match exactly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    reproduce(parser.parse_args().output)
