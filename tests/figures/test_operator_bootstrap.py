"""Check compound clustering, original intervals and portable reproduction."""

import json
import unittest

import numpy as np
import pandas as pd

from mapper.figures.operator_bootstrap import (
    bootstrap_accuracy,
    compound_weights,
    calculate,
    verify_archive,
)
from mapper.figures.style import INPUTS, read


class OperatorBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify_archive()
        cls.settings = json.loads((INPUTS / "operator_bootstrap/settings.json").read_text())
        cls.predictions = read("operator_bootstrap", "compound_predictions.csv")
        cls.baseline = read("operator_combined", "transfer_plot_values.csv")

    def test_original_one_split_draws_reproduce_independent_row_resampling(self):
        primary = self.predictions.query("validation == 'operator_transfer'")
        for _, group in primary.groupby(["template", "configuration"]):
            group = group.sort_values("compound_id")
            labels = group.true_moa.to_numpy()
            correct = group.top1_correct.to_numpy()
            rng = np.random.default_rng(self.settings["seed"])
            indices = np.concatenate(
                [
                    rng.choice(
                        np.flatnonzero(labels == name),
                        size=(2000, int(np.sum(labels == name))),
                        replace=True,
                    )
                    for name in self.settings["classes"]
                ],
                axis=1,
            )
            expected = correct[indices].mean(axis=1)
            _, observed, _ = bootstrap_accuracy(group, self.settings["classes"])
            np.testing.assert_allclose(observed, expected, atol=1e-15, rtol=0)

    def test_duplicating_a_split_does_not_narrow_intervals(self):
        group = self.predictions.query(
            "validation == 'operator_transfer' and template == 'LE_to_DS' and configuration == 'proteomics_text'"
        ).copy()
        twin = group.copy()
        twin["split"] = "duplicated_split"
        point, samples, _ = bootstrap_accuracy(group, self.settings["classes"])
        duplicated, repeated, _ = bootstrap_accuracy(
            pd.concat([group, twin]), self.settings["classes"]
        )
        self.assertEqual(point, duplicated)
        np.testing.assert_array_equal(samples, repeated)

    def test_overlapping_splits_share_compound_weights(self):
        group = self.predictions.query(
            "validation == 'count_matched' and template == 'DS_to_LE' and configuration == 'proteomics_text'"
        )
        ids = sorted(group.compound_id.unique())
        labels = group.groupby("compound_id").true_moa.first().reindex(ids).to_numpy()
        weights = compound_weights(labels, self.settings["classes"], repeats=40)
        _, observed, _ = bootstrap_accuracy(group, self.settings["classes"], repeats=40)
        expected = []
        for draw in weights:
            mapping = dict(zip(ids, draw))
            split_values = []
            for _, split in group.groupby("split"):
                w = split.compound_id.map(mapping).to_numpy()
                split_values.append(np.average(split.top1_correct.astype(float), weights=w))
            expected.append(np.mean(split_values))
        np.testing.assert_allclose(observed, expected, atol=1e-15, rtol=0)
        for name in self.settings["classes"]:
            mask = labels == name
            np.testing.assert_array_equal(weights[:, mask].sum(axis=1), np.repeat(mask.sum(), 40))

    def test_all_means_and_archived_intervals_reproduce(self):
        result, draws = calculate(self.predictions, self.baseline, self.settings)
        expected = read("operator_bootstrap", "plot_values.csv")
        pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=1e-12, rtol=0)
        self.assertEqual(len(draws), 24000)
        self.assertEqual(int(result.empty_draws.sum()), 0)
        np.testing.assert_array_equal(result["mean"], self.baseline["mean"])
        self.assertTrue((result.low <= result["mean"]).all())
        self.assertTrue((result["mean"] <= result.high).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
