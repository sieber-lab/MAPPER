"""Regression checks for the legacy Subanovic coverage identifier."""

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd

from mapper.figures.external import coverage, MAIN_DATASETS
from mapper.figures.style import apply_style, read


class ExternalCoverageTests(unittest.TestCase):
    def test_legacy_identifier_preserves_all_subanovic_profiles(self):
        apply_style()
        source = read("external_inputs", "source_coverage.csv")
        before = source.copy(deep=True)
        expected = source.loc[source.dataset.eq("external"), "retained_genes"].to_numpy()
        self.assertEqual(len(expected), 12)
        fig, ax = plt.subplots()
        try:
            coverage(ax, source, "retained_genes", MAIN_DATASETS)
            clouds = [item for item in ax.collections if isinstance(item, PathCollection)]
            self.assertEqual(len(clouds), len(MAIN_DATASETS))
            actual = np.asarray(clouds[MAIN_DATASETS.index("subanovic")].get_offsets())
            np.testing.assert_array_equal(np.sort(actual[:, 0]), np.sort(expected))
            self.assertEqual((float(actual[:, 0].min()), float(actual[:, 0].max())), (76.0, 187.0))
            self.assertFalse(any(text.get_text() == "—" for text in ax.texts))
            pd.testing.assert_frame_equal(source, before)
        finally:
            plt.close(fig)

    def test_current_identifier_gives_identical_points(self):
        apply_style()
        source = read("external_inputs", "source_coverage.csv")
        current = source.assign(dataset=source.dataset.replace({"external": "subanovic"}))
        fig, axes = plt.subplots(1, 2)
        try:
            for ax, data in zip(axes, [source, current]):
                coverage(ax, data, "retained_genes", MAIN_DATASETS)
            clouds = [
                [item for item in ax.collections if isinstance(item, PathCollection)] for ax in axes
            ]
            self.assertEqual(len(clouds[0]), len(clouds[1]))
            for first, second in zip(*clouds):
                np.testing.assert_array_equal(first.get_offsets(), second.get_offsets())
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
