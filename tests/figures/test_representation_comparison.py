"""CPU-only checks of the Figure 6 draft's data and panel correspondence."""

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mapper.figures.style import apply_style, read
from mapper.figures.representation_comparison import check_tables, comparison_panel, verify_archive
from mapper.figures.external import main_figure_with_representation_comparison


class RepresentationComparisonTests(unittest.TestCase):
    def test_metrics_match_compound_calls(self):
        verify_archive()
        calls = read("external_representation", "case_predictions.csv")
        summary = check_tables(calls, read("external_representation", "metrics.csv"))
        for representation, correct, balanced in [
            ("intensity", 0, 0.0),
            ("significant_intensity", 14, 100 * 83 / 252),
            ("fold_change", 15, 100 * 425 / 924),
        ]:
            row = summary.loc[(representation, "ALL")]
            self.assertEqual(row.correct_top1, correct)
            np.testing.assert_allclose(100 * row.balanced_top1, balanced, atol=1e-10, rtol=0)

    def test_comparison_has_nine_offset_points_and_no_error_bars(self):
        apply_style()
        fig, ax = plt.subplots()
        try:
            comparison_panel(ax)
            self.assertEqual(len(ax.collections), 9)
            self.assertEqual(len(ax.containers), 0)
            values = ax._representation_comparison_values
            self.assertTrue(
                all(
                    v["balanced_accuracy"] == 0
                    for v in values
                    if v["representation"] == "intensity"
                )
            )
            self.assertEqual(sorted({v["dataset"] for v in values}), ["eclipse", "subanovic", "yu"])
            self.assertEqual(len({float(point.get_offsets()[0, 0]) for point in ax.collections}), 9)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            self.assertLess(ax.get_legend().get_window_extent(renderer).y1, ax.bbox.y0)
        finally:
            plt.close(fig)

    def test_final_draft_panels_align_and_keep_other_results(self):
        apply_style()
        fig = main_figure_with_representation_comparison()
        try:
            axes = fig.axes[:9]
            for start in [0, 3, 6]:
                boxes = [ax.get_position() for ax in axes[start : start + 3]]
                np.testing.assert_allclose(
                    [b.width for b in boxes], boxes[0].width, atol=1e-12, rtol=0
                )
                np.testing.assert_allclose([b.y0 for b in boxes], boxes[0].y0, atol=1e-12, rtol=0)
            for column in range(3):
                np.testing.assert_allclose(
                    [axes[index].get_position().x0 for index in [column, column + 3, column + 6]],
                    axes[column].get_position().x0,
                    atol=1e-12,
                    rtol=0,
                )
            self.assertEqual(len(axes[5]._representation_comparison_values), 9)
            self.assertEqual(len(axes[6]._accuracy_count_labels), 9)
            self.assertEqual(len(axes[8].patches), 6)
            self.assertTrue(any(text.get_text() == "j" for text in fig.texts))
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend = axes[5].get_legend().get_window_extent(renderer)
            self.assertGreater(legend.y0, axes[8]._left_title.get_window_extent(renderer).y1 + 2)
            self.assertFalse(any(text.get_text() == "—" for text in axes[0].texts))
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
