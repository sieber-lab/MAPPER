import unittest

import numpy as np

from mapper.analysis.control_distance import (
    distances,
    leave_group_out,
    observed_mean,
    representatives,
    score_control,
)


class ControlDistanceTests(unittest.TestCase):
    def test_rank_distance_ignores_global_gain_and_offset(self):
        x = np.arange(100, dtype=float)
        d = distances(x, 2 * x + 15)
        self.assertAlmostEqual(d["rank_distance"], 0)

    def test_centered_difference_ignores_log_offset(self):
        x = np.arange(100, dtype=float)
        self.assertAlmostEqual(distances(x, x + 13)["median_centered_absolute_difference"], 0)

    def test_missing_controls_not_imputed(self):
        x = np.array([[2.0, np.nan], [4.0, 6.0]])
        m, c = observed_mean(x, minimum=2)
        np.testing.assert_allclose(m[0], 3.0)
        self.assertTrue(np.isnan(m[1]))
        np.testing.assert_array_equal(c, [2, 1])

    def test_aliases_not_double_counted(self):
        np.testing.assert_array_equal(
            representatives([True] * 5, np.array([1, 1, 2, -1, 3])), [0, 2, 4]
        )

    def test_source_group_excluded(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]])
        np.testing.assert_array_equal(leave_group_out(x)[2], [2.0, 3.0])

    def test_matched_comparisons_have_no_self_reference(self):
        x = np.arange(100, dtype=float)
        source = np.stack([x, x + 1, x + 3])
        point, matched = score_control(x + 8, source, np.arange(100), np.ones(100, bool))
        self.assertAlmostEqual(point["rank_distance"], 0)
        self.assertEqual(len(matched), 3)
        self.assertTrue(all(abs(r["source_rank_distance"]) < 1e-12 for r in matched))

    def test_missing_values_cannot_enter_distance(self):
        x = np.arange(100, dtype=float)
        x[0] = np.nan
        with self.assertRaises(ValueError):
            distances(x, np.arange(100))


if __name__ == "__main__":
    unittest.main()
