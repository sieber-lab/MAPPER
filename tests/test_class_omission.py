import itertools
import unittest

import numpy as np
from scipy.stats import spearmanr

from mapper.analysis.class_omission import (
    blocked_permutations,
    correlation,
    exact_class_association,
    holm,
    remove_group_means,
    tanimoto_matrix,
    within_class_association,
)


class AnalysisTests(unittest.TestCase):
    def test_tanimoto_avoids_uint8_overflow(self):
        x = np.ones((2, 2048), dtype=np.uint8)
        x[1, :1024] = 0
        np.testing.assert_array_equal(tanimoto_matrix(x), [[1.0, 0.5], [0.5, 1.0]])

    def test_exact_tied_class_permutation(self):
        x, y = np.array([3, 4, 4, 9]), np.array([0.0, 0.4, 0.5, 1.0])
        actual = exact_class_association(x, y)
        expected = spearmanr(x, y).statistic
        p = np.mean(
            [
                abs(spearmanr(x[list(order)], y).statistic) >= abs(expected) - 1e-12
                for order in itertools.permutations(range(4))
            ]
        )
        self.assertAlmostEqual(actual["spearman"], expected)
        self.assertAlmostEqual(actual["p_raw"], p)

    def test_permutations_do_not_mix_classes(self):
        groups = np.array(["a", "b", "a", "b", "b", "c"])
        permutations = blocked_permutations(groups, count=50, seed=19)
        np.testing.assert_array_equal(groups[permutations], np.tile(groups, (50, 1)))
        for row in permutations:
            self.assertEqual(set(row), set(range(6)))

    def test_between_class_flag_difference_is_not_within_association(self):
        groups = np.array(["a"] * 4 + ["b"] * 4)
        x = np.arange(8.0)
        flag = np.array([0] * 4 + [1] * 4)
        test = within_class_association(x, flag, groups, blocked_permutations(groups, count=20))
        self.assertFalse(test["estimable"])
        self.assertTrue(np.isnan(test["within_class_rank_r"]))

    def test_within_class_effect_is_explicit_residual_correlation(self):
        groups = np.array(["a"] * 4 + ["b"] * 4)
        x = np.arange(8.0)
        flag = np.array([0, 0, 1, 1] * 2)
        test = within_class_association(x, flag, groups, blocked_permutations(groups, count=100))
        expected = correlation(remove_group_means(x, groups), remove_group_means(flag, groups))
        self.assertAlmostEqual(test["within_class_rank_r"], expected)

    def test_holm_keeps_original_order_and_nan(self):
        np.testing.assert_allclose(holm([0.04, 0.001, np.nan, 0.03]), [0.06, 0.003, np.nan, 0.06])


if __name__ == "__main__":
    unittest.main()
