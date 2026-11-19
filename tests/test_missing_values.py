import unittest
import numpy as np
from mapper.preprocessing.missing_values import *
from mapper.preprocessing.missing_values import expand_genes


class ImputationTests(unittest.TestCase):
    def setUp(self):
        self.train = np.array([[10.0, np.nan, np.nan], [12.0, 15.0, np.nan], [14.0, 17.0, np.nan]])
        self.test = np.array([[np.nan, 18.0, 21.0], [9.0, np.nan, np.nan]])
        self.parameters = fit_parameters(self.train)

    def test_training_parameters_match_hand_calculation(self):
        observed = np.array([10.0, 12.0, 15.0, 14.0, 17.0])
        self.assertEqual(self.parameters["n_training_profiles"], 3)
        self.assertEqual(self.parameters["n_observed_values"], 5)
        self.assertEqual(self.parameters["low_percentile"], np.quantile(observed, 0.01))
        self.assertEqual(self.parameters["gaussian_mean"], observed.mean() - 1.8 * observed.std())
        np.testing.assert_array_equal(self.parameters["supported_groups"], [True, True, False])

    def test_observed_values_untouched_and_unsupported_missing_zero(self):
        raw = np.vstack([self.train, self.test])
        for arm in ARMS:
            result = transform(raw, self.parameters, arm, SEED)
            np.testing.assert_array_equal(result[~np.isnan(raw)], raw[~np.isnan(raw)])
            np.testing.assert_array_equal(result[[0, 1, 2, 4], 2], 0)
            self.assertEqual(result[3, 2], 21)

    def test_held_out_extremes_cannot_change_parameters(self):
        before = dict(self.parameters)
        test = self.test.copy()
        test[~np.isnan(test)] *= 10000
        transform(test, self.parameters, "shifted_gaussian", 19)
        for key in before:
            np.testing.assert_array_equal(before[key], self.parameters[key])

    def test_draw_seed_and_gene_expansion(self):
        first = transform(self.train, self.parameters, "shifted_gaussian", SEED)
        np.testing.assert_array_equal(
            first, transform(self.train, self.parameters, "shifted_gaussian", SEED)
        )
        self.assertNotEqual(
            first[0, 1], transform(self.train, self.parameters, "shifted_gaussian", SEED + 1)[0, 1]
        )
        expanded = expand_genes(first, [1, 1, -1, 0])
        np.testing.assert_array_equal(expanded[:, 0], expanded[:, 1])
        np.testing.assert_array_equal(expanded[:, 2], 0)
        np.testing.assert_array_equal(expanded[:, 3], first[:, 0])

    def test_constant_and_zero_arms(self):
        for arm, expected in [("zero", 0), ("low_percentile", self.parameters["low_percentile"])]:
            result = transform(self.train, self.parameters, arm, 1)
            self.assertEqual(result[0, 1], expected)
