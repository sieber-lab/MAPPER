import unittest
import numpy as np
from mapper.preprocessing.median_alignment import *
from mapper.preprocessing.missing_values import expand_genes


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array(
            [[8.0, 10.0, 12.0, np.nan], [11.0, 13.0, 15.0, 17.0], [5.0, np.nan, 9.0, 11.0]]
        )

    def test_reference_training_medians(self):
        self.assertEqual(fit_reference(self.x[:2]), 12.0)

    def test_alignment_and_missingness(self):
        y, medians, shifts = transform(self.x, 12.0)
        np.testing.assert_array_equal(medians, [10.0, 14.0, 9.0])
        np.testing.assert_array_equal(shifts, [2.0, -2.0, 3.0])
        np.testing.assert_array_equal(y[np.isnan(self.x)], 0)
        masked = np.where(np.isnan(self.x), np.nan, y)
        np.testing.assert_allclose(sample_medians(masked), 12.0)

    def test_within_sample_differences_preserved(self):
        y, _, _ = transform(self.x, 12.0)
        for old, new in zip(self.x, y):
            ok = np.isfinite(old)
            np.testing.assert_allclose(np.diff(old[ok]), np.diff(new[ok]))

    def test_test_sample_and_other_test_samples_cannot_change_training(self):
        ref = fit_reference(self.x[:2])
        before = transform(self.x, ref)[0]
        perturbed = self.x.copy()
        perturbed[2] += 100
        after = transform(perturbed, ref)[0]
        np.testing.assert_array_equal(before[:2], after[:2])
        np.testing.assert_allclose(before[2], after[2])
        np.testing.assert_array_equal(before[2:], transform(self.x[2:], ref)[0])

    def test_mapping_keeps_absent_genes_zero_and_shared_groups_identical(self):
        y = transform(self.x, 12.0)[0]
        genes = expand_genes(y, np.array([0, 0, 1, -1]))
        np.testing.assert_array_equal(genes[:, 0], genes[:, 1])
        np.testing.assert_array_equal(genes[:, 3], 0)
        self.assertEqual(genes[2, 2], 0)

    def test_reject_bad_profiles_and_reference(self):
        for bad in [
            np.array([[np.nan, np.nan]]),
            np.array([[np.inf, 2.0]]),
            np.empty((0, 2)),
            np.ones(3),
        ]:
            with self.assertRaises(ValueError):
                fit_reference(bad)
        with self.assertRaises(ValueError):
            transform(self.x, np.nan)

    def test_input_not_mutated(self):
        old = self.x.copy()
        transform(self.x, fit_reference(self.x[:2]))
        np.testing.assert_array_equal(old, self.x)
