import unittest
import numpy as np
from mapper.preprocessing.control_alignment import *
from mapper.preprocessing.missing_values import expand_genes


class AlignmentUnitTests(unittest.TestCase):
    def test_available_mean_and_counts(self):
        means, counts = available_mean([[1, np.nan, np.nan], [3, 4, np.nan]])
        np.testing.assert_array_equal(counts, [2, 1, 0])
        np.testing.assert_allclose(means, [2, 4, np.nan], equal_nan=True)

    def test_reference_groups_not_weighted_by_compound_count(self):
        controls = np.array([[10.0, 12.0], [14.0, 16.0], [100.0, 200.0]])
        np.testing.assert_array_equal(fit_reference(controls, [0, 0, 0, 1]), [12, 14])

    def test_hand_calculation_and_unsupported(self):
        raw = np.array([[12.0, 15.0, 8.0, np.nan], [14.0, 13.0, np.nan, 9.0]])
        means = np.array([[10.0, 12.0, np.nan, 8.0], [14.0, 16.0, 5.0, np.nan]])
        ref = np.array([12.0, 14.0, np.nan, 10.0])
        actual, shifts, support = transform(raw, means, ref)
        np.testing.assert_array_equal(actual, [[14.0, 17.0, 8.0, 0.0], [12.0, 11.0, 0.0, 9.0]])
        self.assertEqual(shifts[0, 3], 2)
        self.assertFalse(support[0, 2])
        self.assertFalse(support[1, 2])

    def test_test_only_control_cannot_change_training(self):
        means = np.array([[10.0, 12.0], [14.0, 16.0], [20.0, 22.0]])
        before = fit_reference(means, [0, 1])
        changed = means.copy()
        changed[2] += 999
        np.testing.assert_array_equal(before, fit_reference(changed, [0, 1]))
        raw = np.array([[12.0, 13.0], [14.0, 17.0], [23.0, 24.0]])
        y = transform(raw, means, before)[0]
        z = transform(raw, changed, before)[0]
        np.testing.assert_array_equal(y[:2], z[:2])
        self.assertFalse(np.array_equal(y[2], z[2]))

    def test_other_test_profile_cannot_change_output(self):
        raw = np.array([[10.0, 20.0], [30.0, 40.0]])
        controls = np.array([[8.0, 18.0], [29.0, 39.0]])
        ref = np.array([11.0, 21.0])
        before = transform(raw, controls, ref)[0]
        raw[1] += 1000
        after = transform(raw, controls, ref)[0]
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[:1], transform(raw[:1], controls[:1], ref)[0])

    def test_no_supported_controls_is_identity_then_zero_fill(self):
        raw = np.array([[3.0, np.nan, 8.0]])
        y = transform(raw, np.full_like(raw, np.nan), np.full(3, np.nan))[0]
        np.testing.assert_array_equal(y, [[3, 0, 8]])

    def test_mapping_preserves_absent_and_shared_genes(self):
        genes = expand_genes(np.array([[3.0, 0.0, 9.0]]), np.array([0, 0, -1, 1]))
        np.testing.assert_array_equal(genes, [[3, 3, 0, 0]])

    def test_input_not_mutated(self):
        x = np.array([[12.0, np.nan]])
        c = np.array([[10.0, 12.0]])
        ref = np.array([13.0, 14.0])
        copies = [a.copy() for a in (x, c, ref)]
        transform(x, c, ref)
        for a, b in zip((x, c, ref), copies):
            np.testing.assert_array_equal(a, b)

    def test_reject_shapes_infinities_and_invalid_group_indices(self):
        for indices in [[], [-1], [2], [0.5]]:
            with self.assertRaises(ValueError):
                fit_reference(np.ones((2, 3)), indices)
        with self.assertRaises(ValueError):
            transform(np.ones((2, 3)), np.ones((3, 2)), np.ones(3))
        with self.assertRaises(ValueError):
            transform(np.array([[np.inf]]), np.ones((1, 1)), np.ones(1))
