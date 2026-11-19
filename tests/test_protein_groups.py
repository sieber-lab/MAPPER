import unittest
import numpy as np
from mapper.preprocessing.protein_groups import *
from mapper.preprocessing.missing_values import expand_genes


class MappingTests(unittest.TestCase):
    def test_only_multiple_gene_annotations_excluded(self):
        mapping = np.array([-1, -1, 0, 0, 2, 3, 4, 4, 4])
        mask, groups = ambiguous_columns(mapping)
        np.testing.assert_array_equal(groups, [0, 4])
        np.testing.assert_array_equal(
            mask, [False, False, True, True, False, False, True, True, True]
        )

    def test_values_dimensions_and_inputs_preserved(self):
        original = np.arange(18.0).reshape(2, 9)
        mapping = np.array([-1, -1, 0, 0, 2, 3, 4, 4, 4])
        saved = original.copy()
        out = exclude_multigene_groups(original, mapping)
        self.assertEqual(out.shape, original.shape)
        mask, _ = ambiguous_columns(mapping)
        np.testing.assert_array_equal(out[:, mask], 0)
        np.testing.assert_array_equal(out[:, ~mask], original[:, ~mask])
        np.testing.assert_array_equal(original, saved)

    def test_no_ambiguity_is_identity(self):
        x = np.arange(12.0).reshape(3, 4)
        np.testing.assert_array_equal(exclude_multigene_groups(x, [-1, 1, 3, 5]), x)

    def test_mask_independent_of_values_and_rows(self):
        x = np.arange(12.0).reshape(3, 4)
        mapping = [0, 0, 1, -1]
        full = exclude_multigene_groups(x, mapping)
        np.testing.assert_array_equal(full[1:], exclude_multigene_groups(x[1:], mapping))

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            ambiguous_columns([-2, 0])
        with self.assertRaises(ValueError):
            exclude_multigene_groups(np.ones((2, 3)), [0, 1])
        with self.assertRaises(ValueError):
            exclude_multigene_groups(np.array([[np.nan, 0]]), [0, 1])
