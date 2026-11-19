"""Portability tolerances must never conceal changed decisions or rankings."""

import unittest
import numpy as np
from mapper.analysis.replay import check_meta_replay, check_ranking_replay


class ReplayTests(unittest.TestCase):
    def test_small_margin_optimization_drift_is_reported(self):
        result = check_meta_replay([6.609068], [0.8 + 4e-7], [True], [6.609052], [0.8], [True])
        self.assertTrue(result["flags_identical"])
        self.assertGreater(result["raw_max_absolute_difference"], 1e-5)

    def test_changed_flags_are_not_accepted(self):
        with self.assertRaises(AssertionError):
            check_meta_replay([1.0], [0.8], [True], [1.0], [0.8], [False])

    def test_large_probability_drift_is_not_accepted(self):
        with self.assertRaises(AssertionError):
            check_meta_replay([1.0], [0.81], [True], [1.0], [0.8], [True])

    def test_changed_ranking_is_not_accepted(self):
        self.assertTrue(check_ranking_replay([0.1, 0.2, 0.2], [0.1, 0.2, 0.2]))
        with self.assertRaises(AssertionError):
            check_ranking_replay([0.2, 0.1], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
