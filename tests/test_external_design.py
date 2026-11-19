"""Synthetic, CPU-only checks of identity isolation and historical thresholds."""

import unittest

import numpy as np
import pandas as pd

from mapper.analysis.external_design import compound_name, make_plan, validate_plan
from mapper.external import pair_inputs as paired
from mapper.uncertainty import calibrate_score_cutoff as score_cutoff


class WorkflowTests(unittest.TestCase):
    def example(self):
        labels = {f"C{i}": "A" if i < 3 else "B" for i in range(6)}
        folds = []
        for j in range(3):
            test = {f"C{j}", f"C{j+3}"}
            folds.append(dict(train=sorted(set(labels) - test), test=sorted(test)))
        cases = pd.DataFrame(
            [
                dict(case_key="one", training_compound="C0", context="omit_C0"),
                dict(case_key="two", training_compound="", context="full51"),
            ]
        )
        return labels, folds, cases

    def test_identity_removed_from_all_calibration_stages(self):
        labels, folds, cases = self.example()
        plan = make_plan(labels, cases, folds)
        context = next(c for c in plan["contexts"] if c["context"] == "omit_C0")
        self.assertNotIn("C0", context["develop"])
        for f in context["calibration"]:
            self.assertNotIn("C0", f["train"] + f["test"])
        self.assertEqual(sum(len(f["test"]) for f in context["calibration"]), 5)

    def test_external_outcomes_do_not_change_plan(self):
        labels, folds, cases = self.example()
        cases["expected"] = ["A", "B"]
        first = make_plan(labels, cases, folds)
        cases["expected"] = ["arbitrary", "wrong"]
        self.assertEqual(first, make_plan(labels, cases, folds))

    def test_incorrect_membership_is_rejected(self):
        labels, folds, cases = self.example()
        plan = make_plan(labels, cases, folds)
        context = next(c for c in plan["contexts"] if c["context"] == "omit_C0")
        context["calibration"][0]["test"].append("C0")
        with self.assertRaises(ValueError):
            validate_plan(plan, cases)

    def test_repeated_case_is_rejected(self):
        labels, folds, cases = self.example()
        with self.assertRaises(ValueError):
            make_plan(labels, pd.concat([cases, cases]), folds)

    def test_low_score_cutoff_matches_original_tie_rule(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        cutoff = score_cutoff(scores, np.ones(10, bool))
        self.assertLess(cutoff, 0.1)
        self.assertFalse(bool(0.1 <= cutoff))

    def test_errors_cannot_set_the_cutoff(self):
        score = score_cutoff([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9], [True] * 6 + [False])
        self.assertEqual(
            score, score_cutoff([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.01], [True] * 6 + [False])
        )

    def test_insufficient_correct_calibration_is_rejected(self):
        with self.assertRaises(ValueError):
            score_cutoff([0.1, 0.2, 0.3, 0.4], [True] * 4)

    def test_pair_order_is_text_then_fingerprint(self):
        result = paired(np.array([[1, 2], [3, 4]]), np.array([[5], [6]]))
        np.testing.assert_array_equal(result, [[5, 1, 2], [6, 1, 2], [5, 3, 4], [6, 3, 4]])

    def test_aliases_are_specific_antibiotic_identities(self):
        self.assertEqual(compound_name("yu", "YuPolymyxin"), "Polymyxin B")
        self.assertEqual(compound_name("yu", "YuPenicillin"), "Penicillin G")
        self.assertEqual(compound_name("smx", "12h"), "Sulfamethoxazole")


if __name__ == "__main__":
    unittest.main()
