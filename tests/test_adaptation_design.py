"""CPU-only split tests, including shared identities across source domains."""

import copy
import unittest

import pandas as pd

from mapper.analysis import adaptation_design as d


class DesignTests(unittest.TestCase):
    def example(self):
        labels = {f"IH{i}": "A" if i < 3 else "B" for i in range(6)}
        labels.update(YU0="A", YU1="B")
        folds = [dict(test=[f"IH{i}", f"IH{i+3}"]) for i in range(3)]
        cases = pd.DataFrame(
            [
                dict(case_key="yu::shared", group_id="IH0", compound_name="Shared"),
                dict(case_key="yu::new", group_id="YU0", compound_name="New"),
            ]
        )
        return labels, cases, folds

    def test_both_domains_use_the_same_identity_group(self):
        labels, cases, folds = self.example()
        plan = d.make_plan(labels, cases, folds)
        shared = next(c for c in plan["contexts"] if c["omitted"] == "IH0")
        self.assertNotIn("IH0", shared["develop"])
        for fold in shared["calibration"]:
            self.assertNotIn("IH0", fold["train"] + fold["test"])

    def test_original_folds_are_preserved(self):
        labels, _, folds = self.example()
        assigned = d.assign_folds(labels, folds)
        for index, fold in enumerate(folds):
            for group in fold["test"]:
                self.assertEqual(assigned[group], index)

    def test_outer_label_does_not_choose_fits_or_cutoffs(self):
        labels, cases, folds = self.example()
        first = d.make_plan(labels, cases, folds)
        cases["expected"] = ["arbitrary", "different"]
        self.assertEqual(first, d.make_plan(labels, cases, folds))

    def test_duplicate_primary_case_is_rejected(self):
        labels, cases, folds = self.example()
        with self.assertRaises(ValueError):
            d.make_plan(labels, pd.concat([cases, cases]), folds)

    def test_outer_leakage_is_detected(self):
        labels, cases, folds = self.example()
        plan = copy.deepcopy(d.make_plan(labels, cases, folds))
        plan["contexts"][0]["calibration"][0]["test"].append(plan["contexts"][0]["omitted"])
        with self.assertRaises(ValueError):
            d.validate_plan(plan, cases)

    def test_unknown_identity_is_rejected(self):
        labels, cases, folds = self.example()
        cases.loc[0, "group_id"] = "unknown"
        with self.assertRaises(ValueError):
            d.make_plan(labels, cases, folds)

    def test_plan_is_deterministic_and_deduplicates_jobs(self):
        labels, cases, folds = self.example()
        plan = d.make_plan(labels, cases, folds)
        self.assertEqual(plan, d.make_plan(labels, cases, folds))
        self.assertLessEqual(len(plan["jobs"]), 4 * len(cases))
        self.assertEqual(plan["total_fits"], 5 * len(plan["jobs"]))


if __name__ == "__main__":
    unittest.main()
