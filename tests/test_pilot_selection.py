"""Bounded pilots select recorded folds; defaults still execute every fold."""

import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from mapper import evaluation


class ToyModel:
    def fit(self, values, target):
        return self

    def predict_proba(self, values):
        return np.tile([0.4, 0.6], (len(values), 1))

    def predict(self, values):
        return np.ones(len(values), dtype=int)


class PilotSelectionTests(unittest.TestCase):
    def inputs(self):
        compounds = pd.Series(np.repeat(["C0", "C1", "C2", "C3"], 2))
        return dict(
            model_type="toy", X=np.arange(16).reshape(8, 2), y=np.tile([0, 1], 4), groups=compounds
        )

    def evaluate(self, selection):
        folds = [
            [
                evaluation.FoldSpec({"C0", "C2"}, {"C1", "C3"}),
                evaluation.FoldSpec({"C1", "C3"}, {"C0", "C2"}),
            ]
        ]
        kwargs = self.inputs()
        with (
            patch.object(evaluation, "build_or_load_compound_folds", return_value=folds),
            patch.object(
                evaluation, "get_model", side_effect=lambda *a, **k: ToyModel()
            ) as factory,
        ):
            result = evaluation.evaluate_model(
                **kwargs,
                true_moa_by_sample=pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"]),
                compound_by_sample=kwargs["groups"],
                fold_selection=selection,
                seed_splits=1,
            )
        return result, factory.call_count

    def test_default_runs_all_fixed_folds(self):
        result, count = self.evaluate(None)
        self.assertEqual(count, 2)
        self.assertEqual(len(result["per_fold_outputs"]), 2)

    def test_pilot_runs_only_the_requested_original_fold(self):
        result, count = self.evaluate([(0, 1)])
        self.assertEqual(count, 1)
        self.assertEqual(set(result["per_fold_outputs"][0].CompoundID), {"C0", "C2"})
        self.assertEqual(result["per_fold_outputs"][0].fold_idx.unique().tolist(), [1])

    def test_invalid_pilot_is_rejected_before_fitting(self):
        with self.assertRaises(ValueError):
            self.evaluate([(9, 9)])
        with self.assertRaises(ValueError):
            self.evaluate([])

    def test_leave_one_compound_out_uses_one_requested_identity(self):
        with patch.object(evaluation, "get_model", return_value=ToyModel()) as factory:
            result = evaluation.evaluate_model_LOO(
                **self.inputs(), seed_repeats=1, test_compounds=["C0"]
            )
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(set(result["per_fold_outputs"][0].CompoundID), {"C0"})


if __name__ == "__main__":
    unittest.main()
