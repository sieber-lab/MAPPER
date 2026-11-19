"""Numerical and design invariants for the portable analysis entry points."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from mapper.analysis.bootstrap import binary_ranking, bootstrap_weights
from mapper.analysis.configuration_selection import compound_metrics, ranked_counts, weighted_ap
from mapper.analysis.nested_meta import fit_fixed, nested
from mapper.analysis.paired_comparisons import corrected_test, holm_adjust
from mapper.analysis.treatment_distance import time_mismatch
from mapper.primary_jobs import prepare_job
from mapper.screens import configurations


class AnalysisTests(unittest.TestCase):
    def test_reference_text_rows_follow_original_backbone_order(self):
        root = Path(__file__).resolve().parents[1]
        text = pd.read_pickle(root / "data/embeddings/250808_semanticAugmentation_embeddings.pkl")
        text = text.loc[text.Model.eq("BioBERT")].sort_values("MoA", kind="stable")
        with np.load(root / "data/inhouse/paper_training.npz", allow_pickle=False) as source:
            np.testing.assert_array_equal(source["text_classes"], text.MoA.to_numpy(str))
            np.testing.assert_array_equal(source["text"], np.stack(text.Embedding))

    def test_matching_ties_use_average_precision(self):
        truth = np.array([1, 0, 1, 0, 0, 1])
        scores = np.array([0.7, 0.7, 0.3, 0.3, 0.2, 0.1])
        auc, ap = binary_ranking(truth, scores, np.ones((1, 6)))
        self.assertAlmostEqual(auc[0], roc_auc_score(truth, scores))
        self.assertAlmostEqual(ap[0], average_precision_score(truth, scores))
        counts = ranked_counts(scores, truth, np.arange(6), 6)
        self.assertAlmostEqual(weighted_ap(counts, np.ones((1, 6)))[0], ap[0])

    def test_bootstrap_unit_is_compound_with_fixed_class_counts(self):
        labels = np.array(["a"] * 3 + ["b"] * 4)
        weights = bootstrap_weights(labels, 100, 42)
        np.testing.assert_array_equal(weights[:, :3].sum(axis=1), 3)
        np.testing.assert_array_equal(weights[:, 3:].sum(axis=1), 4)

    def test_metrics_are_not_taken_after_averaging_cv_repetitions(self):
        scores = np.array([[[0.9, 0.1], [0.6, 0.4]], [[0.1, 0.9], [0.1, 0.9]]])
        result = compound_metrics(scores, np.array([0, 1]), np.ones((1, 2)))
        self.assertEqual(result["top1_accuracy"][0], 0.5)

    def test_recorded_screens_and_feature_order(self):
        feature, encoder = configurations("features"), configurations("encoders")
        self.assertEqual(len(feature), 192)
        self.assertEqual(len(encoder), 9)
        full = next(
            x
            for x in feature
            if x["id"] == "semantic_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+InfoAlign+MIC+GC"
        )
        self.assertEqual(
            full["stored_feature_order"],
            ["Text_Embedding", "Proteomics", "ECFP", "CellPaint", "MIC", "GC"],
        )
        self.assertEqual(full["seed_splits"], 1)
        self.assertEqual(full["cv_iterations"], 5)

    def test_original_meta_architecture_and_compound_isolation(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.normal(size=(48, 15)), columns=nested.META_FEATURES)
        frame["CompoundID"] = [f"C{i}" for i in range(24)] * 2
        frame["role"] = ["id"] * 24 + ["ood"] * 24
        frame["correct"] = [1] * 24 + [0] * 24
        frame["error"] = 1 - frame.correct
        pairs = [
            (
                frame.loc[~frame.CompoundID.isin([f"C{j}" for j in range(24) if j % 3 == i])],
                frame.loc[frame.CompoundID.isin([f"C{j}" for j in range(24) if j % 3 == i])],
            )
            for i in range(3)
        ]
        fitted, inner = fit_fixed(frame, pairs)
        model = fitted["model"]
        self.assertEqual(len(inner), 3)
        self.assertEqual(model.steps[1][1].degree, 2)
        self.assertEqual(model.steps[-1][1].C, 1.0)
        self.assertEqual(model.steps[-1][1].class_weight, "balanced")
        with self.assertRaises(ValueError):
            fit_fixed(frame, [(frame, pairs[0][1]), *pairs[1:]])

    def test_primary_job_checks_compound_leakage_and_block_order(self):
        arrays = dict(
            compounds=np.array(["a", "b", "c"]),
            profiles=np.array(["a1", "b1", "c1"]),
            labels=np.array(["A", "B", "A"]),
            text=np.array([[1.0], [2.0]]),
            text_classes=np.array(["A", "B"]),
            protein=np.array([[3.0], [4.0], [5.0]]),
        )
        fields = dict(
            compound_ids="compounds",
            profile_ids="profiles",
            labels="labels",
            text="text",
            text_classes="text_classes",
            feature_blocks={"proteomics": ["protein"]},
        )
        plan = dict(classes=["A", "B"], jobs={"job": dict(train=["a", "b"], predict=["c"])})
        prepared = prepare_job(arrays, plan, "job", fields, "proteomics")
        np.testing.assert_array_equal(
            prepared["train"], [[1.0, 3.0], [2.0, 3.0], [1.0, 4.0], [2.0, 4.0]]
        )
        np.testing.assert_array_equal(prepared["target"], [1, 0, 0, 1])
        plan["jobs"]["job"]["train"].append("c")
        with self.assertRaises(ValueError):
            prepare_job(arrays, plan, "job", fields, "proteomics")

    def test_corrected_test_and_holm_keep_defined_family(self):
        result = corrected_test(np.array([0.1, 0.2, -0.05, 0.04]), np.full(4, 0.25))
        self.assertEqual(result["correction_factor"], 0.5)
        np.testing.assert_allclose(holm_adjust(np.array([0.04, 0.001, 0.03])), [0.06, 0.003, 0.06])

    def test_treatment_time_adjustment(self):
        self.assertEqual(time_mismatch(60, 60, 8), 0)
        self.assertAlmostEqual(time_mismatch(30, 60, 8), 1 / 3)
        self.assertEqual(time_mismatch(1440, 60, 8), 1)


if __name__ == "__main__":
    unittest.main()
