"""Recorded jobs reconstructed by the public entry point keep original inputs."""

import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "study_preparation", ROOT / "scripts/prepare_study_job.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class StudyPreparationTests(unittest.TestCase):
    def check_original(self, kind, relative, **kwargs):
        actual, plan, fields, _ = study.prepare(kind, **kwargs)
        with np.load(
            ROOT / "data/study_inputs" / relative / "matrices.npz", allow_pickle=False
        ) as expected:
            for key in ["proteomics", "multimodal", "sample_ids", "labels", "text", "text_classes"]:
                np.testing.assert_array_equal(actual[key], expected[key])
        job = plan["jobs"]["study"]
        self.assertFalse(set(job["train"]) & set(job["predict"]))

    def test_whole_operator_keeps_original_lexical_order(self):
        self.check_original("operators", "operator_whole/LE_to_DS", design="whole", job="LE_to_DS")

    def test_class_reference_selection_is_preserved(self):
        self.check_original(
            "operators",
            "operator_class_mixed/class_01_context_01",
            design="class_mixed",
            job="class_01_context_01",
        )

    def test_measurement_batch_purges_other_profiles_of_test_compounds(self):
        self.check_original("batches", "measurement_batch/C1", job="C1")

    def test_fold_fitted_imputation_and_control_alignment_match(self):
        for name in ["shifted_gaussian", "control_relative", "log2fc_873"]:
            self.check_original("preprocessing", f"preprocessing/{name}", preprocessing=name)


if __name__ == "__main__":
    unittest.main()
