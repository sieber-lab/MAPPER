"""Protect calibration, feature order and original final-epoch behaviour."""

import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from mapper.features import assemble_blocks, public_name, stored_name
from mapper.uncertainty import calibrate_score_cutoff, flag_uncertain


class PublicAPITests(unittest.TestCase):
    def test_example_csv_recreates_the_four_paper_profiles_exactly(self):
        root = Path(__file__).resolve().parents[1]
        example = root / "docs/examples/corramycin"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sample.npz"
            subprocess.run(
                [sys.executable, str(root / "scripts/prepare_external.py"),
                 "--treated", str(example / "treated.csv"), "--controls", str(example / "controls.csv"),
                 "--compound-id", "JH70", "--output", str(output)],
                check=True, capture_output=True, text=True,
            )
            with np.load(output) as prepared, np.load(root / "data/study_inputs/external/matrices.npz") as study:
                selected = study["external_case_keys"] == "inhouse_corramycin::inhouse_corramycin"
                self.assertEqual(selected.sum(), 4)
                np.testing.assert_array_equal(prepared["proteins"], study["genes"])
                np.testing.assert_array_equal(prepared["features"], study["external_proteomics"][selected])

    def test_infoalign_rename_does_not_sort_feature_blocks(self):
        blocks = {
            "InfoAlign": np.array([[10.0, 11.0]]),
            "ECFP": np.array([[20.0]]),
            "Proteomics": np.array([[30.0, 31.0]]),
        }
        np.testing.assert_array_equal(
            assemble_blocks(blocks, ["CellPaint", "ECFP", "Proteomics"]),
            [[10.0, 11.0, 20.0, 30.0, 31.0]],
        )
        self.assertEqual(stored_name("InfoAlign"), "CellPaint")
        self.assertEqual(public_name("CellPaint"), "InfoAlign")

    def test_ambiguous_names_rejected(self):
        with self.assertRaises(ValueError):
            assemble_blocks({"InfoAlign": [[1]], "CellPaint": [[2]]}, ["InfoAlign"])

    def test_cutoff_is_original_negative_risk_quantile(self):
        scores = np.array([0.2, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.1])
        correct = np.array([True] * 10 + [False])
        expected = -np.nextafter(np.quantile(-scores[correct], 0.9, method="higher"), np.inf)
        self.assertEqual(calibrate_score_cutoff(scores, correct), expected)
        np.testing.assert_array_equal(flag_uncertain(scores, expected), scores <= expected)

    def test_small_calibration_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            calibrate_score_cutoff([0.1, 0.2, 0.3], [True, True, True])

    def test_tabm_keeps_published_stopping_implementation(self):
        path = Path(__file__).resolve().parents[1] / "src/mapper/models/tabm.py"
        tree = ast.parse(path.read_text())
        wrapper = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TabMWrapper"
        )
        fit = next(
            node
            for node in wrapper.body
            if isinstance(node, ast.FunctionDef) and node.name == "fit"
        )
        saved = [
            node
            for node in ast.walk(fit)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "best_state"
                for target in node.targets
            )
        ]
        self.assertTrue(any(ast.unparse(node.value) == "self.model.state_dict()" for node in saved))


if __name__ == "__main__":
    unittest.main()
