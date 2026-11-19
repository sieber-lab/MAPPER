"""Compare portable reference predictions with the final Figure 5 outputs."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from mapper.uncertainty import calibrate_score_cutoff


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "results/verification")
    args = parser.parse_args()
    config = json.loads((ROOT / "data/configurations/reference.json").read_text())
    summary = []
    for arm, label in [("multimodal", "Full multimodal"), ("proteomics", "Proteomics-only")]:
        prediction = args.predictions / ("reference_" + arm)
        current = pd.read_csv(prediction / "pairwise_predictions.csv.gz")
        keys = ["SampleID", "CompoundID", "CandidateMoA", "Seed"]
        expected = []
        for seed in config["seeds"]:
            path = ROOT / f"data/checks/reference_predictions/{arm}_seed_{seed}.csv"
            part = pd.read_csv(path, float_precision="round_trip").rename(
                columns={"seed": "Seed", "ProbaPos": "Score"}
            )
            expected.append(part)
        expected = pd.concat(expected, ignore_index=True)
        # Each profile/class has ten descriptions in the same saved row order.
        left = current.groupby(keys, sort=True).Score.mean()
        right = expected.groupby(keys, sort=True).Score.mean().reindex(left.index)
        np.testing.assert_allclose(left, right, atol=2e-6, rtol=1e-5)
        calls = pd.read_csv(prediction / "compound_predictions.csv").set_index("CompoundID")
        original = pd.read_csv(
            ROOT / "data/figure_inputs/figure5_application/full_vs_proteomics_low_score_calls.csv"
        )
        original = (
            original.loc[original.predictor.eq(label)].set_index("CompoundID").loc[calls.index]
        )
        np.testing.assert_array_equal(calls.PredictedMoA, original.PredictedMoA)
        np.testing.assert_allclose(calls.PredScore, original.top_score, atol=2e-6, rtol=1e-5)
        np.testing.assert_array_equal(calls.Uncertain, ~original.accepted)
        calibration = pd.read_csv(
            ROOT / f"data/analysis/reference_calibration/{arm}.csv", float_precision="round_trip"
        )
        correct = calibration.role.eq("id") & calibration.correct.eq(1)
        cutoff = calibrate_score_cutoff(calibration.PredScore, correct)
        np.testing.assert_equal(cutoff, config["score_cutoffs"][arm])
        summary.append(
            dict(
                arm=arm,
                seed_profile_class_means=len(left),
                maximum_score_difference=float(np.max(np.abs(left - right))),
                compound_calls_match=True,
                uncertainty_flags_match=True,
                cutoff_reproduced=True,
            )
        )
    report = ROOT / "results/verification/reference.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
