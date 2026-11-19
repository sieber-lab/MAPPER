"""Matched known-class metrics and compound-bootstrap intervals (Table S15)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from mapper.analysis.bootstrap import bootstrap_weights, macro_f1


def main():
    folder = ROOT / "data/analysis/significance_selection"
    calls = pd.read_csv(folder / "compound_predictions.csv", float_precision="round_trip")
    reference = pd.read_csv(folder / "reference_metrics.csv", float_precision="round_trip")
    compounds = sorted(calls.CompoundID.unique())
    truth = calls.groupby("CompoundID").TrueMoA.first().reindex(compounds).to_numpy()
    weights = np.r_[np.ones((1, len(compounds))), bootstrap_weights(truth, 5000, 20260920)]
    rows = []
    for arm in ("unfiltered", "nominal", "bh005"):
        part = (
            calls.loc[calls.arm.eq(arm) & calls.role.eq("id")]
            .set_index("CompoundID")
            .loc[compounds]
        )
        correct = part.TrueMoA.eq(part.PredictedMoA).to_numpy()
        top2 = correct | part.TrueMoA.eq(part.SecondMoA).to_numpy()
        values = dict(
            top1_accuracy=(weights @ correct) / weights.sum(axis=1),
            top2_accuracy=(weights @ top2) / weights.sum(axis=1),
            macro_f1=macro_f1(part.TrueMoA, part.PredictedMoA, weights, sorted(set(truth))),
        )
        for metric, samples in values.items():
            lower, upper = np.quantile(samples[1:], [0.025, 0.975])
            expected = reference.loc[
                reference.arm.eq(arm)
                & reference.scheme.eq("compound_within_class")
                & reference.scope.eq("id")
                & reference.method.eq("classifier")
                & reference.metric.eq(metric)
            ].iloc[0]
            np.testing.assert_allclose(
                [samples[0], lower, upper],
                expected[["estimate", "ci_low", "ci_high"]].to_numpy(float),
                atol=2e-12,
                rtol=0,
            )
            rows.append(
                dict(selection=arm, metric=metric, estimate=samples[0], ci_low=lower, ci_high=upper)
            )
    output = ROOT / "results/reports"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "significance_selection.csv", index=False)
    print("Reproduced all nine point estimates and confidence intervals.")


if __name__ == "__main__":
    main()
