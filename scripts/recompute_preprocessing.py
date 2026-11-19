"""Recompute repeated-CV preprocessing metrics and their compound intervals."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from mapper.analysis.preprocessing_metrics import metric_values, weighted_metrics


def main():
    rows = []
    for name in ["missing_value", "normalization", "gene_mapping", "control_relative"]:
        folder = ROOT / "data/analysis/preprocessing" / name
        data = pd.read_csv(folder / "compound_candidate_scores.csv", float_precision="round_trip")
        reference = pd.read_csv(folder / "metric_summary.csv", float_precision="round_trip")
        compounds, classes = sorted(data.CompoundID.unique()), sorted(data.Text_MoA.unique())
        labels = data.groupby("CompoundID").True_MoA.first().reindex(compounds)
        truth = np.array([classes.index(label) for label in labels])
        weights = pd.read_csv(folder / "bootstrap_compound_multiplicities.csv", index_col=0)[
            compounds
        ].to_numpy()
        for (configuration, arm), part in data.groupby(["configuration", "arm"]):
            points, samples = [], []
            for iteration, block in part.groupby("cv_iter"):
                scores = (
                    block.pivot(index="CompoundID", columns="Text_MoA", values="ProbaPos")
                    .loc[compounds, classes]
                    .to_numpy()
                )
                points.append(metric_values(scores, truth))
                samples.append(weighted_metrics(scores, truth, weights))
            draws = np.mean(samples, axis=0)
            for index, metric in enumerate(
                ["top1_accuracy", "top2_accuracy", "macro_f1", "pairwise_pr_auc"]
            ):
                estimate = np.mean([point[metric] for point in points])
                low, high = np.quantile(draws[:, index], [0.025, 0.975])
                expected = reference[
                    reference.configuration.eq(configuration)
                    & reference.arm.eq(arm)
                    & reference.metric.eq(metric)
                ].iloc[0]
                np.testing.assert_allclose(
                    [estimate, low, high],
                    expected[["estimate", "ci_low", "ci_high"]].to_numpy(float),
                    atol=2e-12,
                    rtol=0,
                )
                rows.append(
                    dict(
                        preprocessing=name,
                        configuration=configuration,
                        arm=arm,
                        metric=metric,
                        estimate=estimate,
                        ci_low=low,
                        ci_high=high,
                    )
                )
            print(name, configuration, arm, "matched", flush=True)
    output = ROOT / "results/reports"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "preprocessing.csv", index=False)
    print(f"Reproduced {len(rows)} estimates and confidence intervals.")


if __name__ == "__main__":
    main()
