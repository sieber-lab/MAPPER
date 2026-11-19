"""Replay fixed nested meta-model fits using saved primary-model outputs."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from mapper.analysis.nested_meta import fit_fixed, apply_fitted
from mapper.analysis.replay import check_meta_replay, check_ranking_replay
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output directory")
    plan = json.loads((ROOT / "data/configurations/uncertainty_plan.json").read_text())
    reference = pd.read_csv(
        ROOT / "data/figure_inputs/uncertainty/fixed_outer_calls.csv", float_precision="round_trip"
    )
    rows, comparisons, ranking_tables = [], [], []
    args.output.mkdir(parents=True)
    for context in sorted(plan["contexts"], key=lambda item: item["context"]):
        name = context["context"]
        folder = ROOT / "data/analysis/uncertainty/contexts" / name
        read = lambda name: pd.read_csv(folder / name, float_precision="round_trip")
        development, test = read("meta_training.csv"), read("outer_predictions.csv")
        assert not set(development.CompoundID) & set(test.CompoundID)
        assert set(test.CompoundID) == set(context["test"])
        pairs = [
            (read(f"inner_{i}_meta_training.csv"), read(f"inner_{i}_calibration.csv"))
            for i in range(3)
        ]
        fitted, _ = fit_fixed(development, pairs)
        scored = apply_fitted(fitted, test)
        expected = reference.loc[reference.context.eq(name)].set_index(["CompoundID", "role"])
        observed = scored.set_index(["CompoundID", "role"]).loc[expected.index]
        check = check_meta_replay(
            observed.raw_risk,
            observed.p_error,
            observed.uncertain,
            expected.raw_risk,
            expected.risk,
            expected.flag,
        )
        np.testing.assert_allclose(
            fitted["raw_threshold"], expected.raw_threshold, rtol=1e-5, atol=1e-7
        )
        comparisons.append(dict(context=name, **check))
        ranking_tables.append(
            pd.DataFrame(
                dict(
                    actual=observed.p_error.to_numpy(),
                    expected=expected.risk.to_numpy(),
                actual_raw=observed.raw_risk.to_numpy(),
                expected_raw=expected.raw_risk.to_numpy(),
                    role=expected.index.get_level_values("role"),
                    correct=expected.correct.to_numpy(),
                )
            )
        )
        scored["context"] = name
        rows.append(scored)
        print(name, "matched", flush=True)
    ranked = pd.concat(ranking_tables, ignore_index=True)
    check_ranking_replay(ranked.actual, ranked.expected)
    check_ranking_replay(ranked.actual_raw, ranked.expected_raw)
    # The published ranking benchmark includes all 51 known-class and all 51
    # omitted-class predictions. Correct-known filtering is for false alarms,
    # not the AUROC/AP calculation.
    benchmark = ranked
    truth = benchmark.role.eq("ood").astype(int)
    metrics = {}
    for label, function in [
        ("auroc", roc_auc_score),
        ("average_precision", average_precision_score),
    ]:
        actual, expected = function(truth, benchmark.actual), function(truth, benchmark.expected)
        assert actual == expected, (label, actual, expected)
        metrics[label] = float(actual)
    pd.concat(rows, ignore_index=True).to_csv(args.output / "outer_predictions.csv", index=False)
    (args.output / "verification.json").write_text(
        json.dumps(
            dict(
                contexts=len(rows),
                fixed_original_architecture=True,
                primary_refits=0,
                flags_match=True,
                ranking_order_identical=True,
                benchmark_metrics_identical=metrics,
                benchmark_cases=len(benchmark),
                context_comparisons=comparisons,
                numerical_tolerances=dict(
                    raw=dict(rtol=1e-5, atol=1e-7), probability=dict(rtol=1e-5, atol=1e-6)
                ),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
