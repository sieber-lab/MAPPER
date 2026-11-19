"""Recompute the six-method ranking and threshold results from paired calls."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from mapper.analysis.uncertainty_metrics import ranking_draws


def main():
    source = ROOT / "data/figure_inputs/uncertainty"
    calls = pd.read_csv(source / "six_method_calls.csv", float_precision="round_trip")
    calls["method"] = calls.method.replace({"fixed_original_meta": "meta_model"})
    reference = pd.read_csv(source / "metrics.csv", float_precision="round_trip")
    reference["method"] = reference.method.replace({"fixed_original_meta": "meta_model"})
    methods = [
        "low_max_score",
        "high_predictive_entropy",
        "high_expected_entropy",
        "z_gate_global",
        "z_gate_class",
        "meta_model",
    ]
    with np.load(
        ROOT / "data/analysis/uncertainty/bootstrap_weights.npz", allow_pickle=False
    ) as saved:
        weights = saved["compound_within_class"]
        order = saved["compound_order"].tolist()
    rows = []
    for method in methods:
        part = calls.loc[calls.method.eq(method)]
        known = part.loc[part.role.eq("id")].set_index("CompoundID").loc[order]
        omitted = part.loc[part.role.eq("ood")].set_index("CompoundID").loc[order]
        joined = pd.concat([known, omitted])
        labels, risk = joined.role.eq("ood").to_numpy(int), joined.risk.to_numpy(float)
        draws = ranking_draws(labels, risk, np.c_[weights, weights])
        correct = known.correct.to_numpy(int)
        known_flags, omitted_flags = known.flag.to_numpy(int), omitted.flag.to_numpy(int)
        draws.update(
            ood_detection=(weights @ omitted_flags) / weights.sum(axis=1),
            correct_id_false_alarm=(weights @ (known_flags * correct)) / (weights @ correct),
        )
        row = dict(
            method=method,
            auroc=float(roc_auc_score(labels, risk)),
            average_precision=float(average_precision_score(labels, risk)),
            ood_detection=float(omitted_flags.mean()),
            correct_id_false_alarm=float((known_flags * correct).sum() / correct.sum()),
            ood_count=int(omitted_flags.sum()),
            ood_n=len(omitted),
            correct_id_count=int((known_flags * correct).sum()),
            correct_id_n=int(correct.sum()),
            known_errors_flagged=int(known_flags[correct == 0].sum()),
            known_errors_n=int((correct == 0).sum()),
        )
        for metric, values in draws.items():
            row[metric + "_ci_low"], row[metric + "_ci_high"] = np.quantile(values, [0.025, 0.975])
        expected = reference.loc[reference.method.eq(method)].iloc[0]
        for field, value in row.items():
            if field != "method":
                np.testing.assert_allclose(
                    value, expected[field], atol=2e-12, rtol=0, err_msg=method + ":" + field
                )
        rows.append(row)
    destination = ROOT / "results/reports"
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / "uncertainty_comparison.csv", index=False)
    print(
        json.dumps(
            dict(
                status="passed",
                methods=len(rows),
                paired_compounds=len(order),
                bootstrap_draws=len(weights),
            )
        )
    )


if __name__ == "__main__":
    main()
