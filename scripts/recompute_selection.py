"""Reapply configuration selection to the saved paired bootstrap score arrays."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from mapper.analysis.selection_policies import policies


def main():
    folder = ROOT / "data/analysis/configuration_selection"
    configurations = pd.read_csv(
        folder / "included_configurations.csv", float_precision="round_trip"
    )
    reference = pd.read_csv(folder / "policy_summary.csv", float_precision="round_trip")
    with np.load(folder / "configuration_bootstrap_metrics.npz", allow_pickle=False) as archive:
        names = archive["configurations"].tolist()
        assert names == configurations.configuration.tolist() and len(names) == 192
        inbag = archive["inbag_fold_pr_auc"]
        metrics = {
            key: archive[key]
            for key in archive.files
            if key not in {"configurations", "inbag_fold_pr_auc"}
        }
    rows = []
    for policy, eligible in policies(names).items():
        selected = eligible[np.argmax(inbag[eligible], axis=0)]
        for metric, values in metrics.items():
            draws = values[selected, np.arange(values.shape[1])]
            estimate, low, high = draws.mean(), *np.quantile(draws, [0.025, 0.975])
            expected = reference[reference.policy.eq(policy) & reference.metric.eq(metric)].iloc[0]
            np.testing.assert_allclose(
                [estimate, low, high],
                expected[["mean_oob", "oob_percentile_2_5", "oob_percentile_97_5"]].to_numpy(float),
                atol=2e-12,
                rtol=0,
            )
            rows.append(
                dict(
                    policy=policy,
                    metric=metric,
                    mean_oob=estimate,
                    ci_low=low,
                    ci_high=high,
                    n_configurations=len(eligible),
                    n_draws=values.shape[1],
                )
            )
    output = ROOT / "results/reports"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "selection.csv", index=False)
    print(f"Reproduced {len(rows)} policy/metric summaries from paired bootstrap arrays.")


if __name__ == "__main__":
    main()
