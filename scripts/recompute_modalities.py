"""Recalculate exact TabM Shapley values and matched InfoAlign increments."""

from pathlib import Path
import itertools
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd


def main():
    source = ROOT / "data/figure_inputs/model_screen"
    folds = pd.read_csv(source / "all_fold_metrics.csv", float_precision="round_trip")
    reference = pd.read_csv(source / "verified_tabm_shapley.csv", float_precision="round_trip")
    features = ["Proteomics", "ECFP", "CellPaint", "MIC", "GC"]
    means = folds.loc[folds.model_type.eq("tabm")].groupby("var_tag").pr_auc.mean()
    values = {
        frozenset(tag.split("+")) if tag != "(Text only)" else frozenset(): value
        for tag, value in means.items()
    }
    assert len(values) == 32
    rows, contrasts = [], []
    for feature in features:
        others = [f for f in features if f != feature]
        shapley = 0.0
        for count in range(5):
            for items in itertools.combinations(others, count):
                subset = frozenset(items)
                shapley += (
                    math.factorial(count)
                    * math.factorial(4 - count)
                    / math.factorial(5)
                    * (values[subset | {feature}] - values[subset])
                )
        np.testing.assert_allclose(
            shapley,
            reference.loc[reference.feature.eq(feature), "shapley"].item(),
            atol=2e-12,
            rtol=0,
        )
        rows.append(dict(feature=feature.replace("CellPaint", "InfoAlign"), shapley=shapley))
    for subset in values:
        if "CellPaint" not in subset:
            contrasts.append(
                dict(
                    features="+".join(
                        f.replace("CellPaint", "InfoAlign") for f in features if f in subset
                    )
                    or "Text only",
                    without_infoalign=values[subset],
                    with_infoalign=values[subset | {"CellPaint"}],
                    difference=values[subset | {"CellPaint"}] - values[subset],
                )
            )
    out = ROOT / "results/reports"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "tabm_shapley.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "infoalign_contribution.csv", index=False)
    print("Reproduced five exact Shapley values and 16 matched InfoAlign increments.")


if __name__ == "__main__":
    main()
