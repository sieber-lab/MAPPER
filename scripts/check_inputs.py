"""Check immutable figure tables and the main reported numerical endpoints."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from mapper.figures.style import INPUTS, read, sha
from mapper.figures.operator_bootstrap import verify_archive as check_operator
from mapper.figures.representation_comparison import verify_archive as check_representation
from mapper.figures.uncertainty import final_calls


def main():
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        assert sha(INPUTS / name) == record["sha256"], name
    check_operator()
    check_representation()
    assert len(read("overview", "ecfp_embedding.csv")) == 51
    assert len(read("overview", "proteomics_embedding.csv")) == 200
    calls = final_calls()
    for method, omitted, false_alarm in [("low_max_score", 35, 2), ("meta_model", 22, 4)]:
        rows = calls.loc[calls.method.eq(method)]
        assert int(rows.loc[rows.role.eq("ood"), "flag"].sum()) == omitted
        assert int(rows.loc[rows.role.eq("id") & rows.correct.eq(1), "flag"].sum()) == false_alarm
    pair = read("difficulty", "paired_role_calls.csv").query('method=="low_max_score"')
    grid = pd.crosstab(pair.id_flag, pair.ood_flag).reindex(
        index=[False, True], columns=[False, True], fill_value=0
    )
    np.testing.assert_array_equal(grid.to_numpy(), [[14, 33], [2, 2]])
    external = read("external_comparison", "case_comparison_with_similarity.csv")
    for arm, correct in [("proteomics", 22), ("ecfp_only", 34), ("ecfp_proteomics", 37)]:
        rows = external.loc[external.arm.eq(arm)]
        assert len(rows) == 43 and int(rows.predicted.eq(rows.expected).sum()) == correct
    conditions = read("figure3", "metric_source.csv")
    assert conditions.groupby(["Model", "Method", "metric_key"]).size().eq(25).all()
    annotations = read("similarities", "calculated_annotations.csv").set_index("compound_id")
    np.testing.assert_allclose(annotations.loc["JH32", "mcs"], 1 / 3)
    np.testing.assert_allclose(annotations.loc["JH27", "tanimoto"], 17 / 33)
    print(
        json.dumps(
            dict(
                status="passed",
                frozen_files=len(manifest["files"]),
                compounds=51,
                biological_profiles=200,
                external_cases=43,
            )
        )
    )


if __name__ == "__main__":
    main()
