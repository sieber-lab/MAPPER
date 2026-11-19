"""Recompute the six training-library similarity annotations in Figures 4/ED2.

Uses the original ECFP bits and the recorded MCS definition. An incomplete
MCS search fails verification rather than being reported as an exact maximum.
Install requirements/chemistry_checks.txt for the recorded RDKit version.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFMCS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reports/similarities")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing results")
    raw = pd.read_pickle(ROOT / "data/inhouse/X_ecfp.pkl")
    raw.index = raw.index.str.rsplit("_", n=1).str[0]
    x = raw.groupby(level=0).first().astype(bool)
    source = ROOT / "data/figure_inputs"
    structures = pd.read_csv(source / "overview/structures.csv").set_index("compound_id")
    expected = pd.read_csv(source / "similarities/calculated_annotations.csv").set_index(
        "compound_id"
    )
    assert set(x.index) == set(structures.index) and len(x) == 51
    molecules = {
        cid: Chem.MolFromSmiles(row.smiles.split(".")[0]) for cid, row in structures.iterrows()
    }
    rows = []
    for cid in expected.index:
        query = molecules[cid]
        neighbors = structures.index[
            (structures.moa == structures.loc[cid, "moa"]) & structures.index.to_series().ne(cid)
        ]
        best_atoms, best_neighbor, cancelled = -1, None, 0
        for other in neighbors:
            result = rdFMCS.FindMCS(
                [query, molecules[other]],
                timeout=args.timeout,
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareOrder,
                ringMatchesRingOnly=True,
                completeRingsOnly=False,
                matchValences=True,
            )
            cancelled += int(result.canceled)
            if result.numAtoms > best_atoms:
                best_atoms, best_neighbor = result.numAtoms, other
        fingerprints = x.loc[neighbors]
        similarity = (fingerprints & x.loc[cid]).sum(axis=1) / (fingerprints | x.loc[cid]).sum(
            axis=1
        )
        rows.append(
            dict(
                compound_id=cid,
                mcs=best_atoms / query.GetNumAtoms(),
                mcs_neighbor=best_neighbor,
                tanimoto=float(similarity.max()),
                tanimoto_neighbor=similarity.idxmax(),
                cancelled_pairs=cancelled,
            )
        )
    results = pd.DataFrame(rows).set_index("compound_id")
    if not results.cancelled_pairs.eq(0).all():
        raise ValueError(
            "An incomplete MCS search cannot certify the maximum; rerun with a longer timeout"
        )
    np.testing.assert_allclose(results.mcs, expected.loc[results.index, "mcs"], atol=1e-12)
    np.testing.assert_allclose(
        results.tanimoto, expected.loc[results.index, "tanimoto"], atol=1e-12
    )
    args.output.mkdir(parents=True)
    results.to_csv(args.output / "annotations.csv")
    (args.output / "verification.json").write_text(
        json.dumps(
            dict(
                status="passed",
                rdkit=rdBase.rdkitVersion,
                reference_compounds=51,
                query_compounds=len(results),
                timeout_seconds_per_pair=args.timeout,
            ),
            indent=2,
        )
        + "\n"
    )
    print("Reproduced all six MCS and ECFP annotations using training-library neighbors only.")


if __name__ == "__main__":
    main()
