"""Create 873-protein fold-change profiles from matched log2 abundances.

Input CSVs contain one mapped reference protein per row and one biological
profile per column, with protein identifiers in the first column. Mapping
ambiguities must be resolved using the supplied reference annotations first.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from mapper.analysis.representations import observed_mean


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--treated", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument(
        "--compound-id", required=True, help="All replicates in this file belong to this identity"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output")
    treated, controls = [pd.read_csv(p, index_col=0) for p in [args.treated, args.controls]]
    if not treated.index.is_unique or not controls.index.is_unique:
        parser.error("Resolve duplicated protein rows before preparing profiles")
    with np.load(ROOT / "data/protein_mapping/prepared/panel.npz", allow_pickle=False) as panel:
        proteins = panel["selected_genes"]
    t = treated.reindex(proteins).to_numpy(float).T
    c = controls.reindex(proteins).to_numpy(float).T
    if np.isinf(t).any() or np.isinf(c).any():
        parser.error("Use missing values, not infinite intensities")
    reference, _ = observed_mean(c)
    differences = t - reference
    present = np.isfinite(differences)
    features = np.where(present, differences, 0.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        observed=present,
        proteins=proteins,
        profile_ids=treated.columns.to_numpy(str),
        compound_ids=np.repeat(args.compound_id, len(t)),
    )
    print(f"Prepared {len(t)} profiles on {len(proteins)} proteins; no significance selection.")


if __name__ == "__main__":
    main()
