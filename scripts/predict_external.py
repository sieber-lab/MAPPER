"""Predict prepared 873-protein profiles with the matching frozen ensemble.

--context selects the training library and its independently calibrated
cutoff. Use the identity-excluded context when scoring a reference antibiotic.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from mapper.external import predict_checkpoint
from mapper.uncertainty import flag_uncertain
from mapper.paths import artifact_root


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--profiles",
        type=Path,
        required=True,
        help="NPZ from prepare_external.py; ECFP arms also need ecfp",
    )
    parser.add_argument(
        "--arm", choices=["proteomics", "ecfp_only", "ecfp_proteomics"], default="proteomics"
    )
    parser.add_argument("--context", default="full51")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output")
    import torch

    torch.set_num_threads(4)
    root = artifact_root() / ("external_ecfp" if args.arm == "ecfp_only" else "external")
    plan = json.loads((root / "prepared/plan.json").read_text())
    context = next((c for c in plan["contexts"] if c["context"] == args.context), None)
    if context is None:
        parser.error("Context is not in the fixed training plan")
    with np.load(root / "prepared/matrices.npz", allow_pickle=False) as source:
        text, text_classes = source["text"], source["text_classes"]
        proteins = source["genes"] if args.arm != "ecfp_only" else None
    with np.load(args.profiles, allow_pickle=False) as source:
        ids, compound_ids = source["profile_ids"], source["compound_ids"]
        if args.arm != "ecfp_only":
            np.testing.assert_array_equal(source["proteins"], proteins)
        blocks = {
            "proteomics": lambda: source["features"],
            "ecfp_only": lambda: source["ecfp"],
            "ecfp_proteomics": lambda: np.column_stack([source["features"], source["ecfp"]]),
        }
        values = blocks[args.arm]().copy()
    if set(compound_ids) & set(context["develop"]):
        parser.error(
            "A test compound remains in the selected training library; choose its omitted context"
        )
    scores = []
    for seed in plan["seeds"]:
        folder = root / "fits" / args.arm / context["main_job"] / f"seed_{seed}"
        scores.append(predict_checkpoint(folder, values, text))
    paired = np.mean(scores, axis=0)
    classes = sorted(set(text_classes))
    profile_scores = np.column_stack(
        [paired[:, text_classes == label].mean(axis=1) for label in classes]
    )
    table = pd.DataFrame(profile_scores, columns=classes)
    table["compound_id"] = compound_ids
    averaged = table.groupby("compound_id", sort=False)[classes].mean()
    cutoff = json.loads(
        (root / "calibration" / args.arm / args.context / "threshold.json").read_text()
    )["score_cutoff"]
    ranking = np.argsort(-averaged.to_numpy(), axis=1, kind="stable")
    output = averaged.copy()
    output["predicted_class"] = np.asarray(classes)[ranking[:, 0]]
    output["maximum_score"] = averaged.to_numpy().max(axis=1)
    output["uncertain"] = flag_uncertain(output.maximum_score, cutoff)
    output["score_cutoff"] = cutoff
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output)
    print(output[["predicted_class", "maximum_score", "uncertain"]].to_string())


if __name__ == "__main__":
    main()
