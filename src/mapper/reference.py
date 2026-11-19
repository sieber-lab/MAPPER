"""Prediction with the final 51-antibiotic models used in Figure 5."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .external import predict_checkpoint
from .paths import artifact_root, repository_root
from .uncertainty import flag_uncertain


def predict(profiles, arm="multimodal"):
    root = repository_root()
    config = json.loads((root / "data/configurations/reference.json").read_text())
    if arm not in config["numeric_widths"]:
        raise ValueError("Choose multimodal or proteomics")
    with np.load(root / "data/inhouse/paper_training.npz", allow_pickle=False) as source:
        text, text_classes = source["text"], source["text_classes"]
    values = np.asarray(profiles[arm], dtype=float)
    ids, compounds = np.asarray(profiles["sample_ids"]).astype(str), np.asarray(
        profiles["compounds"]
    ).astype(str)
    if values.shape != (len(ids), config["numeric_widths"][arm]) or len(compounds) != len(ids):
        raise ValueError("Profiles do not match the declared feature order and width")
    if len(set(ids)) != len(ids) or not np.isfinite(values).all():
        raise ValueError("Profile IDs must be unique and model inputs finite")
    if set(compounds) & set(config["jobs"]["full51"]["train"]):
        raise ValueError(
            "A test identity is in this training library; use compound-held-out validation instead"
        )
    by_seed = []
    for seed in config["seeds"]:
        scores = predict_checkpoint(artifact_root() / f"reference/{arm}/seed_{seed}", values, text)
        frame = pd.DataFrame(
            dict(
                SampleID=np.repeat(ids, len(text)),
                CompoundID=np.repeat(compounds, len(text)),
                CandidateMoA=np.tile(text_classes, len(ids)),
                Score=scores.ravel(),
                Seed=seed,
            )
        )
        by_seed.append(frame)
    pairs = pd.concat(by_seed, ignore_index=True)
    profile_scores = pairs.groupby(
        ["CompoundID", "SampleID", "CandidateMoA"], sort=True
    ).Score.mean()
    mean = profile_scores.groupby(level=["CompoundID", "CandidateMoA"]).mean().unstack()
    ranked = np.argsort(-mean.to_numpy(), axis=1, kind="stable")
    result = mean.copy()
    result["PredictedMoA"] = mean.columns.to_numpy()[ranked[:, 0]]
    result["PredScore"] = mean.to_numpy().max(axis=1)
    result["SecondMoA"] = mean.columns.to_numpy()[ranked[:, 1]]
    result["SecondScore"] = np.take_along_axis(mean.to_numpy(), ranked[:, 1:2], axis=1)[:, 0]
    result["Margin"] = result.PredScore - result.SecondScore
    result["Uncertain"] = flag_uncertain(result.PredScore, config["score_cutoffs"][arm])
    result["ScoreCutoff"] = config["score_cutoffs"][arm]
    return result, pairs


def main():
    import argparse

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--profiles", type=Path, default=repository_root() / "data/holdout/paper_profiles.npz"
    )
    parser.add_argument("--arm", choices=["multimodal", "proteomics"], default="multimodal")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing predictions")
    import torch

    torch.set_num_threads(4)
    with np.load(args.profiles, allow_pickle=False) as source:
        profiles = {key: source[key] for key in source.files}
    summary, pairwise = predict(profiles, args.arm)
    args.output.mkdir(parents=True)
    summary.to_csv(args.output / "compound_predictions.csv")
    pairwise.to_csv(args.output / "pairwise_predictions.csv.gz", index=False)
    print(summary[["PredictedMoA", "PredScore", "Uncertain"]].to_string())


if __name__ == "__main__":
    main()
