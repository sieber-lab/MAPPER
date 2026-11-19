"""Fixed class-representation control matrices and profile-first aggregation."""

import numpy as np
import pandas as pd
from .control_vectors import label_permutation

ARMS = [
    "semantic_aug",
    "unaugmented",
    "onehot",
    "random_fixed_0",
    "random_fixed_1",
    "random_fixed_2",
    "random_augmented",
    "nonsense",
    "description_permuted",
    "label_permuted_0",
    "label_permuted_1",
    "label_permuted_2",
    "feature_only_native",
    "feature_only_matched",
]


def profile_indices(data, compounds):
    return np.flatnonzero(np.isin(data["compounds"], compounds))


def is_multiclass(arm):
    return arm.startswith("feature_only_")


def text_bank(data, arm, view=None):
    if view is not None and arm == "heldout_descriptions":
        ids = {"unseen_9_10": [9, 10], "seen_3_4": [3, 4], "seen_1_to_8": list(range(1, 9))}[view]
        return data["banks"]["semantic_aug"][:, np.array(ids) - 1], ids
    if arm == "semantic_aug" and view in ["original_only", "wording_9_10"]:
        ids = [1] if view == "original_only" else [9, 10]
        return data["banks"]["semantic_aug"][:, np.array(ids) - 1], ids
    key = "semantic_aug" if arm.startswith("label_permuted_") else arm
    if arm == "heldout_descriptions":
        ids = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2]
    elif arm == "unaugmented":
        ids = [1] * 10
    else:
        ids = list(range(1, 11))
    return data["banks"][key], ids


def assemble(data, arm, iteration, fold, *, training, view="standard"):
    """Return X, target and row provenance without fitting any transformation."""
    assert arm in ARMS
    split = data["folds"][iteration][fold]
    profiles = profile_indices(data, split["train" if training else "test"])
    true_labels = data["labels"]
    train_labels = true_labels.copy()
    if training and arm.startswith("label_permuted_"):
        mapping, _ = label_permutation(
            data["compound_labels"], split, iteration, fold, int(arm.rsplit("_", 1)[1])
        )
        for p in profiles:
            train_labels[p] = mapping[data["compounds"][p]]
    if is_multiclass(arm):
        repetitions = 90 if training and arm == "feature_only_matched" else 1
        p = np.repeat(profiles, repetitions)
        frame = pd.DataFrame(
            {
                "ProfileIndex": p,
                "SampleID": data["sample_ids"][p],
                "CompoundID": data["compounds"][p],
                "True_MoA": true_labels[p],
                "Training_MoA": train_labels[p],
                "ExposureSlot": np.tile(np.arange(repetitions), len(profiles)),
            }
        )
        y = pd.Index(data["classes"]).get_indexer(train_labels[p])
        assert (y >= 0).all()
        return data["numeric"][p], y, frame
    bank, description_ids = text_bank(data, arm, None if training else view)
    n_classes, n_aug, _ = bank.shape
    p = np.repeat(profiles, n_classes * n_aug)
    candidates = np.tile(np.repeat(np.arange(n_classes), n_aug), len(profiles))
    slots = np.tile(np.arange(n_aug), len(profiles) * n_classes)
    ids = np.asarray(description_ids)[slots]
    text = bank[candidates, slots]
    x = np.concatenate([text, data["numeric"][p]], axis=1)
    frame = pd.DataFrame(
        {
            "RowID": p * 90 + candidates * 10 + slots,
            "ProfileIndex": p,
            "SampleID": data["sample_ids"][p],
            "CompoundID": data["compounds"][p],
            "True_MoA": true_labels[p],
            "Training_MoA": train_labels[p],
            "CandidateMoA": data["classes"][candidates],
            "Aug_Index": slots + 1,
            "DescriptionID": ids,
        }
    )
    # RowID exactly matches historical rows only for the ten-slot standard view.
    frame["Label"] = (frame.CandidateMoA == frame.Training_MoA).astype(int)
    return x, frame.Label.to_numpy(int), frame


def candidate_predictions(frame, scores, classes, multiclass):
    if not multiclass:
        out = frame.copy()
        out["Score"] = np.asarray(scores)[:, 1]
        return out
    scores = np.asarray(scores)
    assert scores.shape == (len(frame), len(classes))
    out = frame.loc[frame.index.repeat(len(classes))].reset_index(drop=True)
    out["CandidateMoA"] = np.tile(classes, len(frame))
    out["Score"] = scores.reshape(-1)
    out["Label"] = (out.True_MoA == out.CandidateMoA).astype(int)
    return out


def compound_predictions(pairwise):
    # First average descriptions within a biological profile, then profiles.
    # Each profile has the same number of candidate texts within any given view.
    profiles = (
        pairwise.groupby(["CompoundID", "True_MoA", "SampleID", "CandidateMoA"], sort=True)
        .Score.mean()
        .reset_index()
    )
    return (
        profiles.groupby(["CompoundID", "True_MoA", "CandidateMoA"], sort=True)
        .Score.mean()
        .reset_index()
    )
