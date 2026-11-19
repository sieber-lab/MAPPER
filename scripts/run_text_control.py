"""Fit one fixed full-feature class-representation control on one saved fold."""

import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from sklearn.preprocessing import StandardScaler
from mapper.analysis.text_controls import (
    ARMS,
    assemble,
    is_multiclass,
    candidate_predictions,
    compound_predictions,
)


def load_inputs():
    folder = ROOT / "data/analysis/text_controls"
    with np.load(folder / "matrices.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    with np.load(folder / "text_banks.npz", allow_pickle=False) as source:
        data["banks"] = {key: source[key] for key in source.files}
    data["folds"] = json.loads((folder / "folds.json").read_text())["folds"]
    data["compound_labels"] = dict(zip(data["compounds"], data["labels"]))
    return data


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model", choices=["tabm", "lightgbm"], required=True)
    parser.add_argument("--iteration", type=int, choices=range(5), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument(
        "--fixed-duration",
        action="store_true",
        help="One-hot TabM: 100 epochs without early stopping",
    )
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.fixed_duration and (args.arm != "onehot" or args.model != "tabm"):
        parser.error("The retained fixed-duration diagnostic is one-hot TabM only")
    data = load_inputs()
    x, y, rows = assemble(data, args.arm, args.iteration, args.fold, training=True)
    test_x, _, test_rows = assemble(data, args.arm, args.iteration, args.fold, training=False)
    assert not set(rows.CompoundID) & set(test_rows.CompoundID)
    summary = dict(
        model=args.model,
        arm=args.arm,
        iteration=args.iteration,
        fold=args.fold,
        seed=42,
        training_pairs=len(y),
        input_width=x.shape[1],
        test_rows=len(test_x),
        fixed_duration=args.fixed_duration,
    )
    if args.inspect:
        print(json.dumps(summary, indent=2))
        return
    if args.output is None or args.output.exists():
        parser.error("A new --output directory is required")
    from mapper.models.factory import get_model

    if args.model == "tabm":
        import torch

        torch.set_num_threads(4)
    scaler = StandardScaler()
    model = get_model(
        args.model, input_dim=x.shape[1], n_classes=9 if is_multiclass(args.arm) else 2, seed=42
    )
    if args.fixed_duration:
        model.patience = 101
    if args.model == "lightgbm":
        model.set_params(n_jobs=4)
    model.fit(scaler.fit_transform(x), y)
    predictions = candidate_predictions(
        test_rows,
        model.predict_proba(scaler.transform(test_x)),
        data["classes"],
        is_multiclass(args.arm),
    )
    args.output.mkdir(parents=True)
    predictions.to_csv(args.output / "predictions.csv.gz", index=False)
    compound_predictions(predictions).to_csv(args.output / "compound_scores.csv", index=False)
    with (args.output / "model.pkl").open("wb") as stream:
        pickle.dump(dict(model=model, standard_scaler=scaler), stream)
    (args.output / "configuration.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
