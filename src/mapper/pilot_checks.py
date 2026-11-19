"""Checkpoint and reporting checks used only by bounded reproducibility pilots."""

import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def checkpoint_roundtrip(model, values, folder, model_type):
    """Reload fitted parameters without refitting or changing training settings."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    probe = np.asarray(values[: min(128, len(values))])
    expected = model.predict_proba(probe)
    if model_type == "tabm":
        import torch
        from .external import restore_checkpoint

        state = dict(
            input_dim=model.input_dim,
            seed=model.seed,
            n_classes=model.n_classes,
            state_dict={k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()},
        )
        torch.save(state, folder / "checkpoint.pt")
        with (folder / "quantile.pkl").open("wb") as stream:
            pickle.dump(model.scaler, stream)
        with (folder / "quantile.pkl").open("rb") as stream:
            quantile = pickle.load(stream)
        saved = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        if model.n_classes == 2:
            restored = restore_checkpoint(saved, quantile)
        else:
            from .models.factory import get_model

            restored = get_model(
                "tabm", input_dim=model.input_dim, n_classes=model.n_classes, seed=model.seed
            )
            restored.scaler = quantile
            restored.model = model.Model(
                n_num_features=model.input_dim,
                cat_cardinalities=[],
                n_classes=model.n_classes,
                backbone=dict(type="MLP", n_blocks=3, d_block=512, dropout=0.1),
                bins=None,
                num_embeddings=None,
                arch_type="tabm",
                k=32,
                share_training_batches=True,
            ).to(restored.device)
            restored.model.load_state_dict(saved["state_dict"])
            restored.model.eval()
    elif model_type == "pytorch_mlp":
        import torch
        from lightning.pytorch import Trainer
        from .models.factory import get_model
        from .models.mlp import MLPPredictor

        state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
        torch.save(state, folder / "checkpoint.pt")
        restored = get_model(
            model_type, input_dim=model.input_dim, n_classes=model.n_classes, seed=model.seed
        )
        restored.model = MLPPredictor(
            input_dim=model.input_dim,
            width_list=model.width_list,
            depth=model.depth,
            activation=model.activation,
            lr=model.lr,
            optimizer="adam",
            dropout=model.dropout,
            n_classes=model.n_classes,
        )
        restored.model.load_state_dict(
            torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        )
        restored.trainer = Trainer(
            accelerator=restored.device,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
    else:
        with (folder / "model.pkl").open("wb") as stream:
            pickle.dump(model, stream)
        with (folder / "model.pkl").open("rb") as stream:
            restored = pickle.load(stream)
    actual = restored.predict_proba(probe)
    np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=1e-6)
    result = dict(
        rows=len(probe),
        maximum_absolute_difference=float(np.max(np.abs(actual - expected))),
        checkpoint_reload_passed=True,
    )
    (folder / "roundtrip.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def report_pairs(frame, output):
    """Original profile-then-compound aggregation, never a cross-fold pool."""
    profiles = frame.groupby(["CompoundID", "True_MoA", "SampleID", "CandidateMoA"]).Score.mean()
    compounds = (
        profiles.groupby(level=["CompoundID", "True_MoA", "CandidateMoA"]).mean().reset_index()
    )
    wide = compounds.pivot(index=["CompoundID", "True_MoA"], columns="CandidateMoA", values="Score")
    if not np.isfinite(wide.to_numpy()).all():
        raise ValueError("Incomplete candidate grid")
    order = np.argsort(-wide.to_numpy(), axis=1, kind="stable")
    labels = wide.index.get_level_values("True_MoA").to_numpy()
    prediction = wide.columns.to_numpy()[order[:, 0]]
    top2 = np.any(wide.columns.to_numpy()[order[:, :2]] == labels[:, None], axis=1)
    result = dict(
        compounds=len(wide),
        top1_accuracy=float(accuracy_score(labels, prediction)),
        top2_accuracy=float(top2.mean()),
        macro_f1=float(f1_score(labels, prediction, average="macro", zero_division=0)),
    )
    compounds.to_csv(Path(output) / "compound_scores.csv", index=False)
    (Path(output) / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
