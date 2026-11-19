"""Frozen-checkpoint inference for the control-relative external input models."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np


def pair_inputs(features, text):
    """Original profile -> candidate/description order, text block first."""
    features, text = np.asarray(features), np.asarray(text)
    result = np.empty((len(features), len(text), text.shape[1] + features.shape[1]), dtype=float)
    result[:, :, : text.shape[1]] = text[None]
    result[:, :, text.shape[1] :] = features[:, None]
    return result.reshape(-1, result.shape[-1])


def restore_checkpoint(checkpoint, quantile):
    """Original TabM construction and stored preprocessing; no fitting."""
    from .models.factory import get_model

    wrapper = get_model(
        "tabm", input_dim=checkpoint["input_dim"], n_classes=2, seed=checkpoint["seed"]
    )
    wrapper.scaler = quantile
    wrapper.model = wrapper.Model(
        n_num_features=checkpoint["input_dim"],
        cat_cardinalities=[],
        n_classes=2,
        backbone=dict(type="MLP", n_blocks=3, d_block=512, dropout=0.1),
        bins=None,
        num_embeddings=None,
        arch_type="tabm",
        k=32,
        share_training_batches=True,
    ).to(wrapper.device)
    wrapper.model.load_state_dict(checkpoint["state_dict"])
    wrapper.model.eval()
    return wrapper


def predict_checkpoint(model_directory, features, text):
    """Return one matching score per profile/description for a trusted bundle.

    `features` must already use the fitted protein panel and block order.
    This does not align identifiers, fit scaling, select a panel or calibrate
    uncertainty. Both stored preprocessing stages are applied unchanged.
    """
    import torch

    directory = Path(model_directory)
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    with (directory / "preprocessors.pkl").open("rb") as stream:
        preprocessors = pickle.load(stream)
    raw = pair_inputs(features, text)
    if raw.shape[1] != checkpoint["input_dim"] or not np.isfinite(raw).all():
        raise ValueError("Input dimensions/values differ from the fitted representation")
    wrapper = restore_checkpoint(checkpoint, preprocessors["quantile_transformer"])
    ordinary, members = wrapper.predict_proba(
        preprocessors["standard_scaler"].transform(raw), return_experts=True, mc_passes=0
    )
    np.testing.assert_allclose(members.mean(axis=1), ordinary, atol=5e-7, rtol=0)
    return ordinary[:, 1].reshape(len(features), len(text)).copy()
