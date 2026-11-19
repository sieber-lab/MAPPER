"""Paper TabM wrapper; original training and inference definitions unchanged."""

import math
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from .tabm_reference import Model, make_parameter_groups


def _enable_dropout_only(model: nn.Module):
    """
    Keep the model in eval mode, but force *only* Dropout layers to train().
    This avoids BatchNorm running-stat updates while enabling stochastic dropout.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()


class TabMWrapper:
    def __init__(
        self,
        input_dim,
        n_epochs=100,
        batch_size=256,
        patience=3,
        lr=1e-4,
        weight_decay=3e-4,
        seed=None,
        n_classes: int = 2,
    ):
        self.Model = Model
        self.make_parameter_groups = make_parameter_groups
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.n_classes = n_classes
        self.model = None
        self.scaler = QuantileTransformer(output_distribution="normal", subsample=1_000_000)

    def _set_seed(self):
        seed = self.seed or 0
        random.seed(seed)
        np.random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 3)

    def fit(self, X, y):
        self._set_seed()
        X = pd.DataFrame(X)
        X.columns = X.columns.astype(str)
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

        # y dtype: CE expects long
        y_tensor = torch.tensor(
            (y.values if hasattr(y, "values") else y), dtype=torch.long, device=self.device
        )

        self.data = {"x": X_tensor, "y": y_tensor}

        self.model = self.Model(
            n_num_features=self.input_dim,
            cat_cardinalities=[],
            n_classes=self.n_classes,
            backbone={"type": "MLP", "n_blocks": 3, "d_block": 512, "dropout": 0.1},
            bins=None,
            num_embeddings=None,
            arch_type="tabm",
            k=32,
            share_training_batches=True,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.make_parameter_groups(self.model), lr=self.lr, weight_decay=self.weight_decay
        )

        def apply_model(idx):
            return self.model(self.data["x"][idx], None).float()  # (B, k, C)

        def loss_fn(y_pred, y_true):
            k = y_pred.shape[-2]
            return F.cross_entropy(y_pred.flatten(0, 1), y_true.repeat_interleave(k))

        best_loss = math.inf
        best_state = None
        patience_left = self.patience

        for epoch in range(self.n_epochs):
            self.model.train()
            losses = []
            for batch_idx in torch.randperm(len(y_tensor), device=self.device).split(
                self.batch_size
            ):
                optimizer.zero_grad()
                loss = loss_fn(apply_model(batch_idx), y_tensor[batch_idx])
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            epoch_loss = np.mean(losses)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = self.model.state_dict()
                patience_left = self.patience
            else:
                patience_left -= 1
            if patience_left <= 0:
                break

        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    # --- in Codebase/eval_utils3.py (class TabMWrapper) ---

    def predict_proba(self, X, return_experts: bool = False, mc_passes: int = 0):
        """
        If mc_passes > 1, run MC-dropout at inference (stochastic passes with Dropout active).
        Returns:
        - mean_arr: (N, C)
        - if return_experts:
            * when mc_passes <= 1: exps_arr (N, k, C)
            * when mc_passes >  1: exps_arr_mc (N, mc_passes, k, C)
        """
        X = pd.DataFrame(X)
        X.columns = X.columns.astype(str)
        X_scaled = self.scaler.transform(X)
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

        def _forward_batches():
            # one deterministic pass (Dropout disabled)
            all_mean, all_exps = [], []
            with torch.no_grad():
                for batch in torch.arange(len(x_tensor), device=self.device).split(8192):
                    out = self.model(x_tensor[batch], None).float()  # (B, k, C) or (B, k, 1)
                    if self.n_classes == 2:
                        if out.shape[-1] == 1:
                            p1 = torch.sigmoid(out).squeeze(-1)  # (B, k)
                            p0 = 1.0 - p1
                            probs = torch.stack([p0, p1], dim=-1)  # (B, k, 2)
                        else:
                            probs = torch.softmax(out, dim=-1)  # (B, k, 2)
                    else:
                        probs = torch.softmax(out, dim=-1)  # (B, k, C)
                    mean_probs = probs.mean(dim=1)  # (B, C)
                    all_mean.append(mean_probs.cpu().numpy())
                    if return_experts:
                        all_exps.append(probs.cpu().numpy())
            mean_arr = np.concatenate(all_mean, axis=0)
            if return_experts:
                exps_arr = np.concatenate(all_exps, axis=0)  # (N, k, C)
                return mean_arr, exps_arr
            return mean_arr

        if mc_passes is None or mc_passes <= 1:
            return _forward_batches()

        # --- MC-Dropout: T stochastic passes with only Dropout active
        T = int(mc_passes)
        all_T = []
        with torch.no_grad():
            for _ in range(T):
                _enable_dropout_only(self.model)  # enable only Dropout stochasticity
                pass_mean = []
                pass_exps = [] if return_experts else None
                for batch in torch.arange(len(x_tensor), device=self.device).split(8192):
                    out = self.model(x_tensor[batch], None).float()  # (B, k, C) or (B, k, 1)
                    if self.n_classes == 2:
                        if out.shape[-1] == 1:
                            p1 = torch.sigmoid(out).squeeze(-1)
                            p0 = 1.0 - p1
                            probs = torch.stack([p0, p1], dim=-1)
                        else:
                            probs = torch.softmax(out, dim=-1)
                    else:
                        probs = torch.softmax(out, dim=-1)
                    pass_mean.append(probs.mean(dim=1).cpu().numpy())  # (B, C)
                    if return_experts:
                        pass_exps.append(probs.cpu().numpy())  # (B, k, C)
                all_T.append(
                    (
                        np.concatenate(pass_mean, axis=0),
                        np.concatenate(pass_exps, axis=0) if return_experts else None,
                    )
                )

        # stack across T
        mean_stack = np.stack([m for (m, _) in all_T], axis=1)  # (N, T, C)
        mean_arr = mean_stack.mean(axis=1)  # (N, C)

        if return_experts:
            exps_stack = np.stack([e for (_, e) in all_T], axis=1)  # (N, T, k, C)
            return mean_arr, exps_stack

        return mean_arr

    def predict(self, X):
        proba = self.predict_proba(X)
        if self.n_classes == 2:
            return (proba[:, 1] > 0.5).astype(int)
        return np.argmax(proba, axis=1)
