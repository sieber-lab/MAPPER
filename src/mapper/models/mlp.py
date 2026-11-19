"""Original custom PyTorch/Lightning MLP used in the model comparison."""

import numpy as np
import pandas as pd
import random
from typing import List
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import TensorDataset, DataLoader
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.core.mixins import HyperparametersMixin
from sklearn.preprocessing import StandardScaler


class MLPPredictor(L.LightningModule, HyperparametersMixin):
    def __init__(
        self,
        input_dim: int,
        width_list: List = None,
        depth: int = 3,
        activation: str = "relu",
        lr: float = 1e-3,
        optimizer: str = "adam",
        dropout: float = 0.5,
        n_classes: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.width_list = width_list if width_list is not None else [256] * depth
        self.depth = depth
        self.activation = activation
        self.lr = lr
        self.optimizer = optimizer
        self.dropout = dropout
        self.n_classes = n_classes
        self.save_hyperparameters()

        layer_list = []
        layer_list.append(
            block2(
                input_dim=input_dim,
                width=self.width_list[0],
                activation=activation,
                dropout=dropout,
            )
        )
        for i in range(depth - 1):
            layer_list.append(
                block2(
                    input_dim=self.width_list[i],
                    width=self.width_list[i + 1],
                    activation=activation,
                    dropout=dropout,
                )
            )

        out_dim = 1 if n_classes == 2 else n_classes
        layer_list.append(nn.Linear(self.width_list[-1], out_dim))
        self.layers = nn.Sequential(*layer_list)
        self.layers.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        if self.n_classes == 2:
            loss = nn.BCEWithLogitsLoss()(logits[:, 0], y.float())
        else:
            loss = nn.CrossEntropyLoss()(logits, y.long())
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self.forward(x)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return optim.Adam(
                self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4
            )
        elif self.optimizer == "sgd":
            return optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer == "adamw":
            return optim.AdamW(self.parameters(), lr=self.lr)
        return optim.Adam(self.parameters(), lr=self.lr)


class block2(nn.Module):
    def __init__(self, input_dim, width, activation, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.activation = activation
        self.linear1 = nn.Linear(input_dim, width)
        self.batchnorm = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x


class SklearnLikeLightningMLP:
    def __init__(
        self,
        input_dim,
        batch_size=128,
        max_epochs=30,
        width_list=[256, 128],
        depth=2,
        activation="relu",
        lr=1e-4,
        dropout=0.5,
        n_classes: int = 2,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.width_list = width_list
        self.depth = depth
        self.activation = activation
        self.lr = lr
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = "gpu" if torch.cuda.is_available() else "cpu"  #'gpu' instead of 'cuda'?
        self.model = None
        self.trainer = None
        self.seed = seed

    def fit(self, X, y):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.model = MLPPredictor(
            input_dim=self.input_dim,
            width_list=self.width_list,
            depth=self.depth,
            activation=self.activation,
            lr=self.lr,
            optimizer="adam",
            dropout=self.dropout,
            n_classes=self.n_classes,
        )

        # dtype of y depends on task
        y_tensor = torch.tensor(
            y.values if hasattr(y, "values") else y,
            dtype=torch.long if self.n_classes > 2 else torch.float32,
        )

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32), y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        early_stop = EarlyStopping(monitor="train_loss", patience=5, mode="min", verbose=False)
        self.trainer = Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.device,
            logger=False,
            enable_checkpointing=False,
            callbacks=[early_stop],
        )
        self.trainer.fit(self.model, train_loader)
        return self

    def predict_proba(self, X):
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32), torch.zeros(len(X))),
            batch_size=self.batch_size,
        )
        preds = self.trainer.predict(self.model, dataloaders=test_loader)
        logits = torch.cat(preds).cpu()

        if self.n_classes == 2:
            p1 = torch.sigmoid(logits.squeeze(1))
            p0 = 1 - p1
            return np.vstack([p0.numpy(), p1.numpy()]).T
        else:
            probs = torch.softmax(logits, dim=1).numpy()
            return probs

    def predict(self, X):
        proba = self.predict_proba(X)
        if self.n_classes == 2:
            return (proba[:, 1] > 0.5).astype(int)
        return np.argmax(proba, axis=1)
