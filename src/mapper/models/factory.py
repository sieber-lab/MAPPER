"""The six original model families and their fixed paper settings."""


def get_model(name, input_dim=None, n_classes: int = 2, seed: int = 42):
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        if n_classes == 2:
            return LGBMClassifier(
                n_estimators=200,
                random_state=seed,
                min_data_in_leaf=1,
                min_data_in_bin=1,
                feature_pre_filter=False,
                verbosity=-1,
                class_weight="balanced",
                bagging_fraction=0.9,
                feature_fraction=0.9,
                bagging_freq=1,
            )
        else:
            return LGBMClassifier(
                n_estimators=200,
                random_state=seed,
                objective="multiclass",
                num_class=n_classes,
                min_data_in_leaf=1,
                min_data_in_bin=1,
                feature_pre_filter=False,
                verbosity=-1,
                bagging_fraction=0.9,
                feature_fraction=0.9,
                bagging_freq=1,
            )

    elif name == "xgboost":
        from xgboost import XGBClassifier

        if n_classes == 2:
            return XGBClassifier(
                n_estimators=300,
                use_label_encoder=False,
                eval_metric="logloss",
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
            )
        else:
            return XGBClassifier(
                n_estimators=300,
                use_label_encoder=False,
                eval_metric="mlogloss",
                objective="multi:softprob",
                num_class=n_classes,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
            )

    elif name == "mlp":
        from sklearn.neural_network import MLPClassifier

        # sklearn MLP handles multiclass natively
        return MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400, random_state=seed)

    elif name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=400, random_state=seed, class_weight="balanced" if n_classes == 2 else None
        )

    elif name == "pytorch_mlp":
        from .mlp import SklearnLikeLightningMLP

        if input_dim is None:
            raise ValueError("Must pass input_dim for Pytorch MLP.")
        return SklearnLikeLightningMLP(
            input_dim=input_dim,
            batch_size=128,
            max_epochs=40,
            width_list=[512, 256],
            depth=2,
            activation="relu",
            lr=1e-4,
            dropout=0.5,
            n_classes=n_classes,
            seed=seed,
        )

    elif name == "tabm":
        from .tabm import TabMWrapper

        if input_dim is None:
            raise ValueError("Must pass input_dim for TabM.")
        return TabMWrapper(input_dim=input_dim, n_classes=n_classes, seed=seed)

    else:
        raise ValueError(f"Unsupported model: {name}")
