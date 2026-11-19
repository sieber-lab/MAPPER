"""Original compound-held-out evaluation and feature assembly."""

import os
import json
import warnings
from dataclasses import dataclass
from typing import List, Iterable, Dict, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score
from sklearn.utils import shuffle
from tqdm import tqdm


def get_model(*args, **kwargs):
    from .models.factory import get_model as factory

    return factory(*args, **kwargs)


def _mlflow():
    import mlflow

    return mlflow


class _Tracking:
    def __getattr__(self, name):
        return getattr(_mlflow(), name)


mlflow = _Tracking()


@dataclass
class FoldSpec:
    train_compounds: set
    test_compounds: set


def _cmp_label_map(true_moa_by_sample: pd.Series, compound_by_sample: pd.Series) -> pd.Series:
    df = pd.DataFrame({"MoA": true_moa_by_sample, "CompoundID": compound_by_sample})
    per_cmp = df.groupby("CompoundID")["MoA"].nunique()
    bad = per_cmp[per_cmp > 1]
    if len(bad):
        raise ValueError(
            "Some compounds map to multiple MoAs; cannot stratify at the compound level. "
            f"Examples: {list(bad.index)[:5]}"
        )
    return df.groupby("CompoundID")["MoA"].first()


def build_or_load_compound_folds(
    true_moa_by_sample: pd.Series,
    compound_by_sample: pd.Series,
    n_splits: int,
    n_iters: int,
    shuffle: bool,
    base_seed: int,
    manifest_path: str | None,
):
    """
    Return `folds`: list of length n_iters; each item is a list of length n_splits of FoldSpec.
    If `manifest_path` exists, load and validate; else build, save, and return.
    Validation: same compounds and same MoA per compound as current data.
    """
    cmp_to_moa = _cmp_label_map(true_moa_by_sample, compound_by_sample)
    compounds = cmp_to_moa.index.to_numpy()
    y_cmp = cmp_to_moa.values.astype(str)

    def _serialize(folds):
        serial = {
            "n_splits": n_splits,
            "n_iters": n_iters,
            "base_seed": base_seed,
            "compounds": list(map(str, compounds)),
            "labels": list(map(str, y_cmp)),
            "folds": [
                [
                    {
                        "train": sorted(list(map(str, fs.train_compounds))),
                        "test": sorted(list(map(str, fs.test_compounds))),
                    }
                    for fs in iter_folds
                ]
                for iter_folds in folds
            ],
        }
        return serial

    def _deserialize(data):
        if (data.get("n_splits") != n_splits) or (data.get("n_iters") != n_iters):
            raise ValueError("Existing manifest has different n_splits/n_iters.")
        if data.get("compounds") != list(map(str, compounds)):
            raise ValueError("Existing manifest compounds differ from current data.")
        if data.get("labels") != list(map(str, y_cmp)):
            raise ValueError("Existing manifest compound labels differ from current data.")
        out = []
        for iter_folds in data["folds"]:
            cur = []
            for x in iter_folds:
                cur.append(FoldSpec(train_compounds=set(x["train"]), test_compounds=set(x["test"])))
            out.append(cur)
        return out

    # Load if present
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            data = json.load(f)
        return _deserialize(data)

    # Build fresh
    folds = []
    from sklearn.model_selection import StratifiedKFold

    for it in range(n_iters):
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=int(base_seed + 1000 * it)
        )
        iter_folds = []
        for tr_idx, te_idx in skf.split(compounds, y_cmp):
            tr_set = set(compounds[tr_idx].astype(str))
            te_set = set(compounds[te_idx].astype(str))
            iter_folds.append(FoldSpec(train_compounds=tr_set, test_compounds=te_set))
        folds.append(iter_folds)

    # Save if path given
    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(_serialize(folds), f, indent=2)

    return folds


def prepare_features_for_sample(sample_id, feature_dicts):
    return {mod: np.array(list(feature_dicts[mod][sample_id].values())) for mod in feature_dicts}


def build_features(df, modalities):
    return np.stack(df.apply(lambda r: np.concatenate([r[m] for m in modalities]), axis=1))


def generate_backbone_df(text_df, model_name, y_labels, sample_ids, text_col="Embedding"):
    df = text_df[text_df["Model"] == model_name].reset_index(drop=True)
    text_embeddings = df.groupby("MoA")[text_col].apply(list).to_dict()

    rows = []
    for sample_id in sample_ids:
        true_moa = y_labels[sample_id]
        for moa, embeddings in text_embeddings.items():
            for i, emb in enumerate(embeddings):
                rows.append(
                    {
                        "SampleID": sample_id,
                        "CompoundID": sample_id.split("_")[0],
                        "True_MoA": true_moa,
                        "Text_MoA": moa,
                        "Text_Embedding": emb,
                        "Aug_Index": i + 1,
                        "Label": int(moa == true_moa),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_model(
    model_type: str,
    X,
    y,
    groups,
    cv_splits: int = 5,
    seed_splits: int = 5,
    base_seed: int = 42,
    true_moa_by_sample: pd.Series | None = None,
    compound_by_sample: pd.Series | None = None,
    shuffle: bool = True,
    random_state: int = 42,
    predictions_dir: str | None = None,
    run_name: str | None = None,
    sample_ids: pd.Series | None = None,
    candidate_moas: pd.Series | None = None,
    cv_iterations: int = 1,
    folds_manifest: str | None = None,
    fold_selection: list[tuple[int, int]] | None = None,
):
    """
    Binary (pairwise) with repeated CV at the COMPOUND level.
    - Build or load a shared compound-level fold MANIFEST (cv_iterations × cv_splits).
    - For each (cv_iter, fold_idx), train seed_splits replicas.
    - Leakage-free scaling per fold; dumps include cv_iter in filenames/columns.
    """
    y_ser = y if isinstance(y, pd.Series) else pd.Series(y)
    groups_ser = groups if isinstance(groups, pd.Series) else pd.Series(groups)

    # Prepare dumping
    if predictions_dir is not None:
        os.makedirs(predictions_dir, exist_ok=True)
        meta_path = os.path.join(predictions_dir, "meta.json")
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "run_name": run_name,
                        "task": "binary_pairwise",
                        "model_type": model_type,
                        "cv_splits": cv_splits,
                        "seed_splits": seed_splits,
                        "base_seed": base_seed,
                        "cv_iterations": cv_iterations,
                        "folds_manifest": folds_manifest,
                        "fold_selection": fold_selection,
                        "bounded_pilot": fold_selection is not None,
                    },
                    f,
                    indent=2,
                )

    # Build or load compound-level folds
    if (true_moa_by_sample is not None) and (compound_by_sample is not None):
        folds = build_or_load_compound_folds(
            true_moa_by_sample=true_moa_by_sample,
            compound_by_sample=compound_by_sample,
            n_splits=cv_splits,
            n_iters=cv_iterations,
            shuffle=shuffle,
            base_seed=base_seed,
            manifest_path=folds_manifest,
        )
    else:
        raise ValueError(
            "For leakage-free CV you must pass true_moa_by_sample and compound_by_sample."
        )

    f1s, rocs, prs = [], [], []
    per_fold_outputs = []

    def _ix(obj, mask):
        return obj.iloc[mask] if hasattr(obj, "iloc") else obj[mask]

    selected_folds = None if fold_selection is None else {tuple(pair) for pair in fold_selection}
    available_folds = {(i, f) for i, repeat in enumerate(folds) for f in range(len(repeat))}
    if selected_folds is not None and (not selected_folds or not selected_folds <= available_folds):
        raise ValueError("Pilot fold selection is empty or outside the fixed manifest")
    for cv_iter, iter_folds in enumerate(folds):
        for fold_idx, spec in enumerate(iter_folds):
            if selected_folds is not None and (cv_iter, fold_idx) not in selected_folds:
                continue
            tr_mask = groups_ser.astype(str).isin(spec.train_compounds).to_numpy()
            te_mask = groups_ser.astype(str).isin(spec.test_compounds).to_numpy()

            scaler = StandardScaler()
            X_tr_raw = _ix(X, tr_mask)
            X_te_raw = _ix(X, te_mask)
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)

            y_tr = _ix(y_ser, tr_mask)
            y_te = _ix(y_ser, te_mask)
            grp_te = _ix(groups_ser, te_mask)
            samp_te = _ix(sample_ids, te_mask) if sample_ids is not None else None
            cand_te = _ix(candidate_moas, te_mask) if candidate_moas is not None else None

            for rep_idx in range(seed_splits):
                seed = int(base_seed + rep_idx)
                clf = get_model(model_type, input_dim=X_tr.shape[1], n_classes=2, seed=seed)
                clf.fit(X_tr, y_tr)

                y_pred = clf.predict(X_te)
                proba = clf.predict_proba(X_te)[:, 1]

                f1 = f1_score(y_te, y_pred)
                try:
                    roc = roc_auc_score(y_te, proba)
                    pr = average_precision_score(y_te, proba)
                except ValueError:
                    roc, pr = np.nan, np.nan

                f1s.append(float(f1))
                rocs.append(float(roc))
                prs.append(float(pr))

                if hasattr(X, "index"):
                    row_ids = np.asarray(X.index)[te_mask]
                else:
                    row_ids = np.arange(len(y_ser))[te_mask]

                df_out = pd.DataFrame(
                    {
                        "RowID": row_ids.astype(str),
                        "CompoundID": grp_te.values.astype(str),
                        "True": np.asarray(y_te, dtype=int),
                        "Pred": y_pred.astype(int),
                        "ProbaPos": proba.astype(float),
                        "cv_iter": cv_iter,
                        "fold_idx": fold_idx,
                        "rep_idx": rep_idx,
                        "seed": seed,
                    }
                ).set_index("RowID")

                if samp_te is not None:
                    df_out["SampleID"] = samp_te.values.astype(str)
                if cand_te is not None:
                    df_out["CandidateMoA"] = cand_te.values.astype(str)

                per_fold_outputs.append(df_out)

                if predictions_dir is not None:
                    out_file = os.path.join(
                        predictions_dir,
                        f"fold{fold_idx:02d}_iter{cv_iter:02d}_rep{rep_idx:02d}_seed{seed}.pkl",
                    )
                    df_out.to_pickle(out_file)

    return {
        "f1_scores": f1s,
        "roc_auc_scores": rocs,
        "pr_auc_scores": prs,
        "avg_f1": float(np.mean(f1s)) if f1s else np.nan,
        "avg_roc_auc": float(np.nanmean(rocs)) if rocs else np.nan,
        "avg_pr_auc": float(np.nanmean(prs)) if prs else np.nan,
        "per_fold_outputs": per_fold_outputs,
    }


def log_binary_metrics(results):
    mlflow.log_metric("avg_f1", results["avg_f1"])
    mlflow.log_metric("avg_roc_auc", results["avg_roc_auc"])
    mlflow.log_metric("avg_pr_auc", results["avg_pr_auc"])


def log_multiclass_metrics(results):
    mlflow.log_metric("avg_top1_acc", results["avg_top1_acc"])
    mlflow.log_metric("avg_topk_acc", results["avg_topk_acc"])
    mlflow.log_metric("avg_macro_f1", results["avg_macro_f1"])
    mlflow.log_metric("avg_micro_f1", results["avg_micro_f1"])
    mlflow.log_metric("avg_macro_roc_auc_ovr", results["avg_macro_roc_auc_ovr"])
    mlflow.log_metric("avg_macro_ap_ovr", results["avg_macro_ap_ovr"])


def evaluate_model_LOO(
    model_type: str,
    X,
    y,
    groups,
    seed_repeats: int = 5,
    base_seed: int = 42,
    predictions_dir: str | None = None,
    run_name: str | None = None,
    sample_ids: pd.Series | None = None,
    candidate_moas: pd.Series | None = None,
    return_experts: bool = False,
    mc_passes: int = 0,  # <--- NEW
    test_compounds: list[str] | None = None,
):
    """
    Leave-One-Compound-Out (LOO) for binary pairwise.
    ...
    If `return_experts` is True and the model supports it (TabMWrapper), also dump a
    compressed npz with per-expert probs for the test rows:
       - without MC: experts shape = (n_test, k, 2)
       - with MC:    experts shape = (n_test, mc_passes, k, 2)
    """

    # Normalize containers to Series/ndarray like your evaluate_model
    y_ser = y if isinstance(y, pd.Series) else pd.Series(y)
    groups_ser = groups if isinstance(groups, pd.Series) else pd.Series(groups)

    if predictions_dir is not None:
        os.makedirs(predictions_dir, exist_ok=True)
        meta_path = os.path.join(predictions_dir, "meta.json")
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "run_name": run_name,
                        "task": "binary_pairwise_LOO",
                        "test_compounds": test_compounds,
                        "bounded_pilot": test_compounds is not None,
                        "model_type": model_type,
                        "seed_repeats": seed_repeats,
                        "base_seed": base_seed,
                        "return_experts": bool(return_experts),
                        "mc_passes": int(mc_passes),  # <--- NEW
                    },
                    f,
                    indent=2,
                )

    # Unique compounds (stable order)
    unique_compounds = pd.Index(groups_ser.astype(str)).unique().tolist()

    f1s, rocs, prs = [], [], []
    per_fold_outputs = []

    def _ix(obj, mask):
        return obj.iloc[mask] if hasattr(obj, "iloc") else obj[mask]

    if test_compounds is not None and (
        not test_compounds or not set(test_compounds) <= set(unique_compounds)
    ):
        raise ValueError("Pilot compounds are empty or outside the input library")
    for fold_idx, test_cmp in enumerate(unique_compounds):
        if test_compounds is not None and test_cmp not in test_compounds:
            continue
        te_mask = (groups_ser.astype(str) == test_cmp).to_numpy()
        tr_mask = ~te_mask

        # Guard: skip degenerate train with one class
        if np.unique(y_ser[tr_mask]).size < 2:
            continue

        # Leakage-free scaling
        scaler = StandardScaler()
        X_tr_raw = _ix(X, tr_mask)
        X_te_raw = _ix(X, te_mask)
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        y_tr = _ix(y_ser, tr_mask)
        y_te = _ix(y_ser, te_mask)
        grp_te = _ix(groups_ser, te_mask)

        # Optional context
        samp_te = _ix(sample_ids, te_mask) if sample_ids is not None else None
        cand_te = _ix(candidate_moas, te_mask) if candidate_moas is not None else None

        for rep_idx in range(seed_repeats):
            seed = int(base_seed + rep_idx)

            clf = get_model(model_type, input_dim=X_tr.shape[1], n_classes=2, seed=seed)
            clf.fit(X_tr, y_tr)

            experts_np = None
            if return_experts:
                proba_out = None
                try:
                    # TabMWrapper supports mc_passes
                    mean_arr, exps_arr = clf.predict_proba(
                        X_te, return_experts=True, mc_passes=mc_passes
                    )  # <--- NEW
                    proba_out = mean_arr
                    experts_np = exps_arr  # (N_te, k, 2) or (N_te, T, k, 2)
                except TypeError:
                    # Model doesn't accept mc_passes/return_experts; fall back
                    proba_out = clf.predict_proba(X_te)
                except Exception:
                    proba_out = clf.predict_proba(X_te)
                proba = proba_out[:, 1]
            else:
                proba = clf.predict_proba(X_te)[:, 1]

            y_pred = (proba >= 0.5).astype(int)

            # Metrics (AUC/AP may be NaN if test has one class)
            f1 = f1_score(y_te, y_pred)
            try:
                roc = roc_auc_score(y_te, proba)
                pr = average_precision_score(y_te, proba)
            except ValueError:
                roc, pr = np.nan, np.nan

            f1s.append(float(f1))
            rocs.append(float(roc))
            prs.append(float(pr))

            # Build per-(fold,seed) DF (same schema as evaluate_model)
            if hasattr(X, "index"):
                row_ids = np.asarray(X.index)[te_mask]
            else:
                row_ids = np.arange(len(y_ser))[te_mask]

            df_out = pd.DataFrame(
                {
                    "RowID": row_ids.astype(str),
                    "CompoundID": grp_te.values.astype(str),
                    "True": np.asarray(y_te, dtype=int),
                    "Pred": y_pred.astype(int),
                    "ProbaPos": proba.astype(float),
                    "fold_idx": fold_idx,
                    "rep_idx": rep_idx,
                    "seed": seed,
                }
            ).set_index("RowID")

            if samp_te is not None:
                df_out["SampleID"] = samp_te.values.astype(str)
            if cand_te is not None:
                df_out["CandidateMoA"] = cand_te.values.astype(str)

            per_fold_outputs.append(df_out)

            # Dump predictions + (optionally) experts
            if predictions_dir is not None:
                out_file = os.path.join(
                    predictions_dir, f"fold{fold_idx:02d}_rep{rep_idx:02d}_seed{seed}.pkl"
                )
                df_out.to_pickle(out_file)

                if return_experts and (experts_np is not None):
                    np.savez_compressed(
                        os.path.join(
                            predictions_dir,
                            f"fold{fold_idx:02d}_rep{rep_idx:02d}_seed{seed}_experts.npz",
                        ),
                        experts=experts_np,  # shape (N,k,2) OR (N,T,k,2)
                        row_ids=df_out.index.values.astype(str),
                        compound_ids=df_out["CompoundID"].values.astype(str),
                        mc_passes=int(mc_passes),  # <--- NEW
                    )

    return {
        "f1_scores": f1s,
        "roc_auc_scores": rocs,
        "pr_auc_scores": prs,
        "avg_f1": float(np.mean(f1s)) if f1s else np.nan,
        "avg_roc_auc": float(np.nanmean(rocs)) if rocs else np.nan,
        "avg_pr_auc": float(np.nanmean(prs)) if prs else np.nan,
        "per_fold_outputs": per_fold_outputs,
    }


def evaluate_model_multiclass_pairwise(
    model_type: str,
    X,
    y,
    groups,
    sample_ids,
    candidate_moas,
    true_moa_by_sample: pd.Series,
    compound_by_sample: pd.Series,
    cv_splits: int = 5,
    seed_splits: int = 5,
    base_seed: int = 42,
    shuffle: bool = True,
    random_state: int = 42,
    topk: int = 3,
    predictions_dir: str | None = None,
    run_name: str | None = None,
    store_pairwise: bool = False,
    cv_iterations: int = 1,
    folds_manifest: str | None = None,
):
    """
    Pairwise training; evaluate multiclass after aggregating scores per sample.
    Repeated CV at COMPOUND level via shared fold manifest.
    """
    y_ser = y if isinstance(y, pd.Series) else pd.Series(y)
    groups_ser = groups if isinstance(groups, pd.Series) else pd.Series(groups)
    sample_ser = sample_ids if isinstance(sample_ids, pd.Series) else pd.Series(sample_ids)
    cand_ser = (
        candidate_moas if isinstance(candidate_moas, pd.Series) else pd.Series(candidate_moas)
    )

    if predictions_dir is not None:
        os.makedirs(predictions_dir, exist_ok=True)
        meta_path = os.path.join(predictions_dir, "meta.json")
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "run_name": run_name,
                        "task": "pairwise_to_multiclass",
                        "model_type": model_type,
                        "cv_splits": cv_splits,
                        "seed_splits": seed_splits,
                        "base_seed": base_seed,
                        "cv_iterations": cv_iterations,
                        "folds_manifest": folds_manifest,
                    },
                    f,
                    indent=2,
                )

    folds = build_or_load_compound_folds(
        true_moa_by_sample=true_moa_by_sample,
        compound_by_sample=compound_by_sample,
        n_splits=cv_splits,
        n_iters=cv_iterations,
        shuffle=shuffle,
        base_seed=base_seed,
        manifest_path=folds_manifest,
    )
    classes_global = np.unique(true_moa_by_sample.values)

    top1_accs, topk_accs = [], []
    macro_f1s, micro_f1s = [], []
    macro_roc_ovrs, macro_ap_ovrs = [], []
    per_fold_outputs = []

    def _ix(obj, mask):
        return obj.iloc[mask] if hasattr(obj, "iloc") else obj[mask]

    for cv_iter, iter_folds in enumerate(folds):
        for fold_idx, spec in enumerate(iter_folds):
            tr_mask = groups_ser.astype(str).isin(spec.train_compounds).to_numpy()
            te_mask = groups_ser.astype(str).isin(spec.test_compounds).to_numpy()

            scaler = StandardScaler()
            X_tr_raw = _ix(X, tr_mask)
            X_te_raw = _ix(X, te_mask)
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)

            y_tr = _ix(y_ser, tr_mask)
            samp_te = _ix(sample_ser, te_mask)
            cand_te = _ix(cand_ser, te_mask)

            for rep_idx in range(seed_splits):
                seed = int(base_seed + rep_idx)
                clf = get_model(model_type, input_dim=X_tr.shape[1], n_classes=2, seed=seed)
                clf.fit(X_tr, y_tr)

                pair_proba_pos = clf.predict_proba(X_te)[:, 1]
                df_scores = pd.DataFrame(
                    {
                        "SampleID": samp_te.values,
                        "CandidateMoA": cand_te.values,
                        "Score": pair_proba_pos,
                    }
                )
                pivot_scores = df_scores.pivot_table(
                    index="SampleID", columns="CandidateMoA", values="Score", aggfunc="mean"
                ).reindex(columns=classes_global, fill_value=-np.inf)

                test_samples = pivot_scores.index
                y_true = true_moa_by_sample.loc[test_samples].values

                cols = pivot_scores.columns.to_numpy()
                mat = pivot_scores.to_numpy()
                pred_idx = np.argmax(mat, axis=1)
                y_pred = cols[pred_idx]

                top1 = float(np.mean(y_pred == y_true))
                k = min(topk, len(cols))
                order = np.argsort(-mat, axis=1)
                hits = [(y_true[i] in cols[order[i, :k]]) for i in range(len(y_true))]
                topk_acc = float(np.mean(hits))

                present = np.unique(y_true)
                macro_f1 = f1_score(
                    y_true, y_pred, average="macro", labels=present, zero_division=0
                )
                micro_f1 = f1_score(
                    y_true, y_pred, average="micro", labels=present, zero_division=0
                )

                ovrs_roc, ovrs_ap = [], []
                for c in present:
                    y_bin = (y_true == c).astype(int)
                    s_c = pivot_scores[c].to_numpy()
                    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                        continue
                    try:
                        ovrs_roc.append(roc_auc_score(y_bin, s_c))
                        ovrs_ap.append(average_precision_score(y_bin, s_c))
                    except ValueError:
                        pass
                macro_roc = float(np.mean(ovrs_roc)) if ovrs_roc else np.nan
                macro_ap = float(np.mean(ovrs_ap)) if ovrs_ap else np.nan

                top1_accs.append(top1)
                topk_accs.append(topk_acc)
                macro_f1s.append(float(macro_f1))
                micro_f1s.append(float(micro_f1))
                macro_roc_ovrs.append(macro_roc)
                macro_ap_ovrs.append(macro_ap)

                agg_df = pivot_scores.copy()
                agg_df["TrueMoA"] = y_true.astype(str)
                agg_df["PredMoA"] = y_pred.astype(str)
                agg_df["cv_iter"] = cv_iter
                agg_df["fold_idx"] = fold_idx
                agg_df["rep_idx"] = rep_idx
                agg_df["seed"] = seed
                agg_df.index.name = "SampleID"
                per_fold_outputs.append(agg_df)

                if predictions_dir is not None:
                    agg_path = os.path.join(
                        predictions_dir,
                        f"fold{fold_idx:02d}_iter{cv_iter:02d}_rep{rep_idx:02d}_seed{seed}_agg.pkl",
                    )
                    agg_df.to_pickle(agg_path)

                    if store_pairwise:
                        pair_df = df_scores.copy()
                        pair_df["cv_iter"] = cv_iter
                        pair_df["fold_idx"] = fold_idx
                        pair_df["rep_idx"] = rep_idx
                        pair_df["seed"] = seed
                        pair_df.to_pickle(
                            os.path.join(
                                predictions_dir,
                                f"fold{fold_idx:02d}_iter{cv_iter:02d}_rep{rep_idx:02d}_seed{seed}_pairwise.pkl",
                            )
                        )

    return {
        "top1_accs": top1_accs,
        "topk_accs": topk_accs,
        "macro_f1s": macro_f1s,
        "micro_f1s": micro_f1s,
        "macro_roc_aucs_ovr": macro_roc_ovrs,
        "macro_aps_ovr": macro_ap_ovrs,
        "avg_top1_acc": float(np.mean(top1_accs)) if top1_accs else np.nan,
        "avg_topk_acc": float(np.mean(topk_accs)) if topk_accs else np.nan,
        "avg_macro_f1": float(np.mean(macro_f1s)) if macro_f1s else np.nan,
        "avg_micro_f1": float(np.mean(micro_f1s)) if micro_f1s else np.nan,
        "avg_macro_roc_auc_ovr": float(np.nanmean(macro_roc_ovrs)) if macro_roc_ovrs else np.nan,
        "avg_macro_ap_ovr": float(np.nanmean(macro_ap_ovrs)) if macro_ap_ovrs else np.nan,
        "per_fold_outputs": per_fold_outputs,
        "classes_": list(classes_global),
    }


def make_compound_kfolds_simple(
    true_moa_by_sample: pd.Series,
    compound_by_sample: pd.Series,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
):
    """
    Strict, simple splitter:
    - Builds StratifiedKFold at the *compound* level (one label per compound = MoA).
    - NO fallbacks. If a class has < n_splits compounds, sklearn will raise.
    - Returns a list of dicts with {'train_compounds', 'test_compounds'}.
    """
    df = pd.DataFrame({"MoA": true_moa_by_sample, "CompoundID": compound_by_sample})

    # Sanity: exactly one MoA per compound
    per_cmp = df.groupby("CompoundID")["MoA"].nunique()
    bad = per_cmp[per_cmp > 1]
    if len(bad):
        raise ValueError(
            "Some compounds map to multiple MoAs; cannot stratify. "
            f"Examples: {list(bad.index)[:5]}"
        )

    cmp_to_moa = df.groupby("CompoundID")["MoA"].first()
    compounds = cmp_to_moa.index.to_numpy()
    y_cmp = cmp_to_moa.values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    folds = []
    for tr_idx, te_idx in skf.split(compounds, y_cmp):
        folds.append(
            {
                "train_compounds": set(compounds[tr_idx]),
                "test_compounds": set(compounds[te_idx]),
            }
        )
    return folds


def evaluate_model_multiclass_samples(
    model_type: str,
    X,
    y_labels: pd.Series,
    compound_by_sample: pd.Series,
    cv_splits: int = 5,
    seed_splits: int = 5,
    base_seed: int = 42,
    shuffle: bool = True,
    random_state: int = 42,
    topk: int = 3,
    predictions_dir: str | None = None,
    run_name: str | None = None,
    cv_iterations: int = 1,
    folds_manifest: str | None = None,
):
    """
    True multiclass (no text) with repeated CV at the COMPOUND level via a shared fold manifest.
    """
    if not isinstance(y_labels, pd.Series):
        y_labels = pd.Series(y_labels, index=getattr(X, "index", None))

    df_map = pd.DataFrame({"MoA": y_labels, "CompoundID": compound_by_sample})
    per_cmp = df_map.groupby("CompoundID")["MoA"].nunique()
    bad = per_cmp[per_cmp > 1]
    if len(bad):
        raise ValueError(f"Compounds with multiple MoAs found: {list(bad.index)[:5]} ...")

    cmp_to_moa = df_map.groupby("CompoundID")["MoA"].first()
    classes_global = np.unique(y_labels.values)
    le = LabelEncoder().fit(y_labels.values)

    if predictions_dir is not None:
        os.makedirs(predictions_dir, exist_ok=True)
        meta_path = os.path.join(predictions_dir, "meta.json")
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "run_name": run_name,
                        "task": "no_text_multiclass",
                        "model_type": model_type,
                        "cv_splits": cv_splits,
                        "seed_splits": seed_splits,
                        "base_seed": base_seed,
                        "cv_iterations": cv_iterations,
                        "folds_manifest": folds_manifest,
                        "classes": list(map(str, le.classes_)),
                    },
                    f,
                    indent=2,
                )

    folds = build_or_load_compound_folds(
        true_moa_by_sample=y_labels,
        compound_by_sample=compound_by_sample,
        n_splits=cv_splits,
        n_iters=cv_iterations,
        shuffle=shuffle,
        base_seed=base_seed,
        manifest_path=folds_manifest,
    )

    top1_accs, topk_accs = [], []
    macro_f1s, micro_f1s = [], []
    macro_roc_ovrs, macro_ap_ovrs = [], []
    per_fold_outputs = []

    def _ix(obj, mask):
        return obj.iloc[mask] if hasattr(obj, "iloc") else obj[mask]

    for cv_iter, iter_folds in enumerate(folds):
        for fold_idx, spec in enumerate(iter_folds):
            tr_mask = compound_by_sample.astype(str).isin(spec.train_compounds).to_numpy()
            te_mask = compound_by_sample.astype(str).isin(spec.test_compounds).to_numpy()

            scaler = StandardScaler()
            X_tr_raw = _ix(X, tr_mask)
            X_te_raw = _ix(X, te_mask)
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)

            y_tr_lbl = y_labels[tr_mask]
            y_te_lbl = y_labels[te_mask]
            y_tr_int = le.transform(y_tr_lbl.values)
            y_te_int = le.transform(y_te_lbl.values)

            n_classes = len(classes_global)

            for rep_idx in range(seed_splits):
                replicate_seed = int(base_seed + rep_idx)
                clf = get_model(
                    model_type,
                    input_dim=X_tr.shape[1],
                    n_classes=n_classes,
                    seed=replicate_seed,
                )
                clf.fit(X_tr, y_tr_int)

                proba = clf.predict_proba(X_te)
                y_pred_int = np.argmax(proba, axis=1)

                top1 = accuracy_score(y_te_int, y_pred_int)
                k = min(topk, proba.shape[1])
                order = np.argsort(-proba, axis=1)
                hits = [y_te_int[i] in order[i, :k] for i in range(len(y_te_int))]
                topk_acc = float(np.mean(hits))

                present = np.unique(y_te_int)
                macro_f1 = f1_score(
                    y_te_int, y_pred_int, average="macro", labels=present, zero_division=0
                )
                micro_f1 = f1_score(
                    y_te_int, y_pred_int, average="micro", labels=present, zero_division=0
                )

                ovrs_roc, ovrs_ap = [], []
                for c in present:
                    y_bin = (y_te_int == c).astype(int)
                    s_c = proba[:, c]
                    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                        continue
                    try:
                        ovrs_roc.append(roc_auc_score(y_bin, s_c))
                        ovrs_ap.append(average_precision_score(y_bin, s_c))
                    except ValueError:
                        pass
                macro_roc = float(np.mean(ovrs_roc)) if ovrs_roc else np.nan
                macro_ap = float(np.mean(ovrs_ap)) if ovrs_ap else np.nan

                top1_accs.append(float(top1))
                topk_accs.append(float(topk_acc))
                macro_f1s.append(float(macro_f1))
                micro_f1s.append(float(micro_f1))
                macro_roc_ovrs.append(macro_roc)
                macro_ap_ovrs.append(macro_ap)

                fold_df = pd.DataFrame(
                    {
                        "SampleID": y_labels[te_mask].index.astype(str),
                        "TrueInt": y_te_int,
                        "PredInt": y_pred_int,
                        "TrueLbl": y_te_lbl.values.astype(str),
                        "PredLbl": np.asarray(classes_global)[y_pred_int].astype(str),
                        "cv_iter": cv_iter,
                        "fold_idx": fold_idx,
                        "rep_idx": rep_idx,
                        "seed": replicate_seed,
                    }
                ).set_index("SampleID")

                for i, cls in enumerate(classes_global):
                    fold_df[f"proba_{cls}"] = proba[:, i] if proba.shape[1] > i else np.nan
                per_fold_outputs.append(fold_df)

                if predictions_dir is not None:
                    out_file = os.path.join(
                        predictions_dir,
                        f"fold{fold_idx:02d}_iter{cv_iter:02d}_rep{rep_idx:02d}_seed{replicate_seed}.pkl",
                    )
                    fold_df.to_pickle(out_file)

    return {
        "top1_accs": top1_accs,
        "topk_accs": topk_accs,
        "macro_f1s": macro_f1s,
        "micro_f1s": micro_f1s,
        "macro_roc_aucs_ovr": macro_roc_ovrs,
        "macro_aps_ovr": macro_ap_ovrs,
        "avg_top1_acc": float(np.mean(top1_accs)) if top1_accs else np.nan,
        "avg_topk_acc": float(np.mean(topk_accs)) if topk_accs else np.nan,
        "avg_macro_f1": float(np.mean(macro_f1s)) if macro_f1s else np.nan,
        "avg_micro_f1": float(np.mean(micro_f1s)) if micro_f1s else np.nan,
        "avg_macro_roc_auc_ovr": float(np.nanmean(macro_roc_ovrs)) if macro_roc_ovrs else np.nan,
        "avg_macro_ap_ovr": float(np.nanmean(macro_ap_ovrs)) if macro_ap_ovrs else np.nan,
        "per_fold_outputs": per_fold_outputs,
        "classes_": list(classes_global),
    }


def make_compound_loco_splits(
    true_moa_by_sample: pd.Series,
    compound_by_sample: pd.Series,
    require_test_class_in_train: bool = True,
    min_train_classes: int = 2,
) -> list[dict]:
    """
    Build leave-one-compound-out splits at the COMPOUND level.
    Returns a list of dicts: {'train_compounds': set, 'test_compounds': set}
    Skips folds where training would have < min_train_classes classes or
    (optionally) where the test compound's class is absent from training.
    """
    df = pd.DataFrame({"MoA": true_moa_by_sample, "CompoundID": compound_by_sample})

    # Ensure one MoA per compound
    per_cmp = df.groupby("CompoundID")["MoA"].nunique()
    bad = per_cmp[per_cmp > 1]
    if len(bad):
        raise ValueError(
            "Some compounds map to multiple MoAs; cannot do LOCO.\n"
            f"Examples: {list(bad.index)[:5]}"
        )

    cmp_to_moa = df.groupby("CompoundID")["MoA"].first()
    all_compounds = cmp_to_moa.index.to_numpy()

    folds = []
    for test_cmp in all_compounds:
        train_cmps = set(all_compounds) - {test_cmp}
        test_cmps = {test_cmp}

        train_classes = set(cmp_to_moa.loc[list(train_cmps)])
        test_class = cmp_to_moa.loc[test_cmp]

        # guard rails
        if len(train_classes) < min_train_classes:
            continue
        if require_test_class_in_train and (test_class not in train_classes):
            continue

        folds.append({"train_compounds": train_cmps, "test_compounds": test_cmps})
    return folds


def evaluate_model_binary_loco(
    model_type: str,
    X,
    y,
    groups,
    true_moa_by_sample: pd.Series,
    compound_by_sample: pd.Series,
    shuffle: bool = True,
    random_state: int = 42,
):
    """
    Binary (pairwise) evaluation with COMPOUND LOCO.
    """
    y_ser = y if isinstance(y, pd.Series) else pd.Series(y)
    groups_ser = groups if isinstance(groups, pd.Series) else pd.Series(groups)

    f1s, rocs, prs = [], [], []
    y_trues, y_preds, y_probas, splits = [], [], [], []

    def _slice_rows(obj, mask_bool):
        if hasattr(obj, "iloc"):
            return obj.iloc[mask_bool]
        return obj[mask_bool]

    loco_folds = make_compound_loco_splits(
        true_moa_by_sample=true_moa_by_sample,
        compound_by_sample=compound_by_sample,
        require_test_class_in_train=True,
        min_train_classes=2,
    )
    print(len(loco_folds), "LOCO folds built.")
    print("Training model: ", model_type)
    for fold_idx, split in tqdm(enumerate(loco_folds)):
        tr_cmp = split["train_compounds"]
        te_cmp = split["test_compounds"]

        tr_mask = groups_ser.isin(tr_cmp).to_numpy()
        te_mask = groups_ser.isin(te_cmp).to_numpy()

        scaler = StandardScaler()
        X_tr_raw = _slice_rows(X, tr_mask)
        X_te_raw = _slice_rows(X, te_mask)
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        y_tr = _slice_rows(y_ser, tr_mask)
        y_te = _slice_rows(y_ser, te_mask)

        rng = np.random.RandomState(1000 + fold_idx)
        perm = rng.permutation(len(X_tr))
        X_tr = X_tr[perm]
        y_tr = y_tr.iloc[perm] if hasattr(y_tr, "iloc") else y_tr[perm]

        clf = get_model(model_type, input_dim=X.shape[1], n_classes=2)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        proba = clf.predict_proba(X_te)[:, 1]

        y_trues.append(np.asarray(y_te))
        y_preds.append(y_pred)
        y_probas.append(proba)
        splits.append((tr_mask, te_mask))

        f1 = f1_score(y_te, y_pred)
        try:
            roc = roc_auc_score(y_te, proba)
            pr = average_precision_score(y_te, proba)
        except ValueError:
            roc, pr = np.nan, np.nan

        f1s.append(f1)
        rocs.append(roc)
        prs.append(pr)

    return {
        "f1_scores": f1s,
        "roc_auc_scores": rocs,
        "pr_auc_scores": prs,
        "avg_f1": float(np.mean(f1s)) if f1s else np.nan,
        "avg_roc_auc": float(np.nanmean(rocs)) if rocs else np.nan,
        "avg_pr_auc": float(np.nanmean(prs)) if prs else np.nan,
        "splits": splits,
        "y_trues": y_trues,
        "y_preds": y_preds,
        "y_probas": y_probas,
    }


def evaluate_model_multiclass_pairwise_loco(
    model_type: str,
    X,
    y,
    groups,
    sample_ids,
    candidate_moas,
    true_moa_by_sample: pd.Series,
    compound_by_sample: pd.Series,
    topk: int = 3,
    shuffle: bool = True,
    random_state: int = 42,
):
    """
    Same as evaluate_model_multiclass_pairwise, but with COMPOUND LOCO.
    """
    y_ser = y if isinstance(y, pd.Series) else pd.Series(y)
    groups_ser = groups if isinstance(groups, pd.Series) else pd.Series(groups)
    sample_ser = sample_ids if isinstance(sample_ids, pd.Series) else pd.Series(sample_ids)
    cand_ser = (
        candidate_moas if isinstance(candidate_moas, pd.Series) else pd.Series(candidate_moas)
    )

    top1_accs, topk_accs, macro_f1s, micro_f1s = [], [], [], []
    macro_roc_aucs_ovr, macro_aps_ovr = [], []
    per_fold_outputs = []
    classes_global = np.unique(true_moa_by_sample.values)

    def _slice_rows(obj, mask_bool):
        if hasattr(obj, "iloc"):
            return obj.iloc[mask_bool]
        return obj[mask_bool]

    loco_folds = make_compound_loco_splits(
        true_moa_by_sample=true_moa_by_sample,
        compound_by_sample=compound_by_sample,
        require_test_class_in_train=True,
        min_train_classes=2,
    )

    for fold_idx, split in enumerate(loco_folds):
        tr_cmp = split["train_compounds"]
        te_cmp = split["test_compounds"]

        tr_mask = groups_ser.isin(tr_cmp).to_numpy()
        te_mask = groups_ser.isin(te_cmp).to_numpy()

        scaler = StandardScaler()
        X_tr_raw = _slice_rows(X, tr_mask)
        X_te_raw = _slice_rows(X, te_mask)
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        y_tr = _slice_rows(y_ser, tr_mask)
        samp_te = _slice_rows(sample_ser, te_mask)
        cand_te = _slice_rows(cand_ser, te_mask)

        rng = np.random.RandomState(2000 + fold_idx)
        perm = rng.permutation(len(X_tr))
        X_tr = X_tr[perm]
        y_tr = y_tr.iloc[perm] if hasattr(y_tr, "iloc") else y_tr[perm]

        clf = get_model(model_type, input_dim=X.shape[1], n_classes=2)
        clf.fit(X_tr, y_tr)

        pair_proba_pos = clf.predict_proba(X_te)[:, 1]

        df_scores = pd.DataFrame(
            {
                "SampleID": samp_te.values,
                "CandidateMoA": cand_te.values,
                "Score": pair_proba_pos,
            }
        )
        pivot_scores = df_scores.pivot_table(
            index="SampleID", columns="CandidateMoA", values="Score", aggfunc="mean"
        ).reindex(columns=classes_global, fill_value=-np.inf)

        test_samples_unique = pivot_scores.index
        y_true_aligned = true_moa_by_sample.loc[test_samples_unique].values

        cols = pivot_scores.columns.to_numpy()
        argmax_idx = np.argmax(pivot_scores.to_numpy(), axis=1)
        y_pred_labels = cols[argmax_idx]

        top1_acc = float(np.mean(y_pred_labels == y_true_aligned))
        k = int(min(topk, len(cols)))
        order = np.argsort(-pivot_scores.to_numpy(), axis=1)
        hits = [(y_true_aligned[i] in cols[order[i, :k]]) for i in range(len(y_true_aligned))]
        topk_acc = float(np.mean(hits))

        present = np.unique(y_true_aligned)
        macro_f1 = f1_score(
            y_true_aligned, y_pred_labels, average="macro", labels=present, zero_division=0
        )
        micro_f1 = f1_score(
            y_true_aligned, y_pred_labels, average="micro", labels=present, zero_division=0
        )

        ovrs_roc, ovrs_ap = [], []
        for c in present:
            y_bin = (y_true_aligned == c).astype(int)
            s_c = pivot_scores[c].to_numpy()
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            try:
                ovrs_roc.append(roc_auc_score(y_bin, s_c))
                ovrs_ap.append(average_precision_score(y_bin, s_c))
            except ValueError:
                pass
        macro_roc = float(np.mean(ovrs_roc)) if ovrs_roc else np.nan
        macro_ap = float(np.mean(ovrs_ap)) if ovrs_ap else np.nan

        top1_accs.append(top1_acc)
        topk_accs.append(topk_acc)
        macro_f1s.append(float(macro_f1))
        micro_f1s.append(float(micro_f1))
        macro_roc_aucs_ovr.append(macro_roc)
        macro_aps_ovr.append(macro_ap)

        out = pivot_scores.copy()
        out["TrueMoA"] = y_true_aligned
        out["PredMoA"] = y_pred_labels
        out.index.name = "SampleID"
        per_fold_outputs.append(out)

    return {
        "top1_accs": top1_accs,
        "topk_accs": topk_accs,
        "macro_f1s": macro_f1s,
        "micro_f1s": micro_f1s,
        "macro_roc_aucs_ovr": macro_roc_aucs_ovr,
        "macro_aps_ovr": macro_aps_ovr,
        "avg_top1_acc": float(np.mean(top1_accs)) if top1_accs else np.nan,
        "avg_topk_acc": float(np.mean(topk_accs)) if topk_accs else np.nan,
        "avg_macro_f1": float(np.mean(macro_f1s)) if macro_f1s else np.nan,
        "avg_micro_f1": float(np.mean(micro_f1s)) if micro_f1s else np.nan,
        "avg_macro_roc_auc_ovr": (
            float(np.nanmean(macro_roc_aucs_ovr)) if macro_roc_aucs_ovr else np.nan
        ),
        "avg_macro_ap_ovr": float(np.nanmean(macro_aps_ovr)) if macro_aps_ovr else np.nan,
        "per_fold_outputs": per_fold_outputs,
        "classes_": list(classes_global),
    }


def evaluate_model_multiclass_samples_loco(
    model_type: str,
    X,  # sample-level features (DF or ndarray), index = SampleID
    y_labels: pd.Series,  # index=SampleID -> class string
    compound_by_sample: pd.Series,  # index=SampleID -> CompoundID
    topk: int = 3,
    shuffle: bool = True,
    random_state: int = 42,
):
    """
    Classic multiclass, COMPOUND LOCO at the sample level.
    """
    if not isinstance(y_labels, pd.Series):
        y_labels = pd.Series(y_labels, index=getattr(X, "index", None))

    df = pd.DataFrame({"MoA": y_labels, "CompoundID": compound_by_sample})
    per_cmp = df.groupby("CompoundID")["MoA"].nunique()
    bad = per_cmp[per_cmp > 1]
    if len(bad):
        raise ValueError(f"Compounds with multiple MoAs found: {list(bad.index)[:5]} ...")

    cmp_to_moa = df.groupby("CompoundID")["MoA"].first()
    classes_global = np.unique(y_labels.values)
    le = LabelEncoder().fit(classes_global)

    def _ix(obj, mask):
        return obj.iloc[mask] if hasattr(obj, "iloc") else obj[mask]

    # build compound list
    all_compounds = cmp_to_moa.index.to_numpy()
    # map sample -> is in test compound
    sample_to_cmp = compound_by_sample

    top1_accs, topk_accs, macro_f1s, micro_f1s, macro_roc_ovrs, macro_ap_ovrs = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    per_fold_outputs = []

    for fold_idx, test_cmp in enumerate(all_compounds):
        train_cmps = set(all_compounds) - {test_cmp}
        test_cmps = {test_cmp}

        train_classes = set(cmp_to_moa.loc[list(train_cmps)])
        test_class = cmp_to_moa.loc[test_cmp]
        if len(train_classes) < 2 or (test_class not in train_classes):
            # skip degenerate folds
            continue

        tr_mask = sample_to_cmp.isin(train_cmps).to_numpy()
        te_mask = sample_to_cmp.isin(test_cmps).to_numpy()

        scaler = StandardScaler()
        X_tr_raw = _ix(X, tr_mask)
        X_te_raw = _ix(X, te_mask)
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        y_tr_lbl = y_labels[tr_mask]
        y_te_lbl = y_labels[te_mask]
        y_tr_int = le.transform(y_tr_lbl.values)
        y_te_int = le.transform(y_te_lbl.values)

        clf = get_model(model_type, input_dim=X_tr.shape[1], n_classes=len(classes_global))
        clf.fit(X_tr, y_tr_int)
        proba = clf.predict_proba(X_te)  # (N_te, C)
        y_pred_int = np.argmax(proba, axis=1)

        # metrics
        top1 = accuracy_score(y_te_int, y_pred_int)
        k = min(topk, proba.shape[1])
        order = np.argsort(-proba, axis=1)
        hits = [y_te_int[i] in order[i, :k] for i in range(len(y_te_int))]
        topk_acc = float(np.mean(hits)) if len(hits) else np.nan

        present = np.unique(y_te_int)
        macro_f1 = f1_score(y_te_int, y_pred_int, average="macro", labels=present, zero_division=0)
        micro_f1 = f1_score(y_te_int, y_pred_int, average="micro", labels=present, zero_division=0)

        ovrs_roc, ovrs_ap = [], []
        for c in present:
            y_bin = (y_te_int == c).astype(int)
            s_c = proba[:, c]
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            try:
                ovrs_roc.append(roc_auc_score(y_bin, s_c))
                ovrs_ap.append(average_precision_score(y_bin, s_c))
            except ValueError:
                pass
        macro_roc = float(np.mean(ovrs_roc)) if ovrs_roc else np.nan
        macro_ap = float(np.mean(ovrs_ap)) if ovrs_ap else np.nan

        top1_accs.append(float(top1))
        topk_accs.append(float(topk_acc))
        macro_f1s.append(float(macro_f1))
        micro_f1s.append(float(micro_f1))
        macro_roc_ovrs.append(macro_roc)
        macro_ap_ovrs.append(macro_ap)

        fold_df = pd.DataFrame(
            {
                "SampleID": y_labels[te_mask].index,
                "TrueInt": y_te_int,
                "PredInt": y_pred_int,
            }
        ).set_index("SampleID")
        for i, cls in enumerate(classes_global):
            fold_df[f"proba_{cls}"] = proba[:, i] if proba.shape[1] > i else np.nan
        per_fold_outputs.append(fold_df)

    return {
        "top1_accs": top1_accs,
        "topk_accs": topk_accs,
        "macro_f1s": macro_f1s,
        "micro_f1s": micro_f1s,
        "macro_roc_aucs_ovr": macro_roc_ovrs,
        "macro_aps_ovr": macro_ap_ovrs,
        "avg_top1_acc": float(np.mean(top1_accs)) if top1_accs else np.nan,
        "avg_topk_acc": float(np.mean(topk_accs)) if topk_accs else np.nan,
        "avg_macro_f1": float(np.mean(macro_f1s)) if macro_f1s else np.nan,
        "avg_micro_f1": float(np.mean(micro_f1s)) if micro_f1s else np.nan,
        "avg_macro_roc_auc_ovr": float(np.nanmean(macro_roc_ovrs)) if macro_roc_ovrs else np.nan,
        "avg_macro_ap_ovr": float(np.nanmean(macro_ap_ovrs)) if macro_ap_ovrs else np.nan,
        "per_fold_outputs": per_fold_outputs,
        "classes_": list(classes_global),
    }


def make_class_loso_splits(
    true_moa_by_sample: pd.Series,  # index=SampleID -> MoA
    compound_by_sample: pd.Series,  # index=SampleID -> CompoundID
    min_train_classes: int = 2,
) -> list[dict]:
    """
    Build leave-one-class-out splits at the COMPOUND level.
    For each MoA class C:
      - train = all compounds whose MoA != C
      - test  = all compounds whose MoA == C
    Skips folds where training would have < min_train_classes classes.
    """
    df = pd.DataFrame({"MoA": true_moa_by_sample, "CompoundID": compound_by_sample})
    # ensure one MoA per compound
    per_cmp = df.groupby("CompoundID")["MoA"].nunique()
    bad = per_cmp[per_cmp > 1]
    if len(bad):
        raise ValueError(
            "Some compounds map to multiple MoAs; cannot do LOSO.\n"
            f"Examples: {list(bad.index)[:5]}"
        )

    cmp_to_moa = df.groupby("CompoundID")["MoA"].first()  # CompoundID -> MoA
    classes = sorted(cmp_to_moa.unique())
    folds = []
    for C in classes:
        test_compounds = set(cmp_to_moa[cmp_to_moa == C].index.astype(str))
        train_compounds = set(cmp_to_moa[cmp_to_moa != C].index.astype(str))
        train_classes = set(cmp_to_moa.loc[list(train_compounds)])
        if len(train_classes) < min_train_classes:
            continue
        folds.append(
            {
                "heldout_class": str(C),
                "train_compounds": train_compounds,
                "test_compounds": test_compounds,
            }
        )
    return folds


def evaluate_model_LOSO(
    model_type: str,
    X,
    y,
    groups,
    sample_ids: pd.Series,
    candidate_moas: pd.Series,
    true_moa_by_sample: pd.Series,
    seed_repeats: int = 5,
    base_seed: int = 42,
    predictions_dir: str | None = None,
    run_name: str | None = None,
    return_experts: bool = False,
    mc_passes: int = 0,
    drop_heldout_candidate_in_train: bool = False,  # set True for stricter zero-shot
):
    """
    Leave-One-Class-Out for binary pairwise.
    Splits are at the compound level, grouped by MoA.
    If `drop_heldout_candidate_in_train=True`, remove all TRAIN rows whose CandidateMoA == heldout class.
    """
    y_ser = y if isinstance(y, pd.Series) else pd.Series(y)
    groups_ser = groups if isinstance(groups, pd.Series) else pd.Series(groups)
    samp_ser = sample_ids if isinstance(sample_ids, pd.Series) else pd.Series(sample_ids)
    cand_ser = (
        candidate_moas if isinstance(candidate_moas, pd.Series) else pd.Series(candidate_moas)
    )

    if predictions_dir is not None:
        os.makedirs(predictions_dir, exist_ok=True)
        meta_path = os.path.join(predictions_dir, "meta.json")
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "run_name": run_name,
                        "task": "binary_pairwise_LOSO",
                        "model_type": model_type,
                        "seed_repeats": seed_repeats,
                        "base_seed": base_seed,
                        "return_experts": bool(return_experts),
                        "mc_passes": int(mc_passes),
                        "drop_heldout_candidate_in_train": bool(drop_heldout_candidate_in_train),
                    },
                    f,
                    indent=2,
                )

    # Build LOSO folds at the compound level
    cmp_by_sample = true_moa_by_sample.index.to_series().map(lambda s: s.split("_")[0])

    folds = make_class_loso_splits(
        true_moa_by_sample=true_moa_by_sample.astype(str),
        compound_by_sample=cmp_by_sample.astype(str),
        min_train_classes=2,
    )

    # (optional sanity print)
    print(
        f"[LOSO] Built {len(folds)} folds; held-out MoAs (first 10):",
        [f["heldout_class"] for f in folds][:10],
    )

    f1s, rocs, prs = [], [], []
    per_fold_outputs = []

    def _ix(obj, mask):
        return obj.iloc[mask] if hasattr(obj, "iloc") else obj[mask]

    for fold_idx, spec in enumerate(folds):
        heldout = spec["heldout_class"]
        tr_cmp = spec["train_compounds"]
        te_cmp = spec["test_compounds"]

        tr_mask = groups_ser.astype(str).isin(tr_cmp).to_numpy()
        te_mask = groups_ser.astype(str).isin(te_cmp).to_numpy()

        # Optional: drop TRAIN rows that use the held-out candidate MoA
        if drop_heldout_candidate_in_train:
            tr_mask = tr_mask & (cand_ser.astype(str) != heldout).to_numpy()

        scaler = StandardScaler()
        X_tr_raw = _ix(X, tr_mask)
        X_te_raw = _ix(X, te_mask)
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        y_tr = _ix(y_ser, tr_mask)
        y_te = _ix(y_ser, te_mask)
        grp_te = _ix(groups_ser, te_mask)
        samp_te = _ix(samp_ser, te_mask)
        cand_te = _ix(cand_ser, te_mask)

        for rep_idx in range(seed_repeats):
            seed = int(base_seed + rep_idx)
            clf = get_model(model_type, input_dim=X_tr.shape[1], n_classes=2, seed=seed)
            clf.fit(X_tr, y_tr)

            experts_np = None
            if return_experts:
                try:
                    mean_arr, exps_arr = clf.predict_proba(
                        X_te, return_experts=True, mc_passes=mc_passes
                    )
                    proba = mean_arr[:, 1]
                    experts_np = exps_arr  # (N,k,2) or (N,T,k,2)
                except Exception:
                    proba = clf.predict_proba(X_te)[:, 1]
            else:
                proba = clf.predict_proba(X_te)[:, 1]

            y_pred = (proba >= 0.5).astype(int)
            f1 = f1_score(y_te, y_pred)
            try:
                roc = roc_auc_score(y_te, proba)
                pr = average_precision_score(y_te, proba)
            except ValueError:
                roc, pr = np.nan, np.nan
            f1s.append(float(f1))
            rocs.append(float(roc))
            prs.append(float(pr))

            row_ids = (
                np.asarray(X.index)[te_mask]
                if hasattr(X, "index")
                else np.arange(len(y_ser))[te_mask]
            )
            df_out = pd.DataFrame(
                {
                    "RowID": row_ids.astype(str),
                    "CompoundID": grp_te.values.astype(str),
                    "True": np.asarray(y_te, dtype=int),
                    "Pred": y_pred.astype(int),
                    "ProbaPos": proba.astype(float),
                    "fold_idx": fold_idx,
                    "rep_idx": rep_idx,
                    "seed": seed,
                    "heldout_class": heldout,
                }
            ).set_index("RowID")
            df_out["SampleID"] = samp_te.values.astype(str)
            df_out["CandidateMoA"] = cand_te.values.astype(str)
            per_fold_outputs.append(df_out)

            if predictions_dir is not None:
                out_file = os.path.join(
                    predictions_dir, f"fold{fold_idx:02d}_rep{rep_idx:02d}_seed{seed}.pkl"
                )
                df_out.to_pickle(out_file)
                if return_experts and (experts_np is not None):
                    np.savez_compressed(
                        os.path.join(
                            predictions_dir,
                            f"fold{fold_idx:02d}_rep{rep_idx:02d}_seed{seed}_experts.npz",
                        ),
                        experts=experts_np,
                        row_ids=df_out.index.values.astype(str),
                        compound_ids=df_out["CompoundID"].values.astype(str),
                        mc_passes=int(mc_passes),
                        heldout_class=str(heldout),
                    )

    return {
        "f1_scores": f1s,
        "roc_auc_scores": rocs,
        "pr_auc_scores": prs,
        "avg_f1": float(np.mean(f1s)) if f1s else np.nan,
        "avg_roc_auc": float(np.nanmean(rocs)) if rocs else np.nan,
        "avg_pr_auc": float(np.nanmean(prs)) if prs else np.nan,
        "per_fold_outputs": per_fold_outputs,
    }


def evaluate_model_multiclass_pairwise_LOSO(
    model_type: str,
    X,
    y,
    groups,
    sample_ids,
    candidate_moas,
    true_moa_by_sample: pd.Series,
    topk: int = 3,
    seed_repeats: int = 5,
    base_seed: int = 42,
    predictions_dir: str | None = None,
    run_name: str | None = None,
    drop_heldout_candidate_in_train: bool = False,
):
    """
    LOSO for pairwise→multiclass aggregation.
    """
    y_ser = y if isinstance(y, pd.Series) else pd.Series(y)
    groups_ser = groups if isinstance(groups, pd.Series) else pd.Series(groups)
    sample_ser = sample_ids if isinstance(sample_ids, pd.Series) else pd.Series(sample_ids)
    cand_ser = (
        candidate_moas if isinstance(candidate_moas, pd.Series) else pd.Series(candidate_moas)
    )

    classes_global = np.unique(true_moa_by_sample.values)

    if predictions_dir is not None:
        os.makedirs(predictions_dir, exist_ok=True)

    cmp_by_sample = true_moa_by_sample.index.to_series().map(lambda s: s.split("_")[0])
    folds = make_class_loso_splits(
        true_moa_by_sample=true_moa_by_sample.astype(str),
        compound_by_sample=cmp_by_sample.astype(str),
        min_train_classes=2,
    )

    top1s, topks, macro_f1s, micro_f1s, macro_roc_ovr, macro_ap_ovr = [], [], [], [], [], []
    per_fold_outputs = []

    def _ix(obj, mask):
        return obj.iloc[mask] if hasattr(obj, "iloc") else obj[mask]

    for fold_idx, spec in enumerate(folds):
        heldout = spec["heldout_class"]
        tr_cmp = spec["train_compounds"]
        te_cmp = spec["test_compounds"]

        tr_mask = groups_ser.astype(str).isin(tr_cmp).to_numpy()
        te_mask = groups_ser.astype(str).isin(te_cmp).to_numpy()

        if drop_heldout_candidate_in_train:
            tr_mask = tr_mask & (cand_ser.astype(str) != heldout).to_numpy()

        scaler = StandardScaler()
        X_tr_raw = _ix(X, tr_mask)
        X_te_raw = _ix(X, te_mask)
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        y_tr = _ix(y_ser, tr_mask)
        samp_te = _ix(sample_ser, te_mask)
        cand_te = _ix(cand_ser, te_mask)

        for rep_idx in range(seed_repeats):
            seed = int(base_seed + rep_idx)
            clf = get_model(model_type, input_dim=X_tr.shape[1], n_classes=2, seed=seed)
            clf.fit(X_tr, y_tr)

            pair_pos = clf.predict_proba(X_te)[:, 1]
            df_scores = pd.DataFrame(
                {
                    "SampleID": samp_te.values,
                    "CandidateMoA": cand_te.values,
                    "Score": pair_pos,
                }
            )
            pivot = df_scores.pivot_table(
                index="SampleID", columns="CandidateMoA", values="Score", aggfunc="mean"
            ).reindex(columns=classes_global, fill_value=-np.inf)

            test_samples = pivot.index
            y_true = true_moa_by_sample.loc[test_samples].values

            cols = pivot.columns.to_numpy()
            mat = pivot.to_numpy()
            pred_idx = np.argmax(mat, axis=1)
            y_pred = cols[pred_idx]

            top1 = float(np.mean(y_pred == y_true))
            k = min(topk, len(cols))
            order = np.argsort(-mat, axis=1)
            hits = [(y_true[i] in cols[order[i, :k]]) for i in range(len(y_true))]
            topk_acc = float(np.mean(hits)) if len(hits) else np.nan

            present = np.unique(y_true)
            macro_f1 = f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0)
            micro_f1 = f1_score(y_true, y_pred, average="micro", labels=present, zero_division=0)

            ovrs_roc, ovrs_ap = [], []
            for c in present:
                y_bin = (y_true == c).astype(int)
                s_c = pivot[c].to_numpy()
                if y_bin.sum() in (0, len(y_bin)):
                    continue
                try:
                    ovrs_roc.append(roc_auc_score(y_bin, s_c))
                    ovrs_ap.append(average_precision_score(y_bin, s_c))
                except ValueError:
                    pass
            macro_roc = float(np.mean(ovrs_roc)) if ovrs_roc else np.nan
            macro_ap = float(np.mean(ovrs_ap)) if ovrs_ap else np.nan

            top1s.append(top1)
            topks.append(topk_acc)
            macro_f1s.append(float(macro_f1))
            micro_f1s.append(float(micro_f1))
            macro_roc_ovr.append(macro_roc)
            macro_ap_ovr.append(macro_ap)

            agg_df = pivot.copy()
            agg_df["TrueMoA"] = y_true.astype(str)
            agg_df["PredMoA"] = y_pred.astype(str)
            agg_df["rep_idx"] = rep_idx
            agg_df["seed"] = seed
            agg_df["heldout_class"] = heldout
            agg_df.index.name = "SampleID"
            per_fold_outputs.append(agg_df)

            if predictions_dir is not None:
                agg_path = os.path.join(
                    predictions_dir, f"fold{fold_idx:02d}_rep{rep_idx:02d}_seed{seed}_agg.pkl"
                )
                agg_df.to_pickle(agg_path)

    return {
        "top1_accs": top1s,
        "topk_accs": topks,
        "macro_f1s": macro_f1s,
        "micro_f1s": micro_f1s,
        "macro_roc_aucs_ovr": macro_roc_ovr,
        "macro_aps_ovr": macro_ap_ovr,
        "avg_top1_acc": float(np.mean(top1s)) if top1s else np.nan,
        "avg_topk_acc": float(np.mean(topks)) if topks else np.nan,
        "avg_macro_f1": float(np.mean(macro_f1s)) if macro_f1s else np.nan,
        "avg_micro_f1": float(np.mean(micro_f1s)) if micro_f1s else np.nan,
        "avg_macro_roc_auc_ovr": float(np.nanmean(macro_roc_ovr)) if macro_roc_ovr else np.nan,
        "avg_macro_ap_ovr": float(np.nanmean(macro_ap_ovr)) if macro_ap_ovr else np.nan,
        "per_fold_outputs": per_fold_outputs,
        "classes_": list(classes_global),
    }
