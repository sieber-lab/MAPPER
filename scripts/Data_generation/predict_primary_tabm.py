# Codebase/predict_primary_tabm.py
from __future__ import annotations
import argparse
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any

# Reuse your training-side helpers
from Codebase.train_utils import (
    load_base_data,
    make_feature_dicts,
    assemble_pairwise_training_table,
    load_inference_data  # builds backbone_df + X + y using Text_Embedding + selected modalities
)

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Predict with a single TabM bundle or an ensemble of TabM primary models on a held-out test set (pairwise scoring + multiclass aggregation)."
    )

    # --- Required: trained model dir (single bundle or ensemble root) ---
    p.add_argument("--model_dir", required=True, type=str,
                   help="Path to exported model bundle (contains model.pkl, config.json, feature_order.json) "
                        "or ensemble root (contains ensemble_manifest.json and seed_* subdirs).")

    # --- Data roots (defaults target your 'Special' test-only folder) ---
    p.add_argument("--base_dir", default="Data/Special", type=str,
                   help="Folder with ONLY the held-out test samples (X_*.pkl and y_labels.pkl for MoA class list).")
    p.add_argument("--emb_sem", default="Embeddings/250808_semanticAugmentation_embeddings.pkl", type=str)
    p.add_argument("--emb_sentence", default="Embeddings/250808_sentenceAugmentation_embeddings.pkl", type=str)
    p.add_argument("--emb_word", default="Embeddings/250808_wordAugmentation_embeddings.pkl", type=str)

    # --- Optional test selection (if you still want to subset within Data/Special) ---
    p.add_argument("--test_index_csv", type=str, default=None,
                   help="CSV with a column 'SampleID' listing samples to score. If omitted, all samples in base_dir are used.")
    p.add_argument("--test_index_col", type=str, default="SampleID",
                   help="Column name in --test_index_csv that contains SampleIDs (default: SampleID).")

    # --- Inference-time options (supported by TabMWrapper) ---
    p.add_argument("--mc_passes", type=int, default=0,
                   help="If >1, enable MC-Dropout inference and save *_experts_*.npz per seed (T,k,C) per row if available.")
    p.add_argument("--return_experts", action="store_true",
                   help="Save per-expert probabilities for uncertainty pipelines (per seed). Ignored if model lacks it.")

    # --- Output control ---
    p.add_argument("--out_dir", type=str, default=None,
                   help="Where to write predictions. Default: <model_dir>/predictions")
    p.add_argument("--run_tag", type=str, default=None,
                   help="Optional subfolder inside out_dir (e.g., a test-set name).")

    # --- Safety / checks ---
    p.add_argument("--strict_dim_check", action="store_true",
                   help="If set, raise when X.shape[1] != model.input_dim (if available).")

    return p.parse_args()


# ---------------------------
# Bundle/ensemble loaders
# ---------------------------
def _canon_feature_order(raw: Any, cfg: dict) -> List[str]:
    """Normalize feature order from json (list or dict) or fallback to cfg['modalities']."""
    if isinstance(raw, list):
        feat_order = raw
    elif isinstance(raw, dict):
        if "modalities" in raw and isinstance(raw["modalities"], list):
            feat_order = raw["modalities"]
        elif "feature_order" in raw and isinstance(raw["feature_order"], list):
            feat_order = raw["feature_order"]
        else:
            raise ValueError("feature_order.json exists but lacks a 'modalities' or 'feature_order' list.")
    else:
        feat_order = cfg.get("modalities")
        if not isinstance(feat_order, list) or not len(feat_order):
            raise ValueError("No usable feature order; expected list or 'modalities' key.")
    # Canonicalize + ensure Text_Embedding first
    def _canon_mod(name: str) -> str:
        n = str(name).strip().lower()
        mapping = {
            "text_embedding": "Text_Embedding",
            "text-embedding": "Text_Embedding",
            "text": "Text_Embedding",
            "proteomics": "Proteomics",
            "ecfp": "ECFP",
            "cellpaint": "CellPaint",
            "mic": "MIC",
            "gc": "GC",
        }
        return mapping.get(n, name if isinstance(name, str) else str(name))
    feat_order = [_canon_mod(m) for m in feat_order]
    if "Text_Embedding" in feat_order:
        feat_order = ["Text_Embedding"] + [m for m in feat_order if m != "Text_Embedding"]
    return feat_order


def _load_single_bundle(bundle_dir: Path):
    """Load a single model bundle."""
    cfg = json.loads((bundle_dir / "config.json").read_text())
    model = pickle.loads((bundle_dir / "model.pkl").read_bytes())
    fo_path = bundle_dir / "feature_order.json"
    raw_fo = json.loads(fo_path.read_text()) if fo_path.exists() else cfg.get("modalities")
    feat_order = _canon_feature_order(raw_fo, cfg)
    mod_dims_path = bundle_dir / "modality_dims.json"
    mod_dims = json.loads(mod_dims_path.read_text()) if mod_dims_path.exists() else None
    llm_model = str(cfg.get("llm_model", "BioBERT"))
    aug_type  = str(cfg.get("aug_type", "semantic"))
    return {
        "type": "single",
        "dir": str(bundle_dir.resolve()),
        "model": model,
        "cfg": cfg,
        "feature_order": feat_order,
        "mod_dims": mod_dims,
        "llm_model": llm_model,
        "aug_type": aug_type,
        "seed": int(cfg.get("seed", 0)),
    }


def _load_ensemble_root(root_dir: Path):
    """Load an ensemble root with ensemble_manifest.json and member seed_* dirs."""
    manifest_path = root_dir / "ensemble_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Ensemble root missing ensemble_manifest.json: {manifest_path}")
    man = json.loads(manifest_path.read_text())

    members = []
    for mdir in man.get("member_dirs", []):
        bdir = Path(mdir)
        if not (bdir / "model.pkl").exists() or not (bdir / "config.json").exists():
            raise FileNotFoundError(f"Ensemble member incomplete: {bdir}")
        members.append(_load_single_bundle(bdir))

    # Validate consistency across members (feature_order, llm_model, aug_type)
    if not members:
        raise ValueError("ensemble_manifest.json lists no member_dirs.")
    feat0 = members[0]["feature_order"]
    llm0  = members[0]["llm_model"]
    aug0  = members[0]["aug_type"]
    for m in members[1:]:
        if m["feature_order"] != feat0:
            raise ValueError("Feature order mismatch across ensemble members.")
        if m["llm_model"] != llm0 or m["aug_type"] != aug0:
            raise ValueError("LLM model or augmentation type mismatch across ensemble members.")

    return {
        "type": "ensemble",
        "dir": str(root_dir.resolve()),
        "members": members,
        "feature_order": feat0,
        "llm_model": llm0,
        "aug_type": aug0,
        "manifest": man,
    }


def _load_model_container(model_dir: Path) -> dict:
    """Return a dict describing either a single bundle or an ensemble."""
    if (model_dir / "ensemble_manifest.json").exists():
        return _load_ensemble_root(model_dir)
    # else expect single bundle files present
    required = ["model.pkl", "config.json"]
    for r in required:
        if not (model_dir / r).exists():
            raise FileNotFoundError(f"--model_dir missing required file '{r}': {model_dir}")
    return _load_single_bundle(model_dir)


# ---------------------------
# Utilities (sample selection, aggregation)
# ---------------------------
def _select_samples(all_index: pd.Index, test_index_csv: str | None, col: str) -> pd.Index:
    """If a CSV is provided, intersect with available samples; otherwise return all."""
    if test_index_csv is None:
        return all_index
    df = pd.read_csv(test_index_csv)
    if col not in df.columns:
        raise ValueError(f"'{col}' not found in {test_index_csv}.")
    sel = df[col].astype(str).values
    sel = np.intersect1d(all_index.astype(str).values, sel)
    if len(sel) == 0:
        raise ValueError("No overlapping SampleIDs between Base data and provided test index.")
    return pd.Index(sel.astype(str))


def _aggregate_pairwise(df_pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    From pairwise rows (SampleID, CandidateMoA, Score), produce:
      - per-sample wide score table  (rows=SampleID, cols=MoA; mean across augmentations)
      - per-compound summary with top-1, second, margin (mean across sample rows/augs)
    """
    pivot = df_pair.pivot_table(
        index="SampleID", columns="CandidateMoA", values="Score", aggfunc="mean"
    ).sort_index(axis=1)

    # Sample->Compound
    cmp_by_sample = pd.Series(pivot.index, index=pivot.index).map(lambda s: str(s).split("_")[0])

    cols = pivot.columns.to_numpy()
    M = pivot.to_numpy()
    Mfill = np.where(np.isnan(M), -np.inf, M)
    top1_idx = np.argmax(Mfill, axis=1)
    order = np.argsort(-Mfill, axis=1)
    top2_idx = np.where(Mfill.shape[1] >= 2, order[:, 1], order[:, 0])

    pred_lbl = cols[top1_idx]
    pred_scr = M[np.arange(M.shape[0]), top1_idx]
    sec_lbl  = cols[top2_idx]
    sec_scr  = M[np.arange(M.shape[0]), top2_idx]
    margin   = pred_scr - sec_scr

    sample_summary = pd.DataFrame({
        "SampleID": pivot.index.astype(str),
        "CompoundID": cmp_by_sample.values.astype(str),
        "PredictedMoA": pred_lbl.astype(str),
        "PredScore": pred_scr.astype(float),
        "SecondMoA": sec_lbl.astype(str),
        "SecondScore": sec_scr.astype(float),
        "Margin": margin.astype(float),
    }).set_index("SampleID")

    # Aggregate to compound (mean of scores across samples, then take top-1 on mean)
    comp_scores = pivot.copy()
    comp_scores["CompoundID"] = cmp_by_sample.values.astype(str)
    comp_mean = comp_scores.groupby("CompoundID").mean(numeric_only=True)

    cols2 = comp_mean.columns.to_numpy()
    M2    = comp_mean.to_numpy()
    M2f   = np.where(np.isnan(M2), -np.inf, M2)
    t1c   = np.argmax(M2f, axis=1)
    ord2  = np.argsort(-M2f, axis=1)
    t2c   = np.where(M2f.shape[1] >= 2, ord2[:, 1], ord2[:, 0])

    comp_pred_lbl = cols2[t1c]
    comp_pred_scr = M2[np.arange(M2.shape[0]), t1c]
    comp_sec_lbl  = cols2[t2c]
    comp_sec_scr  = M2[np.arange(M2.shape[0]), t2c]
    comp_margin   = comp_pred_scr - comp_sec_scr

    comp_summary = pd.DataFrame({
        "CompoundID": comp_mean.index.astype(str),
        "PredictedMoA": comp_pred_lbl.astype(str),
        "PredScore": comp_pred_scr.astype(float),
        "SecondMoA": comp_sec_lbl.astype(str),
        "SecondScore": comp_sec_scr.astype(float),
        "Margin": comp_margin.astype(float),
    }).set_index("CompoundID")

    return sample_summary, comp_summary


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"--model_dir not found: {model_dir}")

    out_root = Path(args.out_dir) if args.out_dir else (model_dir / "predictions")
    if args.run_tag:
        out_root = out_root / args.run_tag
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Load bundle OR ensemble
    container = _load_model_container(model_dir)
    if container["type"] == "single":
        members = [container]
        feat_order = container["feature_order"]
        llm_model = container["llm_model"]
        aug_type  = container["aug_type"]
        seeds     = [container["seed"]]
        print("[load] single bundle:", container["dir"])
    else:
        members   = container["members"]
        feat_order = container["feature_order"]
        llm_model = container["llm_model"]
        aug_type  = container["aug_type"]
        seeds     = [m["seed"] for m in members]
        print("[load] ensemble members:", len(members), "seeds:", seeds)

    print("[predict] resolved feature_order:", feat_order)
    print(f"[predict] llm_model={llm_model}, aug_type={aug_type}")

    # 2) Load base data (+ embeddings) for test-only folder

    # selected modalities = bundle modalities excluding Text_Embedding (first)
    if len(feat_order) == 0 or feat_order[0] != "Text_Embedding":
        raise ValueError("Feature order must begin with 'Text_Embedding'.")
    selected_modalities = [m for m in feat_order if m != "Text_Embedding"]

    # Now load *only* what is needed for these modalities + the chosen aug_type
    data = load_inference_data(
        base_dir=args.base_dir,
        emb_sem=args.emb_sem,
        emb_sentence=args.emb_sentence,
        emb_word=args.emb_word,
        selected_modalities=selected_modalities,
        aug_type=aug_type,
    )
    X_prot = data["X_proteomics"]; X_ecfp = data["X_ecfp"]; X_cp = data["X_cellpaint"]
    X_mic = data["X_mic"]; X_gc = data["X_gc"]; y_labels = data["y_labels"]
    text_embeddings = data["text_embeddings"]


    # Selected samples = everything in this folder (or subset via CSV if provided)
    all_samples = X_prot.index
    test_samples = _select_samples(all_samples, args.test_index_csv, args.test_index_col)
    print(f"[select] using {len(test_samples)} samples for prediction.")

    # 3) Build pairwise backbone and feature matrix for TEST samples
    text_df = text_embeddings[aug_type]
    feat_dicts = make_feature_dicts(X_prot, X_ecfp, X_cp, X_mic, X_gc)

    # selected modalities = bundle modalities excluding Text_Embedding (first)
    if len(feat_order) == 0 or feat_order[0] != "Text_Embedding":
        raise ValueError("Feature order must begin with 'Text_Embedding'.")
    selected_modalities = [m for m in feat_order if m != "Text_Embedding"]

    backbone_df, X, _ = assemble_pairwise_training_table(
        text_df=text_df,
        llm_model=llm_model,
        y_labels=y_labels,           # labels not used for scoring; needed to enumerate classes
        samples=test_samples,
        feature_dicts=feat_dicts,
        selected_modalities=selected_modalities,
    )

    # 4) Optional dimension check against the FIRST member
    if args.strict_dim_check:
        first_model = members[0]["model"]
        if hasattr(first_model, "input_dim"):
            exp = int(getattr(first_model, "input_dim"))
            if X.shape[1] != exp:
                raise ValueError(f"Feature dimension mismatch: model expects {exp}, got {X.shape[1]}.")

    # 5) Inference per member (seed)
    print(f"[infer] mc_passes={args.mc_passes}, return_experts={args.return_experts}")
    per_seed_scores: Dict[int, np.ndarray] = {}
    long_rows = []  # for pairwise_predictions_by_seed.csv
    for m in members:
        seed = int(m["seed"])
        model = m["model"]

        if args.return_experts or args.mc_passes > 1:
            try:
                mean_arr, experts = model.predict_proba(X, return_experts=True, mc_passes=int(args.mc_passes))
                proba_pos = mean_arr[:, 1]
            except TypeError:
                mean_arr = model.predict_proba(X)
                proba_pos = mean_arr[:, 1]
                experts = None
        else:
            mean_arr = model.predict_proba(X)
            proba_pos = mean_arr[:, 1]
            experts = None

        per_seed_scores[seed] = proba_pos

        # Save experts per seed if available
        if experts is not None:
            np.savez_compressed(
                out_root / f"pairwise_experts_seed{seed}.npz",
                experts=experts,  # (N, k, C) or (N, T, k, C)
                row_ids=backbone_df.index.astype(str).values
                         if backbone_df.index is not None else np.arange(len(proba_pos)).astype(str),
                sample_ids=backbone_df["SampleID"].astype(str).values,
                compound_ids=backbone_df["CompoundID"].astype(str).values,
                candidate_moas=backbone_df["Text_MoA"].astype(str).values,
                mc_passes=int(args.mc_passes),
                seed=seed,
            )
            print(f"[write] experts (seed={seed}) -> {out_root / f'pairwise_experts_seed{seed}.npz'}")

        # accumulate long format now to avoid reconstructing later
        long_rows.append(pd.DataFrame({
            "RowID":       backbone_df.index.astype(str).values,
            "SampleID":    backbone_df["SampleID"].astype(str).values,
            "CompoundID":  backbone_df["CompoundID"].astype(str).values,
            "CandidateMoA":backbone_df["Text_MoA"].astype(str).values,
            "Aug_Index":   backbone_df["Aug_Index"].astype(int).values,
            "Seed":        seed,
            "Score":       proba_pos.astype(float),
        }))

    # 6) Write per-seed long and wide, plus mean/std
    df_long = pd.concat(long_rows, axis=0, ignore_index=True)
    by_seed_pivot = df_long.pivot_table(
        index=["RowID","SampleID","CompoundID","CandidateMoA","Aug_Index"],
        columns="Seed", values="Score", aggfunc="mean"  # each seed appears once
    ).reset_index()

    seed_cols = sorted([c for c in by_seed_pivot.columns if isinstance(c, (int, np.integer))])
    scores_mat = by_seed_pivot[seed_cols].to_numpy(dtype=float) if seed_cols else None
    if scores_mat is not None:
        by_seed_pivot["ScoreMean"] = np.nanmean(scores_mat, axis=1)
        by_seed_pivot["ScoreStd"]  = np.nanstd(scores_mat, axis=1)
        score_mean_vec = by_seed_pivot["ScoreMean"].to_numpy()
    else:
        # single bundle fallback
        by_seed_pivot["ScoreMean"] = by_seed_pivot[seed_cols[0]].to_numpy()
        by_seed_pivot["ScoreStd"]  = 0.0
        score_mean_vec = by_seed_pivot["ScoreMean"].to_numpy()

    # 7) Save pairwise outputs:
    #    A) canonical pairwise (mean across seeds) -> preserves augmentation rows
    out_pair = pd.DataFrame({
        "SampleID":     by_seed_pivot["SampleID"].astype(str).values,
        "CompoundID":   by_seed_pivot["CompoundID"].astype(str).values,
        "CandidateMoA": by_seed_pivot["CandidateMoA"].astype(str).values,
        "Aug_Index":    by_seed_pivot["Aug_Index"].astype(int).values,
        "Score":        score_mean_vec.astype(float),
    })
    pair_path = out_root / "pairwise_predictions.csv"
    out_pair.to_csv(pair_path, index=False)
    print(f"[write] pairwise (mean across seeds) -> {pair_path}")

    #    B) long per-seed
    by_seed_long_path = out_root / "pairwise_predictions_by_seed.csv"
    df_long.to_csv(by_seed_long_path, index=False)
    print(f"[write] pairwise by-seed (long) -> {by_seed_long_path}")

    #    C) wide per-seed with mean/std
    by_seed_wide_path = out_root / "pairwise_seed_wide.csv"
    by_seed_pivot.to_csv(by_seed_wide_path, index=False)
    print(f"[write] pairwise by-seed (wide) -> {by_seed_wide_path}")

    # 8) Aggregate to multiclass (sample + compound) and save (using MEAN across seeds)
    sample_summary, comp_summary = _aggregate_pairwise(out_pair)
    sample_csv = out_root / "sample_multiclass_summary.csv"
    comp_csv   = out_root / "compound_multiclass_summary.csv"
    sample_summary.to_csv(sample_csv)
    comp_summary.to_csv(comp_csv)
    print(f"[write] sample multiclass -> {sample_csv}")
    print(f"[write] compound multiclass -> {comp_csv}")

    # 9) Save a manifest for reproducibility
    manifest = {
        "model_dir": str(model_dir.resolve()),
        "container_type": container["type"],
        "used_llm_model": llm_model,
        "used_aug_type": aug_type,
        "feature_order": feat_order,
        "n_test_samples": int(len(test_samples)),
        "mc_passes": int(args.mc_passes),
        "return_experts": bool(args.return_experts),
        "pairwise_predictions_csv": str(pair_path.resolve()),
        "pairwise_by_seed_long_csv": str(by_seed_long_path.resolve()),
        "pairwise_by_seed_wide_csv": str(by_seed_wide_path.resolve()),
        "sample_multiclass_csv": str(sample_csv.resolve()),
        "compound_multiclass_csv": str(comp_csv.resolve()),
    }
    if container["type"] == "ensemble":
        manifest["ensemble_manifest"] = str((model_dir / "ensemble_manifest.json").resolve())
        manifest["seeds"] = seeds
    else:
        manifest["seed"] = int(container["seed"])

    (out_root / "predict_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[write] manifest -> {out_root/'predict_manifest.json'}")


if __name__ == "__main__":
    main()
