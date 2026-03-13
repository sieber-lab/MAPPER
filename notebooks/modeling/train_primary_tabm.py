# train_primary_tabm.py
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np

# ✅ Your existing model factory (TabMWrapper etc.)
from modeling.eval_utils import get_model

# ✅ Your existing data + export helpers
from modeling.train_utils import (
    load_base_data, make_feature_dicts, modality_dims_from_frames,
    assemble_pairwise_training_table, export_model_bundle
)

def parse_args():
    p = argparse.ArgumentParser(
        description="Train an ensemble of primary TabM models on the FULL base dataset (pairwise, BioBERT+text)."
    )
    # Data roots
    p.add_argument("--base_dir", default="Data/Base", type=str)
    p.add_argument("--emb_sem", default="Embeddings/250808_semanticAugmentation_embeddings.pkl", type=str)
    p.add_argument("--emb_sentence", default="Embeddings/250808_sentenceAugmentation_embeddings.pkl", type=str)
    p.add_argument("--emb_word", default="Embeddings/250808_wordAugmentation_embeddings.pkl", type=str)

    # Text backbone choices (defaults: BioBERT + semantic)
    p.add_argument("--llm_model", default="BioBERT", type=str)
    p.add_argument("--aug_type", default="semantic", choices=["semantic","sentence","word"], type=str)

    # Modalities (booleans) + convenience flag
    p.add_argument("--use_proteomics", action="store_true", help="Include Proteomics")
    p.add_argument("--use_ecfp",       action="store_true", help="Include ECFP")
    p.add_argument("--use_cellpaint",  action="store_true", help="Include CellPaint")
    p.add_argument("--use_mic",        action="store_true", help="Include MIC")
    p.add_argument("--use_gc",         action="store_true", help="Include GC")
    p.add_argument("--all_modalities", action="store_true",
                   help="Shortcut to include Proteomics, ECFP, CellPaint, MIC, GC")

    # Reproducibility / Ensemble
    p.add_argument("--base_seed", default=42, type=int,
                   help="Base seed; replica seeds are base_seed + [0..num_seeds-1].")
    p.add_argument("--num_seeds", default=5, type=int,
                   help="Number of seed replicas to train (>=1).")

    # Export
    p.add_argument("--export_dir", default=None, type=str,
                   help="Root folder to store the ensemble bundle. "
                        "Default: <base_dir>/models/tabm_<llm>_<aug>_ensemble/")
    p.add_argument("--run_name", default=None, type=str,
                   help="Optional prefix to insert in the ensemble folder name.")

    return p.parse_args()


def main():
    args = parse_args()

    if args.num_seeds < 1:
        raise ValueError("--num_seeds must be >= 1")

    # -----------------------------
    # 1) Load data (once)
    # -----------------------------
    data = load_base_data(
        base_dir=args.base_dir,
        emb_sem=args.emb_sem,
        emb_sentence=args.emb_sentence,
        emb_word=args.emb_word,
    )
    X_prot = data["X_proteomics"]; X_ecfp = data["X_ecfp"]; X_cp = data["X_cellpaint"]
    X_mic = data["X_mic"]; X_gc = data["X_gc"]; y_labels = data["y_labels"]
    text_embeddings = data["text_embeddings"]

    # Sanity (same as before)
    n = X_prot.shape[0]
    assert n == len(y_labels) == X_cp.shape[0] == X_ecfp.shape[0] == X_mic.shape[0] == X_gc.shape[0]

    # Select the text table
    text_df = text_embeddings[args.aug_type]
    samples = X_prot.index

    # -----------------------------
    # 2) Decide modalities
    # -----------------------------
    selected_modalities = []
    if args.all_modalities:
        args.use_proteomics = args.use_ecfp = args.use_cellpaint = args.use_mic = args.use_gc = True

    if args.use_proteomics: selected_modalities.append("Proteomics")
    if args.use_ecfp:       selected_modalities.append("ECFP")
    if args.use_cellpaint:  selected_modalities.append("CellPaint")
    if args.use_mic:        selected_modalities.append("MIC")
    if args.use_gc:         selected_modalities.append("GC")

    # Always include text embedding first (consistent with eval pipelines)
    feature_order = ["Text_Embedding"] + selected_modalities

    # -----------------------------
    # 3) Build pairwise table + features (once)
    # -----------------------------
    feat_dicts = make_feature_dicts(X_prot, X_ecfp, X_cp, X_mic, X_gc)
    backbone_df, X, y = assemble_pairwise_training_table(
        text_df=text_df,
        llm_model=args.llm_model,
        y_labels=y_labels,
        samples=samples,
        feature_dicts=feat_dicts,
        selected_modalities=selected_modalities,   # assemble will prepend Text_Embedding if needed
    )

    print(f"[train] Backbone rows (pairwise): {len(backbone_df):,}")
    print(f"[train] X shape: {X.shape}, y positives: {int(y.sum())}, negatives: {int((1 - y).sum())}")

    # -----------------------------
    # 4) Prepare export root
    # -----------------------------
    base_dir = Path(args.base_dir)   # actual base_dir
    if args.export_dir is None:
        root = base_dir / "models"
        root.mkdir(parents=True, exist_ok=True)
        name_bits = ["tabm", args.llm_model, args.aug_type, "ensemble"]
        if args.run_name:
            name_bits.insert(0, args.run_name)
        export_root = root / "_".join(name_bits)
    else:
        export_root = Path(args.export_dir)
    export_root.mkdir(parents=True, exist_ok=True)

    # Modal dims (stored once at the root and also per-seed via export_model_bundle)
    modal_dims = modality_dims_from_frames(
        X_proteomics=X_prot, X_ecfp=X_ecfp, X_cellpaint=X_cp, X_mic=X_mic, X_gc=X_gc, text_dim=768
    )

    # -----------------------------
    # 5) Train multiple seeds
    # -----------------------------
    seed_dirs = []
    for rep_idx in range(int(args.num_seeds)):
        seed = int(args.base_seed + rep_idx)
        print(f"\n[replica] Training seed={seed}")

        model = get_model("tabm", input_dim=X.shape[1], n_classes=2, seed=seed)
        model.fit(X, y)  # full-dataset fit

        # Per-seed export directory
        seed_dir = export_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "model_type": "tabm",
            "seed": seed,
            "llm_model": args.llm_model,
            "aug_type": args.aug_type,
            "modalities": feature_order,  # includes "Text_Embedding" first
            "n_train_rows": int(len(backbone_df)),
            "class_balance": {
                "pos": int(y.sum()),
                "neg": int((1 - y).sum()),
            },
        }

        export_model_bundle(
            export_dir=seed_dir,
            model_obj=model,
            config=config,
            feature_order=feature_order,
            modality_dims=modal_dims,
        )

        seed_dirs.append(str(seed_dir.resolve()))
        print(f"[replica] ✅ Saved seed={seed} to {seed_dir}")

    # -----------------------------
    # 6) Write top-level ensemble manifest
    # -----------------------------
    ens_manifest = {
        "ensemble_type": "seed_replicas",
        "base_seed": int(args.base_seed),
        "num_seeds": int(args.num_seeds),
        "seeds": [int(args.base_seed + i) for i in range(int(args.num_seeds))],
        "member_dirs": seed_dirs,
        "llm_model": args.llm_model,
        "aug_type": args.aug_type,
        "feature_order": feature_order,
        "modality_dims": modal_dims,
        "train_meta": {
            "base_dir": str(Path(args.base_dir).resolve()),
            "emb_sem": str(Path(args.emb_sem).resolve()),
            "emb_sentence": str(Path(args.emb_sentence).resolve()),
            "emb_word": str(Path(args.emb_word).resolve()),
            "n_train_rows": int(len(backbone_df)),
        },
        "note": "Each member subdir is a complete export bundle (model.pkl, config.json, feature_order.json, modality_dims.json).",
    }
    (export_root / "ensemble_manifest.json").write_text(json.dumps(ens_manifest, indent=2))
    print(f"\n📦 Ensemble manifest: {export_root/'ensemble_manifest.json'}")
    print(f"✅ Trained {args.num_seeds} TabM replicas under: {export_root}")

if __name__ == "__main__":
    main()
