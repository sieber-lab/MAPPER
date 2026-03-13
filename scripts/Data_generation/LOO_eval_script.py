# Codebase/feature_combination_evaluation.py
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import mlflow

from Codebase.eval_utils import (
    evaluate_model,
    evaluate_model_multiclass_pairwise,
    evaluate_model_multiclass_samples,
    build_features,
    generate_backbone_df,
    log_binary_metrics,
    log_multiclass_metrics,
    evaluate_model_LOO,
    # NEW:
    evaluate_model_LOSO,
    evaluate_model_multiclass_pairwise_LOSO,
)

parser = argparse.ArgumentParser(
    description="LOCO ablation pipeline for TabM (FULL vs No-Proteomics)."
)
parser.add_argument('--exp_name', default="experiment", type=str, help="MLflow experiment name")
parser.add_argument('--seed_repeats', default=5, type=int, help="Number of model-seed repeats per LOCO fold")
parser.add_argument('--base_seed', default=42, type=int, help="Base seed; replicate seeds are base_seed + [0..seed_repeats-1]")
parser.add_argument('--task', default="binary_loso_ablation",
    choices=["binary_loo_ablation", "binary_loso_ablation"],
    help="Evaluation mode")
parser.add_argument('--mc_passes', default=0, type=int,
                    help="MC-dropout stochastic forward passes at inference (0/1 disables MC); saved into *_experts.npz")  # <--- NEW
args = parser.parse_args()

def main(exp_name: str, seed_repeats: int, base_seed: int, task: str, mc_passes: int):
    mlflow.set_tracking_uri("http://10.152.253.136:8085")
    mlflow.set_experiment(exp_name)
    os.makedirs("results", exist_ok=True)

    all_results = []

    # ---------- Data ----------
    BASE_DIR = "Data/Base"
    EMB_SEM  = "Embeddings/250808_semanticAugmentation_embeddings.pkl"
    EMB_SEN  = "Embeddings/250808_sentenceAugmentation_embeddings.pkl"
    EMB_WORD = "Embeddings/250808_wordAugmentation_embeddings.pkl"

    with open(f"{BASE_DIR}/X_proteomics.pkl", 'rb') as f: X_proteomics = pickle.load(f)
    with open(f"{BASE_DIR}/y_labels.pkl",    'rb') as f: y_labels     = pickle.load(f)
    with open(f"{BASE_DIR}/X_cellpaint.pkl", 'rb') as f: X_cellpaint  = pickle.load(f)
    with open(f"{BASE_DIR}/X_ecfp.pkl",      'rb') as f: X_ecfp       = pickle.load(f)
    with open(f"{BASE_DIR}/X_mic.pkl",       'rb') as f: X_mic        = pickle.load(f)
    with open(f"{BASE_DIR}/X_gc.pkl",        'rb') as f: X_gc         = pickle.load(f)

    # Ensure labels are a Series and ALL indices are strings
    if isinstance(y_labels, dict):
        y_labels = pd.Series(y_labels)
    for df in (X_proteomics, X_cellpaint, X_ecfp, X_mic, X_gc):
        df.index = df.index.astype(str)
    y_labels.index = y_labels.index.astype(str)

    # Text embeddings
    text_embeddings = {
        "semantic": pd.read_pickle(EMB_SEM),
        "sentence": pd.read_pickle(EMB_SEN),
        "word":     pd.read_pickle(EMB_WORD),
    }
    # -------- Sanity checks --------
    n = X_proteomics.shape[0]
    assert n == len(y_labels), "Mismatch between X_proteomics and y_labels dimensions."
    assert n == X_cellpaint.shape[0] == X_ecfp.shape[0] == X_mic.shape[0] == X_gc.shape[0]
    assert X_proteomics.shape[1] == 4401, "Expected 4401 features in X_proteomics."
    assert X_cellpaint.shape[1]  == 300,  "Expected 300 features in X_cellpaint."
    assert X_ecfp.shape[1]       == 2048, "Expected 2048 features in X_ecfp."
    assert X_mic.shape[1]        == 20,   "Expected 20 features in X_mic."
    assert X_gc.shape[1]         == 10,   "Expected 10 features in X_gc."

    # Dicts of per-sample modality vectors
    feature_dicts = {
        "Proteomics": X_proteomics.to_dict(orient="index"),
        "ECFP":       X_ecfp.to_dict(orient="index"),
        "CellPaint":  X_cellpaint.to_dict(orient="index"),
        "MIC":        X_mic.to_dict(orient="index"),
        "GC":         X_gc.to_dict(orient="index"),
    }

    # Modality sets (pairwise with Text_Embedding)
    modality_sets_text = [
      #  ["Text_Embedding","ECFP", "CellPaint", "MIC", "GC"],
      #  ["Text_Embedding", "Proteomics", "ECFP", "CellPaint", "MIC", "GC"]
      ["Text_Embedding", "ECFP", "CellPaint"]
    ]

    # Modality sets (true multiclass, no Text_Embedding)
    modality_sets_no_text = [
        # ["ECFP", "CellPaint", "MIC", "GC"],
        # ["Proteomics", "ECFP", "CellPaint", "MIC", "GC"]
        ["ECFP", "CellPaint"]
    ]

    # Models & text configs
    model_names = ["tabm"]
    llm_models  = ["BioBERT"]
    text_aug    = {"semantic": text_embeddings["semantic"]}  # switch to `text_embeddings` to run all

    # Modal dims
    modal_dims = {
        "Text_Embedding": 768,
        "Proteomics":     X_proteomics.shape[1],
        "ECFP":           X_ecfp.shape[1],
        "CellPaint":      X_cellpaint.shape[1],
        "MIC":            X_mic.shape[1],
        "GC":             X_gc.shape[1],
    }

    # SampleIDs (strings)
    samples = X_proteomics.index

    # --- Build a sample-level DF (used by no_text) ---
    sample_df = pd.DataFrame({"SampleID": samples})
    sample_df["CompoundID"] = sample_df["SampleID"].map(lambda s: s.split("_")[0])
    for mod in feature_dicts:
        sample_df[mod] = sample_df["SampleID"].map(lambda s: np.array(list(feature_dicts[mod][s].values())))
    comp_by_sample_sampledf = sample_df.set_index("SampleID")["CompoundID"]

    # --- Choose grid & modality sets per task ---
    if task == "no_text":
        llm_grid = [("None", "None", None)]  # (aug_type, llm_model, text_df placeholder)
        modality_sets = modality_sets_no_text
    else:
        llm_grid = [(aug, tdf, llm) for aug, tdf in {"semantic": text_embeddings["semantic"]}.items() for llm in llm_models]
        modality_sets = modality_sets_text

    # ---------- Unified outer loop ----------
    for aug_type, text_df, llm_model in llm_grid:
        # Pairwise backbone only when needed
        if task != "no_text":
            backbone_df = generate_backbone_df(text_df, llm_model, y_labels, samples)

            print("[sanity] backbone_df rows:", len(backbone_df))
            print("[sanity] unique True_MoA:", backbone_df["True_MoA"].nunique())
            print("[sanity] models in embeddings:", text_df["Model"].unique()[:5])


            # Per-sample series for splitting/metrics
            per_sample = backbone_df.drop_duplicates("SampleID")
            true_moa_by_sample = per_sample.set_index("SampleID")["True_MoA"]
            compound_by_sample = per_sample.set_index("SampleID")["CompoundID"]

            # Attach modality vectors to pairwise rows
            for mod in feature_dicts:
                backbone_df[mod] = backbone_df["SampleID"].map(
                    lambda x: np.array(list(feature_dicts[mod][x].values()))
                )

            y_pair  = backbone_df["Label"]
            groups  = backbone_df["CompoundID"]

        for model_type in model_names:
            for mods in modality_sets:

                # ---- Pairwise (binary or pairwise->multiclass) ----
                X_mod = build_features(backbone_df, mods)
                X_df  = pd.DataFrame(X_mod, index=backbone_df.index)

                expected_dim = sum(modal_dims[m] for m in mods)
                assert X_df.shape[1] == expected_dim, (
                    f"❌ Dimension mismatch for {mods}: expected {expected_dim}, got {X_df.shape[1]}"
                )

                run_name = f"{aug_type}_{llm_model}_{model_type}_" + "+".join(mods)
                print(f"▶ {run_name}")

                # per-run prediction dump dir
                pred_dir = os.path.join("predictions", exp_name, task, run_name)
                os.makedirs(pred_dir, exist_ok=True)

                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params({
                        "Task": task,
                        "LLM_Model": llm_model,
                        "Augmentation": aug_type,
                        "Model_Type": model_type,
                        "Modalities": "+".join(mods),
                        "DataSplit": "LOSO" if task == "binary_loso_ablation" else "LOO",
                        "Seed_Repeats": seed_repeats,
                        "Base_Seed": base_seed,
                        "Return_Experts": True,
                        "mc_passes": int(mc_passes),
                    })

                    if task == "binary_loso_ablation":
                        print("Doing LOSO evaluation...")
                        results = evaluate_model_LOSO(
                            model_type=model_type,
                            X=X_df, y=y_pair, groups=groups,
                            sample_ids=backbone_df["SampleID"],
                            candidate_moas=backbone_df["Text_MoA"],
                            true_moa_by_sample=true_moa_by_sample,   # <- per-sample True MoA
                            seed_repeats=seed_repeats,
                            base_seed=base_seed,
                            predictions_dir=pred_dir,
                            run_name=run_name,
                            return_experts=True,
                            mc_passes=mc_passes,
                            drop_heldout_candidate_in_train=True,   # set True for stricter zero-shot
                        )
                    else:
                        print("Doing LOO evaluation...")
                        results = evaluate_model_LOO(
                            model_type=model_type,
                            X=X_df, y=y_pair, groups=groups,
                            seed_repeats=seed_repeats,
                            base_seed=base_seed,
                            predictions_dir=pred_dir, run_name=run_name,
                            sample_ids=backbone_df["SampleID"],
                            candidate_moas=backbone_df["Text_MoA"],
                            return_experts=True,
                            mc_passes=mc_passes,
                        )

                    log_binary_metrics(results)

    # --- save metrics summary ---
    filename = f"results/{exp_name}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(all_results, f)
    print(f"✅ Metrics summary saved to {filename}")
    print(f"📦 Raw predictions are under predictions/{exp_name}/...")

if __name__ == "__main__":
    main(
        exp_name=args.exp_name,
        seed_repeats=int(args.seed_repeats),
        base_seed=int(args.base_seed),
        task=args.task,
        mc_passes=int(args.mc_passes),  # <--- NEW
    )
