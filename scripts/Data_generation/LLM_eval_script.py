# Codebase/feature_combination_evaluation.py
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import mlflow

from Codebase.eval_utils import (
    evaluate_model,                      # binary (pairwise) with cv×seed + dumping
    evaluate_model_multiclass_pairwise,  # pairwise->multiclass with cv×seed + dumping
    evaluate_model_multiclass_samples,   # true multiclass (no_text) with cv×seed + dumping
    build_features,
    generate_backbone_df,
    log_binary_metrics,
    log_multiclass_metrics,
)

parser = argparse.ArgumentParser(
    description="Unified evaluation pipeline (binary / multiclass pairwise / no_text true multiclass) with cv×seed replicates"
)
parser.add_argument('--exp_name', default="experiment", type=str, help="MLflow experiment name")
parser.add_argument('--cv_splits', default=5, type=int, help="Number of cross-val splits (compound-level stratified)")
parser.add_argument('--seed_splits', default=5, type=int, help="Number of model-seed repeats per CV split")
parser.add_argument('--base_seed', default=42, type=int, help="Base seed; replicate seeds are base_seed + [0..seed_splits-1]")
parser.add_argument('--task', default="binary", choices=["binary", "multiclass", "no_text"],
                    help="Evaluation mode")
parser.add_argument('--cv_iterations', default=1, type=int,
                    help="Number of independent CV partitions (for repeated CV).")
parser.add_argument('--folds_manifest', default=None, type=str,
                    help="Path to a shared compound-level fold manifest (JSON). If absent, it will be created.")

args = parser.parse_args()


def main(exp_name: str, cv_splits: int, seed_splits: int, base_seed: int, task: str):
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
    folds_manifest = args.folds_manifest
    if folds_manifest is None:
        # one manifest per experiment+task, shared by all runs in this script
        folds_manifest = os.path.join("predictions", exp_name, "folds", f"{args.task}_compound_folds.json")
        os.makedirs(os.path.dirname(folds_manifest), exist_ok=True)
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
        ["Text_Embedding", "Proteomics"],
    ]

    # Modality sets (true multiclass, no Text_Embedding)
    modality_sets_no_text = [
        ["Proteomics"],
    ]

    # Models & text configs
    model_names = ["pytorch_mlp"]
    llm_models  = ["PubChemDeBERTa", "BERT", "BioBERT"]
    text_aug    = text_embeddings   # switch to `text_embeddings` to run all

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
        llm_grid = [(aug, llm, tdf) for aug, tdf in text_aug.items() for llm in llm_models]
        modality_sets = modality_sets_text

    # ---------- Unified outer loop ----------
    for aug_type, llm_model, text_df in llm_grid:
        # Pairwise backbone only when needed
        if task != "no_text":
            backbone_df = generate_backbone_df(text_df, llm_model, y_labels, samples)

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

                if task == "no_text":
                    # ---- True multiclass on samples ----
                    X_arr = build_features(sample_df, mods)
                    X_df  = pd.DataFrame(X_arr, index=sample_df["SampleID"])

                    expected_dim = sum(modal_dims[m] for m in mods)
                    assert X_df.shape[1] == expected_dim, (
                        f"❌ Dimension mismatch for {mods}: expected {expected_dim}, got {X_df.shape[1]}"
                    )

                    run_name = f"no_text__{model_type}_" + "+".join(mods)
                    print(f"▶ {run_name}")

                    # per-run prediction dump dir
                    pred_dir = os.path.join("predictions", exp_name, "no_text", run_name)
                    os.makedirs(pred_dir, exist_ok=True)

                    with mlflow.start_run(run_name=run_name):
                        mlflow.log_params({
                            "Task": task,
                            "LLM_Model": "None",
                            "Augmentation": "None",
                            "Model_Type": model_type,
                            "Modalities": "+".join(mods),
                            "DataSplit": "BaseOnly",
                            "CV_Splits": cv_splits,
                            "Seed_Splits": seed_splits,
                            "Base_Seed": base_seed,
                        })

                        results = evaluate_model_multiclass_samples(
                            model_type=model_type,
                            X=X_df,
                            y_labels=y_labels.reindex(X_df.index),
                            compound_by_sample=comp_by_sample_sampledf.reindex(X_df.index),
                            cv_splits=cv_splits,
                            seed_splits=seed_splits,
                            base_seed=base_seed,
                            shuffle=True,
                            random_state=42,
                            topk=3,
                            predictions_dir=pred_dir,
                            run_name=run_name,
                            cv_iterations=args.cv_iterations,
                            folds_manifest=folds_manifest,
                        )
                        log_multiclass_metrics(results)

                        all_results.append({
                            "run_name": run_name,
                            "LLM_Model": "None",
                            "Augmentation": "None",
                            "Model_Type": model_type,
                            "Modalities": "+".join(mods),
                            "DataSplit": "BaseOnly",
                            "Task": task,
                            **results
                        })

                else:
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
                            "DataSplit": "BaseOnly",
                            "CV_Splits": cv_splits,
                            "Seed_Splits": seed_splits,
                            "Base_Seed": base_seed,
                            "CV_Iterations": args.cv_iterations,
                            "Folds_Manifest": folds_manifest,   
                        })

                        if task == "binary":
                            results = evaluate_model(
                                model_type=model_type,
                                X=X_df, y=y_pair, groups=groups,
                                cv_splits=cv_splits,
                                seed_splits=seed_splits,
                                base_seed=base_seed,
                                true_moa_by_sample=true_moa_by_sample,
                                compound_by_sample=compound_by_sample,
                                shuffle=True, random_state=42,
                                predictions_dir=pred_dir, run_name=run_name,
                                sample_ids=backbone_df["SampleID"],
                                candidate_moas=backbone_df["Text_MoA"],
                                cv_iterations=args.cv_iterations,
                                folds_manifest=folds_manifest,
                            )

                            log_binary_metrics(results)
                        else:
                            # task == "multiclass" (pairwise -> multiclass)
                            results = evaluate_model_multiclass_pairwise(
                                model_type=model_type,
                                X=X_df, y=y_pair, groups=groups,
                                sample_ids=backbone_df["SampleID"],
                                candidate_moas=backbone_df["Text_MoA"],
                                true_moa_by_sample=true_moa_by_sample,
                                compound_by_sample=compound_by_sample,
                                cv_splits=cv_splits,
                                seed_splits=seed_splits,
                                base_seed=base_seed,
                                shuffle=True,
                                random_state=42,
                                topk=3,
                                predictions_dir=pred_dir,
                                run_name=run_name,
                                store_pairwise=False,
                                cv_iterations=args.cv_iterations,
                                folds_manifest=folds_manifest,
                            )
                                        
                            log_multiclass_metrics(results)

                        all_results.append({
                            "run_name": run_name,
                            "LLM_Model": llm_model,
                            "Augmentation": aug_type,
                            "Model_Type": model_type,
                            "Modalities": "+".join(mods),
                            "DataSplit": "BaseOnly",
                            "Task": task,
                            **results
                        })

    # --- save metrics summary ---
    filename = f"results/{exp_name}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(all_results, f)
    print(f"✅ Metrics summary saved to {filename}")
    print(f"📦 Raw predictions are under predictions/{exp_name}/...")

if __name__ == "__main__":
    main(
        exp_name=args.exp_name,
        cv_splits=int(args.cv_splits),
        seed_splits=int(args.seed_splits),
        base_seed=int(args.base_seed),
        task=args.task
    )
