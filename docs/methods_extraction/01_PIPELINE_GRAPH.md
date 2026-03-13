# 01_PIPELINE_GRAPH

## Step 1 baseline determination (evidence only)
- Candidate baseline A (most likely): **TabM pairwise classifier** used in dedicated `train_primary_tabm.py`/`predict_primary_tabm.py`, plus TabM-only ablation script configuration. [ref: code_repository/scripts/Data_generation/train_primary_tabm.py :: main :: `get_model("tabm", ...)`]
- Evidence for A as primary path: model bundles in `models/tabM_*` persist `"model_type": "tabm"`; external prediction manifests point to TabM ensemble roots. [ref: code_repository/models/tabM_all_modalities/seed_42/config.json :: config key :: `"model_type": "tabm"`]
- Candidate baseline B (task-specific): **pytorch_mlp** for LLM augmentation experiment (`LLM_eval_script.py` hard-codes `model_names = ["pytorch_mlp"]`). [ref: code_repository/scripts/Data_generation/LLM_eval_script.py :: model config :: `model_names = ["pytorch_mlp"]`]
- Candidate baseline C (comparative screen, not single baseline): feature-eval grid runs `tabm`, `pytorch_mlp`, `mlp`, `xgboost`, `lightgbm`, `rf`. [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: model config :: `model_names = ["tabm", "pytorch_mlp", "mlp", "xgboost", "lightgbm", "rf"]`]

## End-to-end pipeline outline (what happens)

### 1) Data inputs
- Base training/evaluation loaders read modality pickles and labels from `Data/Base`. [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: main data load :: `X_proteomics.pkl`, `X_ecfp.pkl`, `y_labels.pkl`]
- Primary prediction loader reads held-out data from `Data/Special` by default (overrideable). [ref: code_repository/scripts/Data_generation/predict_primary_tabm.py :: parse_args :: `--base_dir default="Data/Special"`]
- Text embeddings are loaded from three fixed pickle files (`semantic/sentence/word`). [ref: code_repository/scripts/Data_generation/train_primary_tabm.py :: parse_args :: `--emb_sem`, `--emb_sentence`, `--emb_word`]
- Artifacts produced at this step: in-memory `X_*` dataframes, `y_labels`, `text_embeddings` dict. [ref: code_repository/scripts/Data_generation/Codebase/train_utils.py :: load_base_data :: `return dict(... text_embeddings=...)`]

### 2) Pairwise backbone construction and feature building
- Pairwise rows are built as `(SampleID, CandidateMoA, Aug_Index)` over all candidate MoAs from the selected text table and all augmentations. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: generate_backbone_df :: `for moa, embeddings in text_embeddings.items()`]
- Binary label is constructed as `Label = int(Text_MoA == True_MoA)`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: generate_backbone_df :: `"Label": int(moa == true_moa)`]
- Modality vectors are attached per row from sample-indexed dictionaries, then concatenated by modality order. [ref: code_repository/scripts/Data_generation/Codebase/train_utils.py :: attach_modalities_to_backbone/build_features :: `np.concatenate([r[m] for m in modalities])`]
- Key parameters/config keys: `mods` (modality set), `llm_model`, `aug_type`, `text_col="Embedding"`. [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: loops :: `for mods in modality_sets`]
- Artifacts: pairwise feature matrix `X_df`; for no-text task, sample-level matrix from selected modalities only. [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: no_text branch :: `X_arr = build_features(sample_df, mods)`]

### 3) Split generation (where splits happen)
- Repeated CV tasks (`binary`, `multiclass`, `no_text`) use compound-level **StratifiedKFold** manifests via `build_or_load_compound_folds`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: build_or_load_compound_folds :: `StratifiedKFold(... random_state=base_seed + 1000*it)`]
- LOO ablation uses one held-out compound per fold (`unique_compounds`). [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: evaluate_model_LOO :: `for fold_idx, test_cmp in enumerate(unique_compounds)`]
- LOSO ablation uses one held-out MoA class per fold (`make_class_loso_splits`). [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: make_class_loso_splits :: `train = MoA != C, test = MoA == C`]
- Artifacts: fold manifest JSONs and per-fold prediction dumps (`fold...pkl`). [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: evaluate_model :: `fold{fold_idx:02d}_iter{cv_iter:02d}_rep{rep_idx:02d}_seed{seed}.pkl`]

### 4) Scaling/normalization before model fit (where scaling happens)
- CV evaluators apply `StandardScaler.fit_transform` on train and `transform` on test per fold. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: evaluate_model :: `scaler = StandardScaler(); X_tr = scaler.fit_transform(...); X_te = scaler.transform(...)`]
- TabM model wrapper additionally applies a `QuantileTransformer(output_distribution='normal')` inside `TabMWrapper.fit/transform`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: TabMWrapper :: `self.scaler = QuantileTransformer(output_distribution='normal')`]
- Primary train script (`train_primary_tabm.py`) does not apply outer `StandardScaler`; it relies on TabM internal quantile scaling. [ref: code_repository/scripts/Data_generation/train_primary_tabm.py :: main train loop :: `model.fit(X, y)`]

### 5) Model training
- Model factory resolves by `model_type` to LightGBM/XGBoost/sklearn MLP/RF/Lightning-MLP/TabM. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: get_model :: `if name == "lightgbm" ... elif name == "tabm"`]
- In repeated CV tasks, each fold is fit `seed_splits` times with seeds `base_seed + rep_idx`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: evaluate_model :: `for rep_idx in range(seed_splits)`]
- Primary external model training fits TabM on full pairwise table and exports one bundle per seed. [ref: code_repository/scripts/Data_generation/train_primary_tabm.py :: main :: `for rep_idx in range(int(args.num_seeds))`]
- Artifacts: `model.pkl`, `config.json`, `feature_order.json`, `modality_dims.json`, `ensemble_manifest.json`. [ref: code_repository/scripts/Data_generation/Codebase/train_utils.py :: export_model_bundle :: `model.pkl`, `config.json`]

### 6) Prediction and aggregation
- CV evaluators store per-row probabilities (`ProbaPos`) and fold metadata; multiclass pairwise tasks pivot by `(SampleID, CandidateMoA)` and pick argmax for top-1/top-k metrics. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: evaluate_model_multiclass_pairwise :: `pivot_table(... values="Score")`]
- Primary inference supports ensemble averaging across seed members and optional per-expert tensors (`pairwise_experts_seed*.npz`). [ref: code_repository/scripts/Data_generation/predict_primary_tabm.py :: inference loop :: `return_experts`, `mc_passes`]
- Primary outputs: pairwise mean/by-seed CSVs + sample/compound multiclass summaries + `predict_manifest.json`. [ref: code_repository/scripts/Data_generation/predict_primary_tabm.py :: write outputs :: `pairwise_predictions.csv`, `compound_multiclass_summary.csv`]

### 7) Metric logging/evaluation outputs
- Binary metrics logged: `avg_f1`, `avg_roc_auc`, `avg_pr_auc`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: log_binary_metrics :: `mlflow.log_metric("avg_f1", ...)`]
- Multiclass metrics logged: `avg_top1_acc`, `avg_topk_acc`, `avg_macro_f1`, `avg_micro_f1`, `avg_macro_roc_auc_ovr`, `avg_macro_ap_ovr`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: log_multiclass_metrics :: `avg_top1_acc`]
- Script-level summary artifact is `results/<exp_name>.pkl`. [ref: code_repository/scripts/Data_generation/LLM_eval_script.py :: save summary :: `filename = f"results/{exp_name}.pkl"`]

### 8) Reporting/figure-stage exports
- Figure notebooks consume prediction roots and write `analysis_exports/*` summary/statistics artifacts. [ref: code_repository/notebooks/figure3/figure3_llm_parametric_tests.ipynb :: output text cell :: `Saved analysis artifacts to: analysis_exports/LLM_augmentation_experiment...`]
- Uncertainty notebook trains and exports meta-model artifacts under `analysis_exports/uncertainty_estimator/meta_model/`. [ref: code_repository/notebooks/figure5/figure5_uncertainty_comparison.ipynb :: deploy cell output :: `meta_model.joblib`, `meta_artifact.json`]

## Split/scaling/randomness flags (explicit)
- Splits: compound-level stratified repeated CV (`n_splits`, `n_iters`), plus LOO and LOSO custom loops. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: build_or_load_compound_folds/evaluate_model_LOO/evaluate_model_LOSO :: `n_splits`, `heldout_class`]
- Scaling: `StandardScaler` per fold and TabM `QuantileTransformer` internal scaler. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: evaluate_model + TabMWrapper :: `StandardScaler`, `QuantileTransformer`]
- Randomness control: fold seed schedule (`base_seed + 1000*iter`), replicate seeds (`base_seed + rep_idx`), model internal seeds for Python/NumPy/Torch/CUDA. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: _set_seed/build_or_load_compound_folds :: `random.seed`, `np.random.seed`, `torch.manual_seed`]
