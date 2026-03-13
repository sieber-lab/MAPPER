# 07_METHODS_SECTION_INPUT

## 0) Intended use (for downstream prose model)
- Convert this into manuscript Methods prose without adding new facts.
- Treat each bullet as a factual unit.
- Preserve any `[NEEDS CONFIRMATION]` tags.
- Keep chronology aligned to Figure 2 -> Figure 3 -> Figure 4 -> Figure 5 -> Figure 6.
- Do not add rationale unless explicitly present; this file contains pipeline steps only.

## 1) Project workflow chronology (Figure-linked)

### Stage F2: data overview and modality-level exploratory analysis
- Inputs are precomputed base matrices and labels: `Data/Base/X_proteomics.pkl`, `X_ecfp.pkl`, `X_mic.pkl`, `X_gc.pkl`, `y_labels.pkl`. [ref: notebooks/figure2/figure2_data_overview.ipynb :: code cell :: `pd.read_pickle("Data/Base/X_proteomics.pkl")`]
- MIC panel is generated from MIC matrix using threshold-based MIC extraction (`thresh=0.1` in helper default). [ref: notebooks/analysis/figure2_helpers.py :: `compute_mic_from_matrix` :: `thresh: float = 0.1`]
- Structural chemistry embedding uses UMAP (`n_neighbors=15`, `min_dist=0.1`, `random_state=42`) with PCA fallback. [ref: notebooks/analysis/figure2_helpers.py :: `plot_ecfp_embedding` :: `umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)`]
- Proteomics embedding uses standardized features and UMAP (`n_neighbors=50`, `min_dist=0.5`, `random_state=42`), with PCA fallback. [ref: notebooks/figure2/figure2_data_overview.ipynb :: `plot_proteomics_embedding` :: `umap.UMAP(n_neighbors=50, min_dist=0.5, random_state=random_state)`]
- Operator/batch QC uses KMeans on UMAP coordinates with `n_clusters=3`, `random_state=42`. [ref: notebooks/figure2/figure2_data_overview.ipynb :: code cell :: `KMeans(n_clusters=3, random_state=42)`]

### Stage F3: LLM/augmentation evaluation and text-vs-no-text comparison
- Figure 3c generation runs `LLM_eval_script.py` on binary pairwise task with compound-level fold manifest. [ref: scripts/Data_generation/figure3_data_generation_runs.md :: Figure 3c command :: `LLM_eval_script.py --task binary --folds_manifest fold_manifests/figure3c_compound_folds.json`]
- Observed run metadata for a representative Figure 3c run: `cv_splits=5`, `seed_splits=1`, `cv_iterations=5`, `base_seed=42`. [ref: predictions/LLM_augmentation_experiment/binary/semantic_BioBERT_pytorch_mlp_Text_Embedding+Proteomics/meta.json :: keys :: `cv_splits`, `seed_splits`, `cv_iterations`, `base_seed`]
- Figure 3d compares `task=multiclass` (text+proteomics) versus `task=no_text` (proteomics only) under `Text_vs_No_Text_experiment`. [ref: scripts/Data_generation/figure3_data_generation_runs.md :: Figure 3d commands :: `--task multiclass` and `--task no_text`]
- Figure 3 statistics notebook computes per-resample PR AUC, ROC AUC, and MCC on `(cv_iter, fold_idx)` units and performs repeated-measures analysis. [ref: notebooks/figure3/figure3_llm_parametric_tests.ipynb :: `safe_metrics_one_resample` / `df_for_tests` :: `metric_ls = ['roc_auc','pr_auc','mcc']`]

### Stage F4: feature combination and model family comparison
- Prediction screen is generated with `Feature_eval_script.py` (`task=binary`) and fixed fold manifest. [ref: scripts/Data_generation/figure4_data_generation_runs.md :: Figure 4a command :: `Feature_eval_script.py --task binary --folds_manifest fold_manifests/figure4a_compound_folds.json`]
- Figure 4 compares model families over modality coalitions; script model grid includes `tabm`, `pytorch_mlp`, `mlp`, `xgboost`, `lightgbm`, `rf`. [ref: scripts/Data_generation/Feature_eval_script.py :: model config :: `model_names = ["tabm", "pytorch_mlp", "mlp", "xgboost", "lightgbm", "rf"]`]
- Representative full-modality TabM run metadata shows `cv_splits=5`, `seed_splits=1`, `cv_iterations=5`, `base_seed=42`. [ref: predictions/Feature_augmentation_experiment/binary/semantic_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC/meta.json :: keys :: `cv_splits`, `seed_splits`, `cv_iterations`, `base_seed`]
- Feature-set comparison notebook uses Friedman + posthoc Conover-Friedman with BH correction and reports global-best comparisons. [ref: notebooks/figure4/figure4_feature_replot.ipynb :: `pg.friedman` / `sp.posthoc_conover_friedman(..., p_adjust="fdr_bh")`]
- Exact Shapley values are computed from coalition payoffs for TabM using complete coalition set checks. [ref: notebooks/figure4/figure4_shapley_analysis.ipynb :: `exact_shapley_from_values` :: `assert not missing`]

### Stage F5: uncertainty estimation and external validation views
- LOO and LOSO predictions are generated with `LOO_eval_script.py` for TabM run `semantic_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`. [ref: scripts/Data_generation/figure5_data_generation_runs.md :: Figure 5b/5c commands]
- LOO run metadata: `seed_repeats=5`, `base_seed=42`, `return_experts=true`, `mc_passes=0`. [ref: predictions/Uncertainty_experiment_semantic/binary_loo_ablation/semantic_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC/meta.json :: keys]
- LOSO run metadata: `seed_repeats=5`, `base_seed=42`, `return_experts=true`, `mc_passes=0`, `drop_heldout_candidate_in_train=true`. [ref: predictions/Uncertainty_experiment_semantic/binary_loso_ablation/semantic_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC/meta.json :: keys]
- Uncertainty notebook calibrates all methods to `TARGET_ID_FPR = 0.10`. [ref: notebooks/figure5/figure5_uncertainty_comparison.ipynb :: constants :: `TARGET_ID_FPR = 0.10`]
- Meta-model is `StandardScaler -> PolynomialFeatures(degree=2) -> LogisticRegression(class_weight="balanced")`, wrapped by `CalibratedClassifierCV(method="sigmoid", cv=3)`. [ref: notebooks/figure5/figure5_uncertainty_comparison.ipynb :: deploy block :: `make_pipeline(...)`, `CalibratedClassifierCV(..., cv=3)`]
- Exported uncertainty artifact records `meta_threshold = 0.4690518117171519` and `target_id_fpr = 0.1`. [ref: models/uncertainty_estimator/meta_artifact.json :: keys :: `meta_threshold`, `target_id_fpr`]

### Stage F6: model application to external/eclipsed datasets
- External and eclipse datasets are prepared in dedicated notebooks and saved into paired `test_*` and `train_*_reduced` folders. [ref: notebooks/data_processing/external_proteomics_processing.ipynb :: path constants :: `OUT_TEST_FULL`, `OUT_TRAIN_FULL`; ref: notebooks/data_processing/eclipse_data_processing.ipynb :: path constants :: `OUT_TEST_FULL`, `OUT_TRAIN_FULL`]
- Significant-proteomics filtering uses `PVAL_THRESHOLD = 1.3` (`-log10(p)` scale) and zeroes replicate intensities failing threshold. [ref: notebooks/data_processing/external_proteomics_processing.ipynb :: constants/process cell :: `PVAL_THRESHOLD = 1.3`, `df.loc[~sig_mask, rep_cols] = 0`; ref: notebooks/data_processing/eclipse_data_processing.ipynb :: `PVAL_THRESHOLD = 1.3`]
- Figure 6 application notebooks run `UncertaintyEstimator` on prediction roots for `non_sig_reduced` and `sig_reduced` sets with floor parameters `abs_min_score_floor=0.05`, `score_floor_factor=0.1`, `margin_floor=0.01`. [ref: notebooks/figure6/figure6_external_lab.ipynb :: constructor call :: those args; ref: notebooks/figure6/figure6_eclipse_instrument.ipynb :: constructor call :: those args]
- Figure 6 combined notebook loads `compound_uncertainty_meta.csv` from four roots (eclipse/external x non-sig/sig), colors bars by predicted MoA, and greys bars where `p_error >= meta_threshold`. [ref: notebooks/figure6/figure6_combined.ipynb :: `DATASETS` and `plot_p_error_panel` :: `np.where(df["p_error"] >= meta_threshold, "#aaaaaa", ...)`]
- Figure 6 shipped inference manifests record `used_aug_type="semantic"` for all four external/eclipse roots (`non_sig_reduced`, `sig_reduced`). [ref: predictions/external_data/non_sig_reduced/predict_manifest.json :: key :: `used_aug_type`; ref: predictions/external_data/sig_reduced/predict_manifest.json :: key :: `used_aug_type`; ref: predictions/eclipse_data/non_sig_reduced/predict_manifest.json :: key :: `used_aug_type`; ref: predictions/eclipse_data/sig_reduced/predict_manifest.json :: key :: `used_aug_type`]

## 2) Core method blocks for prose conversion

### 2.1 Data objects and modality matrices
- Base study matrices are loaded from serialized pickles under `Data/Base`. [ref: scripts/Data_generation/Feature_eval_script.py :: main data load :: `BASE_DIR = "Data/Base"` and `X_proteomics.pkl`, `X_ecfp.pkl`, `X_cellpaint.pkl`, `X_mic.pkl`, `X_gc.pkl`, `y_labels.pkl`]
- Text embedding tables are loaded from three augmentation files: semantic, sentence, and word. [ref: scripts/Data_generation/train_primary_tabm.py :: parse args :: `--emb_sem`, `--emb_sentence`, `--emb_word`]
- Pairwise modeling backbone is generated by crossing each sample with each candidate MoA embedding and augmentation index. [ref: scripts/Data_generation/Codebase/eval_utils.py :: `generate_backbone_df` :: loop over `for moa, embeddings in text_embeddings.items()`]
- Binary pairwise label is defined as candidate MoA equals true MoA. [ref: scripts/Data_generation/Codebase/eval_utils.py :: `generate_backbone_df` :: `"Label": int(moa == true_moa)`]

### 2.2 Feature assembly and modality sets
- Pairwise row features are created by concatenating selected modality vectors in declared order (`Text_Embedding`, `Proteomics`, `ECFP`, `CellPaint`, `MIC`, `GC`). [ref: scripts/Data_generation/Codebase/train_utils.py :: `build_features` :: `np.concatenate([r[m] for m in modalities])`; ref: scripts/Data_generation/Feature_eval_script.py :: `modal_dims` keys]
- No-text multiclass variant uses sample-level modality matrices without text pair expansion. [ref: scripts/Data_generation/Feature_eval_script.py :: no-text branch :: `X_arr = build_features(sample_df, mods)`]

### 2.3 Baseline and comparator model definitions
- Most likely primary baseline path for deployment is TabM (`train_primary_tabm.py` + `predict_primary_tabm.py`) with ensemble seeds. [ref: scripts/Data_generation/train_primary_tabm.py :: main :: `get_model("tabm", ...)`; ref: predictions/Figure5_external_validation/Special_v1/predict_manifest.json :: `container_type: "ensemble"`]
- LLM experiment baseline is constrained to `pytorch_mlp`. [ref: scripts/Data_generation/LLM_eval_script.py :: model config :: `model_names = ["pytorch_mlp"]`]
- Feature comparison experiment includes `tabm`, `pytorch_mlp`, `mlp`, `xgboost`, `lightgbm`, and `rf`. [ref: scripts/Data_generation/Feature_eval_script.py :: model config :: `model_names = [...]`]
- TabM wrapper training defaults include `n_epochs=100`, `batch_size=256`, `patience=3`, `lr=1e-4`, `weight_decay=3e-4`. [ref: scripts/Data_generation/Codebase/eval_utils.py :: `TabMWrapper.__init__` :: parameter defaults]
- TabM objective is cross-entropy over expert outputs. [ref: scripts/Data_generation/Codebase/eval_utils.py :: `TabMWrapper.fit` :: `F.cross_entropy(y_pred.flatten(0, 1), y_true.repeat_interleave(k))`]

### 2.4 Splitting, scaling, and randomness controls
- Cross-validation folds are generated at compound level using stratified K-fold over one label per compound. [ref: scripts/Data_generation/Codebase/eval_utils.py :: `build_or_load_compound_folds` :: `StratifiedKFold(...)`]
- Repeated CV uses random state progression `base_seed + 1000*iter`; within-fold seed replicas use `base_seed + rep_idx`. [ref: scripts/Data_generation/Codebase/eval_utils.py :: `build_or_load_compound_folds` and `evaluate_model`]
- In CV evaluators, `StandardScaler` is fit on training rows then applied to held-out rows per fold. [ref: scripts/Data_generation/Codebase/eval_utils.py :: `evaluate_model` :: `scaler.fit_transform(X_tr)` and `scaler.transform(X_te)`]
- TabM model internals apply `QuantileTransformer(output_distribution="normal")` on model inputs. [ref: scripts/Data_generation/Codebase/eval_utils.py :: `TabMWrapper.__init__` / `fit` :: `self.scaler = QuantileTransformer(...)`]
- Seed initialization is explicit for Python/NumPy/Torch in TabM and Lightning wrappers. [ref: scripts/Data_generation/Codebase/eval_utils.py :: `TabMWrapper._set_seed`; ref: `SklearnLikeLightningMLP.fit`]

### 2.5 Metrics and statistical analyses
- Binary metrics computed in run/evaluation pipelines: PR AUC, ROC AUC, MCC, plus thresholded metrics. [ref: scripts/Data_generation/Codebase/eval_utils.py :: `log_binary_metrics`; ref: notebooks/figure3/figure3_llm_parametric_tests.ipynb :: `safe_metrics_one_resample`]
- Multiclass text/no-text notebook computes macro F1, macro PR AUC (OvR), and macro ROC AUC (OvR). [ref: notebooks/figure3/figure3_text_augmentation_effects.ipynb :: metric block :: `macro_f1`, `macro_ap_ovr`, `macro_roc_ovr`]
- Figure 3c repeated-measures setup indexes each outer-fold pair as `cv_cycle = 5*cv_iter + fold_idx`. [ref: notebooks/figure3/figure3_llm_parametric_tests.ipynb :: test table block :: `cv_cycle` definition]
- Repeated-measures ANOVA and Tukey-style posthoc are provided via analysis helper `rm_tukey_hsd`. [ref: notebooks/analysis/model_comparison.py :: `rm_tukey_hsd` :: `pg.rm_anova` and Tukey calculations]
- Figure 4 significance testing uses Friedman and Conover-Friedman with BH correction. [ref: notebooks/figure4/figure4_feature_replot.ipynb :: analysis block :: `pg.friedman`, `sp.posthoc_conover_friedman(..., p_adjust="fdr_bh")`]
- Figure 5 pairwise uncertainty stats include Holm-adjusted p-values and Cliff's delta with bootstrap CI (`N_BOOT = 5000`). [ref: notebooks/figure5/figure5_uncertainty_comparison.ipynb :: stats block :: `holm_adjust`, `N_BOOT = 5000`, `cliffs_delta_boot`]

### 2.6 Uncertainty modeling and deployment logic
- Candidate uncertainty methods evaluated per compound: low max score, high predictive entropy, high expected entropy, Z-gate global, Z-gate class-conditional, and meta-model. [ref: notebooks/figure5/figure5_uncertainty_comparison.ipynb :: `run_eta_loo_per_compound` :: `methods = [...]`]
- Uncertainty outcome metrics are coverage, precision, error recall, and F1 for flagged-error detection. [ref: notebooks/figure5/figure5_uncertainty_comparison.ipynb :: per_rows block :: `coverage`, `precision`, `err_recall`, `f1`]
- Deployed artifact contains feature order, threshold, target ID-FPR, random state, and preprocessing directives (`negate=["PredScore","Margin"]`, `fillna_value=0.0`). [ref: models/uncertainty_estimator/meta_artifact.json :: keys]
- Runtime uncertainty application enforces floor overrides before final p_error/flag (`PredScore` and `Margin` floor rules can force `p_error=1.0`). [ref: scripts/uncertainty/uncertainty_estimator.py :: `_apply_meta_with_floor` :: `df.loc[df["floor_applied"], "p_error_meta"] = 1.0`]

### 2.7 Primary train/predict deployment path
- Training CLI builds a TabM ensemble from base data with modality flags and seed range `base_seed .. base_seed+num_seeds-1`. [ref: scripts/Data_generation/train_primary_tabm.py :: parse args/main loop :: `--base_seed`, `--num_seeds`, `seed = int(args.base_seed + rep_idx)`]
- Prediction CLI consumes either single bundle or ensemble and writes pairwise and multiclass summary tables plus `predict_manifest.json`. [ref: scripts/Data_generation/predict_primary_tabm.py :: outputs block :: `pairwise_predictions.csv`, `compound_multiclass_summary.csv`, `predict_manifest.json`]
- Representative shipped prediction manifests for Figure 5/6 runs show `mc_passes=0`, `return_experts=true`, and seeds `[42,43,44,45,46]`. [ref: predictions/Figure5_external_validation/Special_v1/predict_manifest.json :: keys; ref: predictions/eclipse_data/non_sig_reduced/predict_manifest.json :: keys]

## 3) Explicit discrepancies and alignment notes to preserve in prose draft notes
- Script defaults for `Feature_eval_script.py` and `LLM_eval_script.py` are `seed_splits=5`, `cv_iterations=1`, but representative saved runs use `seed_splits=1`, `cv_iterations=5`. [ref: scripts/Data_generation/Feature_eval_script.py :: parser defaults; ref: predictions/Feature_augmentation_experiment/binary/semantic_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC/meta.json :: observed values; ref: predictions/LLM_augmentation_experiment/binary/semantic_BioBERT_pytorch_mlp_Text_Embedding+Proteomics/meta.json :: observed values]
- `predict_primary_tabm.py` default has `--return_experts` off unless explicitly set, but shipped manifests indicate `return_experts=true`. [ref: scripts/Data_generation/predict_primary_tabm.py :: parse args :: `--return_experts`; ref: predictions/Figure5_external_validation/Special_v1/predict_manifest.json :: `return_experts`]
- Figure 5 runbook optional training example uses `--aug_type semantic`; observed shipped manifests for Figure 5 Special_v1 and all Figure 6 external/eclipse roots also report `used_aug_type="semantic"`. [ref: scripts/Data_generation/figure5_data_generation_runs.md :: optional train command :: `--aug_type semantic`; ref: predictions/Figure5_external_validation/Special_v1/predict_manifest.json :: key :: `used_aug_type`; ref: predictions/external_data/non_sig_reduced/predict_manifest.json :: key :: `used_aug_type`; ref: predictions/external_data/sig_reduced/predict_manifest.json :: key :: `used_aug_type`; ref: predictions/eclipse_data/non_sig_reduced/predict_manifest.json :: key :: `used_aug_type`; ref: predictions/eclipse_data/sig_reduced/predict_manifest.json :: key :: `used_aug_type`]

## 4) Items requiring confirmation before final manuscript prose
- Exact dependency versions (PyTorch/sklearn/XGBoost/LightGBM/etc.) are not pinned in a repository lockfile. [NEEDS CONFIRMATION]. [ref: README.md :: notes :: external dependencies listed without versions]
- One text multiclass manifest path appears as `fold_manifests/figure3d_text_multiclass_compound_folds.json`; presence of that exact file in current repo should be rechecked during manuscript finalization. [NEEDS CONFIRMATION]. [ref: predictions/Text_vs_No_Text_experiment/multiclass/semantic_BioBERT_pytorch_mlp_Text_Embedding+Proteomics/meta.json :: `folds_manifest`]

## 5) MLflow provenance status (confirmed)
- External MLflow stores are available under `/mnt/jh/delta/code/pkg/mlflow/jh/mlruns` and `/mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old`. [ref: /mnt/jh/delta/code/pkg/mlflow/jh :: directory listing :: `mlruns`, `mlruns_old`]
- Figure-generation experiment IDs used in mapping docs are present in `mlruns_old` (including `125845022789070498`, `694949682285093199`, `879354154507378166`, `603034844447165657`). [ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old/125845022789070498/meta.yaml :: key :: `experiment_id`; ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old/694949682285093199/meta.yaml :: key :: `experiment_id`; ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old/879354154507378166/meta.yaml :: key :: `experiment_id`; ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old/603034844447165657/meta.yaml :: key :: `experiment_id`]
- Representative run folders for Figures 3-5 include `mlflow.source.name` tags pointing to `LLM_eval_script_250916.py`, `Feature_eval_script_250917.py`, and `LOO_eval_script_251006.py`. [ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old/125845022789070498/e086e399acc745a0a12161edd198967b/tags/mlflow.source.name :: tag :: `LLM_eval_script_250916.py`; ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old/694949682285093199/2fa80c27dc814e759c29ded5e138aa5b/tags/mlflow.source.name :: tag :: `Feature_eval_script_250917.py`; ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old/879354154507378166/0d54cd002bd7434189fa13eb021c7bf1/tags/mlflow.source.name :: tag :: `LOO_eval_script_251006.py`]
- The train/predict primary TabM path remains artifact-manifest based in curated outputs (`ensemble_manifest.json`, `predict_manifest.json`) for provenance capture. [ref: scripts/Data_generation/train_primary_tabm.py :: export block :: `ensemble_manifest.json`; ref: scripts/Data_generation/predict_primary_tabm.py :: output manifest block :: `predict_manifest.json`]
