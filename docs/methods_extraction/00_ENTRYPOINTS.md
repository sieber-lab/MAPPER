# 00_ENTRYPOINTS

## Step 0 repo scan (quick inventory)
- Top-level workflow folders in `code_repository`: `Data/`, `Embeddings/`, `scripts/`, `notebooks/`, `predictions/`, `models/`, `results/`, `analysis_exports/`, `fold_manifests/`, `docs/`. [ref: code_repository/README.md :: Structure :: `notebooks/`, `docs/`, `images/`]
- Primary runnable script roots: `scripts/Data_generation/*.py`, `scripts/Preprocessing/Preprocessing_pipelines.py`, `notebooks/modeling/*.py`, `notebooks/data_processing/run_preprocessing.py`. [ref: code_repository/scripts/Data_generation/train_primary_tabm.py :: __main__ :: `if __name__ == "__main__"`]
- Key notebooks present for figure analyses and data prep include `notebooks/figure*/`, `notebooks/supplementary/`, plus deterministic data-build notebooks in `notebooks/data_processing/`. [ref: code_repository/notebooks/data_processing/external_proteomics_processing.ipynb :: intro cell :: `This notebook creates deterministic external datasets`]
- MLflow is used in evaluation scripts and metric logging utilities. [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: main :: `mlflow.set_tracking_uri`]
- Local `mlruns/` is not present inside `code_repository`; docs point to an external MLflow store path (`/mnt/jh/delta/code/pkg/mlflow/jh/mlruns`). [ref: code_repository/docs/figure3_5_subplot_data_generation.md :: MLflow access assumptions :: `/mnt/jh/delta/code/pkg/mlflow/jh/mlruns`]

## Candidate entrypoints (priority order)

### A) Python scripts with `if __name__ == "__main__"` (highest priority)
1. `scripts/Data_generation/Feature_eval_script.py`
- Purpose: repeated compound-level CV for feature-combination experiments across models (`binary`, `multiclass`, `no_text`). [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: parser/main :: `choices=["binary", "multiclass", "no_text"]`]
- How to run: `PYTHONPATH=scripts/Data_generation python scripts/Data_generation/Feature_eval_script.py --exp_name ... --task ...` (documented runbook). [ref: code_repository/scripts/Data_generation/figure4_data_generation_runs.md :: Prediction generation command :: `PYTHONPATH=scripts/Data_generation`]
- Inputs: `Data/Base/X_*.pkl`, `Data/Base/y_labels.pkl`, text embedding pickles under `Embeddings/`. [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: main data load :: `BASE_DIR = "Data/Base"`]
- Outputs: per-run prediction folders with `meta.json` and fold dumps under `predictions/<exp>/<task>/<run_name>/`; summary pickle `results/<exp>.pkl`; MLflow params/metrics. [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: main tail :: `filename = f"results/{exp_name}.pkl"`]

2. `scripts/Data_generation/LLM_eval_script.py`
- Purpose: repeated CV for LLM/augmentation comparison (configured to `pytorch_mlp`, `Text_Embedding+Proteomics`, three LLMs, three augmentation sets). [ref: code_repository/scripts/Data_generation/LLM_eval_script.py :: model/text config :: `model_names = ["pytorch_mlp"]`]
- How to run: `PYTHONPATH=scripts/Data_generation python scripts/Data_generation/LLM_eval_script.py --exp_name LLM_augmentation_experiment --task binary ...`. [ref: code_repository/scripts/Data_generation/figure3_data_generation_runs.md :: Figure 3c command :: `LLM_eval_script.py`]
- Inputs: `Data/Base/*.pkl`, `Embeddings/*Augmentation_embeddings.pkl`. [ref: code_repository/scripts/Data_generation/LLM_eval_script.py :: main data load :: `EMB_SEM/EMB_SEN/EMB_WORD`]
- Outputs: `predictions/LLM_augmentation_experiment/binary/*/meta.json` + fold pickles, `results/LLM_augmentation_experiment.pkl`, MLflow logs. [ref: code_repository/scripts/Data_generation/figure3_data_generation_runs.md :: Outputs produced :: `results/LLM_augmentation_experiment.pkl`]

3. `scripts/Data_generation/LOO_eval_script.py`
- Purpose: TabM-only proteomics ablation in `binary_loo_ablation` and `binary_loso_ablation`, with optional experts/MC outputs. [ref: code_repository/scripts/Data_generation/LOO_eval_script.py :: parser :: `choices=["binary_loo_ablation", "binary_loso_ablation"]`]
- How to run: `PYTHONPATH=scripts/Data_generation python scripts/Data_generation/LOO_eval_script.py --exp_name ... --task ... --seed_repeats ...`. [ref: code_repository/scripts/Data_generation/figure5_data_generation_runs.md :: Prediction generation commands :: `LOO_eval_script.py`]
- Inputs: `Data/Base/*.pkl`, `Embeddings/*Augmentation_embeddings.pkl`. [ref: code_repository/scripts/Data_generation/LOO_eval_script.py :: main data load :: `BASE_DIR = "Data/Base"`]
- Outputs: prediction folders with `meta.json`, per-fold `.pkl`, optional `*_experts.npz`, plus `results/<exp>.pkl`, MLflow logs. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: evaluate_model_LOSO :: `fold{fold_idx:02d}_rep{rep_idx:02d}_seed{seed}_experts.npz`]

4. `scripts/Data_generation/train_primary_tabm.py`
- Purpose: train an ensemble of TabM seed replicas on full pairwise training table and export model bundles/manifests. [ref: code_repository/scripts/Data_generation/train_primary_tabm.py :: parse_args/main :: `description="Train an ensemble of primary TabM models..."`]
- How to run: `PYTHONPATH=scripts/Data_generation python scripts/Data_generation/train_primary_tabm.py --all_modalities ...`. [ref: code_repository/scripts/Data_generation/figure5_data_generation_runs.md :: Optional train command :: `train_primary_tabm.py`]
- Inputs: `Data/Base` (or overridden `--base_dir`), embedding pickles, modality flags. [ref: code_repository/scripts/Data_generation/train_primary_tabm.py :: parse_args :: `--base_dir`, `--emb_sem`, `--all_modalities`]
- Outputs: `models/<...>/seed_<seed>/{model.pkl,config.json,feature_order.json,modality_dims.json}` and `ensemble_manifest.json`. [ref: code_repository/scripts/Data_generation/train_primary_tabm.py :: export section :: `ensemble_manifest.json`]

5. `scripts/Data_generation/predict_primary_tabm.py`
- Purpose: run single-bundle or ensemble inference on held-out test set, aggregate pairwise to sample/compound multiclass summaries, optionally save experts. [ref: code_repository/scripts/Data_generation/predict_primary_tabm.py :: parse_args/main :: `description="Predict with a single TabM bundle or an ensemble..."`]
- How to run: `PYTHONPATH=scripts/Data_generation python scripts/Data_generation/predict_primary_tabm.py --model_dir ... --out_dir ... --return_experts`. [ref: code_repository/scripts/Data_generation/figure5_data_generation_runs.md :: Generate external-set predictions :: `predict_primary_tabm.py`]
- Inputs: model bundle/ensemble dir, `Data/Special` by default (or override), embedding pickles. [ref: code_repository/scripts/Data_generation/predict_primary_tabm.py :: parse_args :: `--model_dir`, `--base_dir`, `--emb_sem`]
- Outputs: `pairwise_predictions.csv`, `pairwise_predictions_by_seed.csv`, `pairwise_seed_wide.csv`, `sample_multiclass_summary.csv`, `compound_multiclass_summary.csv`, `predict_manifest.json`, optional `pairwise_experts_seed*.npz`. [ref: code_repository/scripts/Data_generation/predict_primary_tabm.py :: write outputs :: `predict_manifest.json`]

6. `scripts/Preprocessing/Preprocessing_pipelines.py`
- Purpose: full preprocessing (proteomics + chemistry + cell-paint + text augmentation + embeddings). [ref: code_repository/scripts/Preprocessing/Preprocessing_pipelines.py :: main docstring :: `full data preprocessing pipeline`]
- How to run: `python scripts/Preprocessing/Preprocessing_pipelines.py`. [ref: code_repository/scripts/Preprocessing/Preprocessing_pipelines.py :: __main__ :: `main()`]
- Inputs: FASTA, MS tables, compound Excel, MoA descriptions Excel. [ref: code_repository/scripts/Preprocessing/Preprocessing_pipelines.py :: config/steps :: `fasta_path`, `250808_compounds.xlsx`, `MoA_descriptions.xlsx`]
- Outputs: timestamped `Data/Compound_data/*_preprocessed_*.pkl` and `Embeddings/*_embeddings.pkl`. [ref: code_repository/scripts/Preprocessing/Preprocessing_pipelines.py :: save calls :: `to_pickle`, `output_pickle_path`]

### B) CLIs in notebook-backed Python modules (secondary, duplicate logic)
7. `notebooks/modeling/train_primary_tabm.py` and `notebooks/modeling/predict_primary_tabm.py`
- Purpose: CLI equivalents of the `scripts/Data_generation` TabM train/predict path. [ref: code_repository/notebooks/modeling/train_primary_tabm.py :: __main__ :: `main()`]
- Inputs/outputs: same argument structure and artifacts as script versions. [ref: code_repository/notebooks/modeling/predict_primary_tabm.py :: parse_args :: `--model_dir`, `--out_dir`, `--return_experts`]

8. `notebooks/data_processing/run_preprocessing.py`
- Purpose: module-run preprocessing orchestrator equivalent to script pipeline. [ref: code_repository/notebooks/data_processing/run_preprocessing.py :: main :: preprocessing steps 1-5]
- Outputs: same family of timestamped preprocessed pickles and embeddings. [ref: code_repository/notebooks/data_processing/run_preprocessing.py :: save calls :: `*_preprocessed_*.pkl`]

### C) Notebooks that appear to execute end-to-end data generation/evaluation pipelines
9. `notebooks/data_processing/external_proteomics_processing.ipynb`
- Purpose: build deterministic external test/train-reduced datasets and write aligned modality pickles. [ref: code_repository/notebooks/data_processing/external_proteomics_processing.ipynb :: intro cell :: `creates deterministic external datasets`]
- Key outputs: `Data/external_data/test_*`, `train_*_reduced`, `Compound_data` pickles. [ref: code_repository/notebooks/data_processing/external_proteomics_processing.ipynb :: cells with save paths :: `OUT_TEST_FULL`, `OUT_TRAIN_FULL`, `to_pickle`]

10. `notebooks/data_processing/eclipse_data_processing.ipynb`
- Purpose: build deterministic eclipse evaluation datasets and reduced training sets. [ref: code_repository/notebooks/data_processing/eclipse_data_processing.ipynb :: intro cell :: `creates deterministic datasets for eclipse evaluation`]
- Key outputs: `Data/eclipse_data/test_*`, `train_*_reduced`, `Compound_data` pickles. [ref: code_repository/notebooks/data_processing/eclipse_data_processing.ipynb :: cells with save paths :: `OUT_TEST_FULL`, `OUT_TRAIN_FULL`, `to_pickle`]

11. Figure-analysis notebooks consuming generated prediction artifacts
- Figure 3 notebooks consume `predictions/LLM_augmentation_experiment` and `predictions/Text_vs_No_Text_experiment`. [ref: code_repository/notebooks/figure3/figure3_llm_parametric_tests.ipynb :: config cell :: `EXP = "LLM_augmentation_experiment"`]
- Figure 4 notebooks consume `predictions/Feature_augmentation_experiment` and write `analysis_exports/Feature_augmentation_experiment/*`. [ref: code_repository/notebooks/figure4/figure4_feature_replot.ipynb :: config cell :: `EXP = "Feature_augmentation_experiment"`]
- Figure 5 notebooks consume `predictions/Uncertainty_experiment_semantic` (LOO + LOSO) and `predictions/Figure5_external_validation/Special_v1`. [ref: code_repository/notebooks/figure5/figure5_uncertainty_comparison.ipynb :: config cell :: `ETA_EXP_LOO`, `ETA_EXP_LOSO`]

### D) Pipeline runner docs (command orchestrators)
12. `scripts/Data_generation/figure3_data_generation_runs.md`, `figure4_data_generation_runs.md`, `figure5_data_generation_runs.md`
- Purpose: explicit execution order and CLI invocations for figure data regeneration. [ref: code_repository/scripts/Data_generation/figure3_data_generation_runs.md :: command blocks :: `LLM_eval_script.py`, `Feature_eval_script.py`]
- Outputs referenced: prediction roots, result pickles, analysis exports. [ref: code_repository/scripts/Data_generation/figure5_data_generation_runs.md :: Analysis outputs used by Figure 5 combined :: `analysis_exports/uncertainty_estimator/*`]

13. `docs/figure3_5_subplot_data_generation.md`
- Purpose: detailed run-to-panel provenance including MLflow experiment IDs and commands. [ref: code_repository/docs/figure3_5_subplot_data_generation.md :: sections :: `MLflow experiment`]
- Status note: contains commands for versioned scripts not present in current `scripts/Data_generation/` (`Feature_eval_script_250917.py`, `LOO_eval_script_251006.py`). [NEEDS CONFIRMATION]. [ref: code_repository/docs/figure3_5_subplot_data_generation.md :: command blocks :: `Feature_eval_script_250917.py`, `LOO_eval_script_251006.py`]

## MLflow-first files to inspect for run truth (from this scan)
- `scripts/Data_generation/Feature_eval_script.py`, `LLM_eval_script.py`, `LOO_eval_script.py`, and `scripts/Data_generation/Codebase/eval_utils.py` contain `mlflow.*` calls. [ref: code_repository/scripts/Data_generation/LOO_eval_script.py :: main :: `mlflow.set_tracking_uri`]
- Saved run-parameter artifacts in repo: `predictions/**/meta.json` (210 files), `models/**/config.json`, `models/**/ensemble_manifest.json`, `predictions/**/predict_manifest.json`. [ref: code_repository/predictions/Feature_augmentation_experiment/binary/semantic_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC/meta.json :: root keys :: `cv_splits`, `seed_splits`, `cv_iterations`]
