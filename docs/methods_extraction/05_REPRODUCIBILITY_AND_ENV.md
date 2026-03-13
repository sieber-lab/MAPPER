# 05_REPRODUCIBILITY_AND_ENV

## 1) Seed and determinism controls
- Evaluation split determinism: repeated CV fold RNG uses `base_seed + 1000*iter`; per-fold model replicas use `base_seed + rep_idx`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: build_or_load_compound_folds/evaluate_model :: `base_seed`]
- TabM seed handling sets Python/NumPy/Torch/CUDA seeds with offsets (`seed`, `seed+1`, `seed+2`, `seed+3`). [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: TabMWrapper._set_seed :: `random.seed`, `np.random.seed`, `torch.manual_seed`]
- Lightning-MLP wrapper sets Python/NumPy/Torch seeds before fit. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: SklearnLikeLightningMLP.fit :: `random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)`]
- Text augmentation uses `seed=42` default for deterministic shuffles. [ref: code_repository/scripts/Preprocessing/Text_processing.py :: augment_moa_descriptions :: `seed: int = 42`]
- Text embedding helper enables deterministic torch/transformers execution (`torch.use_deterministic_algorithms(True)`, `transformers.enable_full_determinism`). [ref: code_repository/scripts/Preprocessing/text_embeddings.py :: TextEmbedding.set_seed :: determinism calls]

## 2) Dependency specification state
- Explicit environment lockfiles (`requirements*.txt`, `pyproject.toml`, `environment*.yml`, Dockerfile) were not found under `code_repository` in this scan. [NEEDS CONFIRMATION]. [ref: code_repository/README.md :: Notes :: `External dependencies (RDKit, PyTorch, Lightning, XGBoost, etc.) are required`]
- README confirms dependency families but not pinned versions. [NEEDS CONFIRMATION]. [ref: code_repository/README.md :: Notes :: `External dependencies ... are required`]

## 3) Hardware/runtime assumptions in code
- TabM wrapper auto-selects CUDA if available, otherwise CPU (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`). [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: TabMWrapper.__init__ :: `self.device = torch.device('cuda' ... )`]
- Lightning-MLP uses `accelerator='gpu' if torch.cuda.is_available() else 'cpu'`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: SklearnLikeLightningMLP.__init__ :: `self.device = 'gpu' if torch.cuda.is_available() else 'cpu'`]
- LLM perplexity helper defaults to `device='cuda' if available else 'cpu'`. [ref: code_repository/scripts/Preprocessing/Text_processing.py :: compute_llm_perplexities :: `device: str = "cuda" if torch.cuda.is_available() else "cpu"`]

## 4) MLflow configuration and run logging
- Scripts set MLflow tracking URI to `http://10.152.253.136:8085` and experiment to CLI `--exp_name`. [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: main :: `mlflow.set_tracking_uri("http://10.152.253.136:8085")`]
- Each run logs structured params (`Task`, `LLM_Model`, `Augmentation`, `Model_Type`, split/seed params, manifest path) and logs average metrics. [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: mlflow.log_params :: keys]
- Binary and multiclass metric names logged are centralized in `log_binary_metrics` and `log_multiclass_metrics`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: log_binary_metrics/log_multiclass_metrics :: `mlflow.log_metric(...)`]
- Local `mlruns/` tree is not shipped in `code_repository`; docs reference an external MLflow storage location. [ref: code_repository/docs/figure_generation_mlflow_mapping.md :: Source MLflow root :: `/mnt/jh/delta/code/pkg/mlflow/jh/mlruns`]

## 5) Artifact-level reproducibility evidence in repo
- Prediction runs persist `meta.json` with task/model/CV settings (210 files in `predictions/**/meta.json`). [ref: code_repository/predictions/LLM_augmentation_experiment/binary/semantic_BioBERT_pytorch_mlp_Text_Embedding+Proteomics/meta.json :: keys :: `cv_splits`, `seed_splits`, `cv_iterations`]
- Primary ensemble training persists `config.json` and `ensemble_manifest.json` with seeds and modality order. [ref: code_repository/models/tabM_all_modalities/ensemble_manifest.json :: keys :: `seeds`, `feature_order`]
- Inference persists `predict_manifest.json` with model root, used augmentation/model, outputs, and seed members. [ref: code_repository/predictions/Figure5_external_validation/Special_v1/predict_manifest.json :: keys :: `model_dir`, `used_aug_type`, `seeds`]

## 6) Verified default-vs-run discrepancies (important)
- `Feature_eval_script.py` default `seed_splits=5` / `cv_iterations=1` differs from saved run artifacts (`seed_splits=1`, `cv_iterations=5`) for figure experiments. [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: parser defaults :: `seed_splits`, `cv_iterations`]
- `LLM_eval_script.py` has same default-vs-artifact discrepancy (`5/1` defaults vs `1/5` in saved `meta.json`). [ref: code_repository/scripts/Data_generation/LLM_eval_script.py :: parser defaults :: `seed_splits`, `cv_iterations`]
- `predict_primary_tabm.py` default `return_experts=False` differs from saved manifests (`"return_experts": true`). [ref: code_repository/scripts/Data_generation/predict_primary_tabm.py :: parse_args :: `--return_experts`]
- Some saved fold-manifest references in `meta.json` are missing in current curated repo (e.g., `fold_manifests/figure3d_text_multiclass_compound_folds.json`). [NEEDS CONFIRMATION]. [ref: code_repository/predictions/Text_vs_No_Text_experiment/multiclass/semantic_BioBERT_pytorch_mlp_Text_Embedding+Proteomics/meta.json :: key :: `folds_manifest`]

## 7) Path assumptions
- README states relative paths assume original project-level folders (`Data/`, `predictions/`, `Embeddings/`) remain available. [ref: code_repository/README.md :: Notes :: `Relative paths in notebooks/scripts assume the original top-level folders still exist`]
- Saved model/prediction manifests contain absolute paths under `/mnt/nas/...`; portability to other hosts requires path remapping. [ref: code_repository/models/tabM_all_modalities/ensemble_manifest.json :: `member_dirs` :: `/mnt/nas/...`]
