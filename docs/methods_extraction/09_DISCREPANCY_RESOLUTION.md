# 09_DISCREPANCY_RESOLUTION

This file records discrepancies listed in `07_METHODS_SECTION_INPUT.md` section `3) Explicit discrepancies`.
For manuscript figure descriptions, observed saved-run configurations are treated as canonical.

## Discrepancy 1: `seed_splits` and `cv_iterations` defaults vs observed figure runs

- Default in code:
  - `Feature_eval_script.py`: `seed_splits=5`, `cv_iterations=1`. [ref: scripts/Data_generation/Feature_eval_script.py :: argparse parser defaults :: `add_argument('--seed_splits', default=5)`, `add_argument('--cv_iterations', default=1)`]
  - `LLM_eval_script.py`: `seed_splits=5`, `cv_iterations=1`. [ref: scripts/Data_generation/LLM_eval_script.py :: argparse parser defaults :: `add_argument('--seed_splits', default=5)`, `add_argument('--cv_iterations', default=1)`]
- Observed in saved run:
  - Figure 3c representative run: `seed_splits=1`, `cv_iterations=5`. [ref: predictions/LLM_augmentation_experiment/binary/semantic_BioBERT_pytorch_mlp_Text_Embedding+Proteomics/meta.json :: keys :: `seed_splits`, `cv_iterations`]
  - Figure 4 representative run: `seed_splits=1`, `cv_iterations=5`. [ref: predictions/Feature_augmentation_experiment/binary/semantic_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC/meta.json :: keys :: `seed_splits`, `cv_iterations`]
- Decision for manuscript:
  - Use observed saved-run values (`seed_splits=1`, `cv_iterations=5`) for figure run descriptions.
  - Keep parser defaults only as implementation notes.

## Discrepancy 2: `--return_experts` default behavior vs observed saved inference runs

- Default in code:
  - `predict_primary_tabm.py` defines `--return_experts` as `action="store_true"`, so default is false unless flag is passed. [ref: scripts/Data_generation/predict_primary_tabm.py :: parse_args :: `p.add_argument("--return_experts", action="store_true")`]
- Observed in saved run:
  - Representative prediction manifests used for Figure 5/6 report `return_experts: true`. [ref: predictions/Figure5_external_validation/Special_v1/predict_manifest.json :: key :: `return_experts`; ref: predictions/eclipse_data/non_sig_reduced/predict_manifest.json :: key :: `return_experts`]
- Decision for manuscript:
  - Use observed saved-run setting (`return_experts=true`) for figure run descriptions.
  - Keep CLI default behavior only as implementation note.

## Discrepancy 3 (resolved): Figure 5 runbook augmentation vs observed manifests

- Default in code:
  - `train_primary_tabm.py` default is `--aug_type semantic`. [ref: scripts/Data_generation/train_primary_tabm.py :: parse_args :: `add_argument("--aug_type", default="semantic")`]
- Observed in saved run:
  - Figure 5 runbook optional example command shows `--aug_type semantic`. [ref: scripts/Data_generation/figure5_data_generation_runs.md :: optional train command :: `--aug_type semantic`]
  - Figure 5 external-validation manifest shows `used_aug_type: "semantic"`. [ref: predictions/Figure5_external_validation/Special_v1/predict_manifest.json :: key :: `used_aug_type`]
  - Figure 6 external/eclipse manifests show `used_aug_type: "semantic"`. [ref: predictions/external_data/non_sig_reduced/predict_manifest.json :: key :: `used_aug_type`; ref: predictions/external_data/sig_reduced/predict_manifest.json :: key :: `used_aug_type`; ref: predictions/eclipse_data/non_sig_reduced/predict_manifest.json :: key :: `used_aug_type`; ref: predictions/eclipse_data/sig_reduced/predict_manifest.json :: key :: `used_aug_type`]
- Decision for manuscript:
  - Use observed `semantic` augmentation for Figure 5i-m and Figure 6 uncertainty-application descriptions.
  - Treat this item as resolved alignment (runbook, code default, and manifests agree on `semantic`).

## Targeted verification outcomes for prior `[NEEDS CONFIRMATION]` items

### 1) Dependency versions pinning
- Verification result:
  - No lock/spec files were found for `requirements*.txt`, `environment*.yml|yaml`, `pyproject.toml`, `poetry.lock`, `uv.lock`, `conda-lock.yml|yaml`, or `Dockerfile*` in `code_repository`.
  - [NEEDS CONFIRMATION]
- Evidence:
  - Repository file scan output: all target patterns reported `=> 0` matches. [ref: code_repository repo scan :: file search patterns :: `requirements*.txt => 0` ... `Dockerfile* => 0`]
  - README only states dependency families without pinned versions. [ref: README.md :: Notes :: `External dependencies ... are required`]

### 2) MLflow provenance
- Verification result:
  - External MLflow stores exist at `/mnt/jh/delta/code/pkg/mlflow/jh/mlruns` and `/mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old`.
  - Figure 3-5 experiment IDs/runs referenced by mapping docs are present under `mlruns_old`.
  - Representative run tags resolve to the expected source scripts (`LLM_eval_script_250916.py`, `Feature_eval_script_250917.py`, `LOO_eval_script_251006.py`).
  - For the train/predict primary TabM path, provenance remains manifest-based (`ensemble_manifest.json`, `predict_manifest.json`) in model/prediction folders.
- Evidence:
  - External store roots and directories. [ref: /mnt/jh/delta/code/pkg/mlflow/jh :: directory listing :: `mlruns`, `mlruns_old`; ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns :: directory listing :: experiment dirs; ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old :: directory listing :: experiment dirs]
  - Experiment ID presence. [ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old/125845022789070498/meta.yaml :: key :: `experiment_id`; ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old/694949682285093199/meta.yaml :: key :: `experiment_id`; ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old/879354154507378166/meta.yaml :: key :: `experiment_id`; ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old/603034844447165657/meta.yaml :: key :: `experiment_id`]
  - Representative run source tags. [ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old/125845022789070498/e086e399acc745a0a12161edd198967b/tags/mlflow.source.name :: tag :: `LLM_eval_script_250916.py`; ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old/694949682285093199/2fa80c27dc814e759c29ded5e138aa5b/tags/mlflow.source.name :: tag :: `Feature_eval_script_250917.py`; ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old/879354154507378166/0d54cd002bd7434189fa13eb021c7bf1/tags/mlflow.source.name :: tag :: `LOO_eval_script_251006.py`; ref: /mnt/jh/delta/code/pkg/mlflow/jh/mlruns_old/603034844447165657/f8e68a5edff341e289e8281a5fe67f1a/tags/mlflow.source.name :: tag :: `LOO_eval_script_251006.py`]
  - Tracking URI in scripts. [ref: scripts/Data_generation/Feature_eval_script.py :: main :: `mlflow.set_tracking_uri("http://10.152.253.136:8085")`; ref: scripts/Data_generation/LLM_eval_script.py :: main :: `mlflow.set_tracking_uri("http://10.152.253.136:8085")`; ref: scripts/Data_generation/LOO_eval_script.py :: main :: `mlflow.set_tracking_uri("http://10.152.253.136:8085")`]
  - Train/predict manifest provenance artifacts. [ref: scripts/Data_generation/train_primary_tabm.py :: export block :: `ensemble_manifest.json`; ref: scripts/Data_generation/predict_primary_tabm.py :: output manifest block :: `predict_manifest.json`; ref: predictions/Figure5_external_validation/Special_v1/predict_manifest.json :: file :: `predict_manifest.json`]

### 3) `fold_manifests/figure3d_text_multiclass_compound_folds.json` existence
- Verification result:
  - `fold_manifests/figure3d_text_multiclass_compound_folds.json` is missing in current repo.
  - `fold_manifests/figure3d_text_compound_folds.json` exists.
  - Saved multiclass `meta.json` still points to the missing path.
  - [NEEDS CONFIRMATION]
- Evidence:
  - Fold-manifest directory listing. [ref: fold_manifests :: file inventory :: includes `figure3d_text_compound_folds.json`]
  - Missing-file check for requested path. [ref: fold_manifests :: existence check :: `MISSING: fold_manifests/figure3d_text_multiclass_compound_folds.json`]
  - Saved multiclass run metadata path. [ref: predictions/Text_vs_No_Text_experiment/multiclass/semantic_BioBERT_pytorch_mlp_Text_Embedding+Proteomics/meta.json :: key :: `folds_manifest`]
  - Figure 3 runbook uses the existing path. [ref: scripts/Data_generation/figure3_data_generation_runs.md :: Figure 3d command :: `--folds_manifest fold_manifests/figure3d_text_compound_folds.json`]
