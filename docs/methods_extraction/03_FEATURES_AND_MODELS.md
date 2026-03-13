# 03_FEATURES_AND_MODELS

## Baseline candidate inventory (Step 1)
- **Most likely baseline in main training/prediction path:** TabM pairwise classifier trained/exported by `train_primary_tabm.py` and consumed by `predict_primary_tabm.py`. [ref: code_repository/scripts/Data_generation/train_primary_tabm.py :: main :: `model = get_model("tabm", ...)`]
- Supporting evidence: saved ensemble configs under `models/tabM_*` all report `"model_type": "tabm"`; prediction manifests reference these TabM ensemble roots. [ref: code_repository/models/tabM_all_modalities/seed_42/config.json :: config key :: `"model_type": "tabm"`]
- **Task-specific alternate baseline:** `pytorch_mlp` in LLM augmentation experiment where `LLM_eval_script.py` restricts to that model only. [ref: code_repository/scripts/Data_generation/LLM_eval_script.py :: model config :: `model_names = ["pytorch_mlp"]`]
- **Comparative model panel (not single baseline):** feature screen includes `tabm`, `pytorch_mlp`, `mlp`, `xgboost`, `lightgbm`, `rf`. [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: model config :: `model_names = [...]`]

## Feature modalities and computation
- Modalities available in modeling paths: `Text_Embedding`, `Proteomics`, `ECFP`, `CellPaint`, `MIC`, `GC`. [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: modal_dims :: keys]
- Pairwise feature rows include text embedding plus selected sample-level modality vectors concatenated in order. [ref: code_repository/scripts/Data_generation/Codebase/train_utils.py :: assemble_pairwise_training_table :: `X = build_features(backbone_df, selected_modalities)`]
- Text modality comes from embedding tables filtered by model name (`Model == llm_model`) and grouped by `MoA`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: generate_backbone_df :: `df = text_df[text_df["Model"] == model_name]`]
- Primary model export persists modality order in `feature_order.json` / `config.json` for inference-time reconstruction. [ref: code_repository/scripts/Data_generation/Codebase/train_utils.py :: export_model_bundle :: `feature_order.json`]

## Model types, objectives, and training behavior

### `tabm` (TabMWrapper)
- Construction defaults: `n_epochs=100`, `batch_size=256`, `patience=3`, `lr=1e-4`, `weight_decay=3e-4`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: TabMWrapper.__init__ :: parameter defaults]
- Architecture settings passed to `Model`: MLP backbone `n_blocks=3`, `d_block=512`, `dropout=0.1`, `arch_type='tabm'`, `k=32`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: TabMWrapper.fit :: `backbone={...}`]
- Loss/objective: cross-entropy over expert outputs (`F.cross_entropy(y_pred.flatten(0,1), y_true.repeat_interleave(k))`). [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: TabMWrapper.fit :: `loss_fn`]
- Early stopping: manual patience on epoch loss (`patience_left`). [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: TabMWrapper.fit :: `if patience_left <= 0: break`]
- Internal normalization: `QuantileTransformer(output_distribution='normal')` fit on train inputs. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: TabMWrapper.__init__/fit :: `self.scaler.fit_transform`]

### `pytorch_mlp` (SklearnLikeLightningMLP)
- Defaults from factory: `batch_size=128`, `max_epochs=40`, `width_list=[512,256]`, `depth=2`, `lr=1e-4`, `dropout=0.5`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: get_model :: `elif name == "pytorch_mlp"`]
- Binary loss: `BCEWithLogitsLoss`; multiclass loss: `CrossEntropyLoss`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: MLPPredictor.training_step :: `if self.n_classes == 2`]
- Early stopping: Lightning callback monitors `train_loss` with `patience=5`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: SklearnLikeLightningMLP.fit :: `EarlyStopping(monitor="train_loss", patience=5)`]

### Classical models
- LightGBM: `n_estimators=200`; binary uses `class_weight='balanced'`; multiclass sets `objective='multiclass'`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: get_model :: `if name == "lightgbm"`]
- XGBoost: `n_estimators=300`; binary eval metric `logloss`; multiclass objective `multi:softprob`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: get_model :: `elif name == "xgboost"`]
- sklearn MLP: `hidden_layer_sizes=(256,128)`, `max_iter=400`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: get_model :: `elif name == "mlp"`]
- Random Forest: `n_estimators=400`; binary uses `class_weight='balanced'`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: get_model :: `elif name == "rf"`]

## Class weighting and thresholds
- Binary decision threshold in evaluation/prediction is `0.5` (`proba >= 0.5` or `proba[:,1] > 0.5`). [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: evaluate_model_LOSO :: `y_pred = (proba >= 0.5).astype(int)`]
- Explicit class weighting is present only for binary LightGBM/RF factory settings; TabM and Lightning-MLP losses do not specify class weights. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: get_model :: `class_weight='balanced'`]

## Hyperparameter source and precedence
- Source hierarchy in scripted runs:
1. CLI args (`--cv_splits`, `--seed_splits`, `--base_seed`, `--task`, etc.). [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: parser :: CLI args]
2. Script-level fixed grids/constants (`model_names`, `llm_models`, `text_aug`, modality sets). [ref: code_repository/scripts/Data_generation/LLM_eval_script.py :: config block :: `llm_models  = ["PubChemDeBERTa", "BERT", "BioBERT"]`]
3. Model-factory defaults inside `get_model` and wrappers. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: get_model :: defaults]

## Code defaults vs observed run artifacts (discrepancies and alignment notes)
- `Feature_eval_script.py` defaults `seed_splits=5`, `cv_iterations=1`; saved run metadata for `Feature_augmentation_experiment` records `seed_splits=1`, `cv_iterations=5` (CLI override). [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: parser :: `default=5`, `default=1`]
- Same discrepancy for `LLM_eval_script.py` defaults vs saved `LLM_augmentation_experiment` `meta.json` (`1` seed split, `5` CV iterations). [ref: code_repository/scripts/Data_generation/LLM_eval_script.py :: parser :: `seed_splits`, `cv_iterations`]
- `predict_primary_tabm.py` default is `return_experts=False`; saved `predict_manifest.json` files show `"return_experts": true` (CLI override). [ref: code_repository/scripts/Data_generation/predict_primary_tabm.py :: parse_args :: `--return_experts`]
- `figure5_data_generation_runs.md` optional training command and observed Figure 5/6 prediction manifests all specify semantic augmentation for uncertainty application paths. [ref: code_repository/scripts/Data_generation/figure5_data_generation_runs.md :: Optional train command :: `--aug_type semantic`; ref: code_repository/predictions/Figure5_external_validation/Special_v1/predict_manifest.json :: key :: `used_aug_type`; ref: code_repository/predictions/external_data/non_sig_reduced/predict_manifest.json :: key :: `used_aug_type`; ref: code_repository/predictions/eclipse_data/non_sig_reduced/predict_manifest.json :: key :: `used_aug_type`]
- `Text_vs_No_Text` multiclass meta files reference `fold_manifests/figure3d_text_multiclass_compound_folds.json`; only `figure3d_text_compound_folds.json` is present in `code_repository/fold_manifests/`. [NEEDS CONFIRMATION]. [ref: code_repository/predictions/Text_vs_No_Text_experiment/multiclass/semantic_BioBERT_pytorch_mlp_Text_Embedding+Proteomics/meta.json :: key :: `folds_manifest`]
