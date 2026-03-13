# Reproducibility Guide (Figures 3-5, per data subplot)

This document maps each data-containing subplot in Figures 3-5 to:
- the data files loaded by the combined-figure notebooks,
- the generating script(s),
- the MLflow experiment/run IDs,
- the run parameters used.

MLflow root used for provenance:
- `/mnt/jh/delta/code/pkg/mlflow/jh/mlruns`

How script provenance was resolved:
- `mlruns/<experiment_id>/<run_id>/tags/mlflow.source.name`
- run parameters from `mlruns/<experiment_id>/<run_id>/params/*`

## Figure 3c - LLM/Augmentation impact on PR AUC

Notebook consumer:
- `code_repository/notebooks/figure3/figure3_combined.ipynb`
- loaded file: `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/rm_anova_input.csv`

Generating script and MLflow:
- script: `/mnt/nas/PhD/Code/Ms_paint/Codebase/LLM_eval_script_250916.py`
- experiment: `125845022789070498` (`LLM_augmentation_experiment`)
- runs used: 9 (BERT/BioBERT/PubChemDeBERTa x semantic/sentence/word; model `pytorch_mlp`; modalities `Text_Embedding+Proteomics`)

Fixed run parameters:
- `Task=binary`
- `DataSplit=BaseOnly`
- `Model_Type=pytorch_mlp`
- `Modalities=Text_Embedding+Proteomics`
- `CV_Splits=5`
- `Seed_Splits=1`
- `CV_Iterations=5`
- `Base_Seed=42`
- `Folds_Manifest=predictions/LLM_augmentation_experiment/folds/binary_compound_folds.json`

Variable run parameters:
- `LLM_Model in {BERT, BioBERT, PubChemDeBERTa}`
- `Augmentation in {semantic, sentence, word}`

Run IDs:
- `1195cfa99814478cafd911f8dfddf8dc` (`semantic_BERT_pytorch_mlp_Text_Embedding+Proteomics`)
- `e086e399acc745a0a12161edd198967b` (`semantic_BioBERT_pytorch_mlp_Text_Embedding+Proteomics`)
- `b18014ed683b487c8f82fef2386ae0c9` (`semantic_PubChemDeBERTa_pytorch_mlp_Text_Embedding+Proteomics`)
- `f23bff90d80746b9b22cc5fcda7472e9` (`sentence_BERT_pytorch_mlp_Text_Embedding+Proteomics`)
- `385b7399ab3c48d3b6f2332160291e2b` (`sentence_BioBERT_pytorch_mlp_Text_Embedding+Proteomics`)
- `39783464e0b14bfda78f28252dac298d` (`sentence_PubChemDeBERTa_pytorch_mlp_Text_Embedding+Proteomics`)
- `e9028937c1954f319cd54295454096bc` (`word_BERT_pytorch_mlp_Text_Embedding+Proteomics`)
- `691d0469fc96404983648c06c756d579` (`word_BioBERT_pytorch_mlp_Text_Embedding+Proteomics`)
- `9cca8210eec24de79c452bdbff606c43` (`word_PubChemDeBERTa_pytorch_mlp_Text_Embedding+Proteomics`)

## Figure 3d - Text vs no-text grouped Macro F1

Notebook consumer:
- `code_repository/notebooks/figure3/figure3_combined.ipynb`
- cache files (if present):
  - `analysis_exports/text_vs_notext_proteomics_grouped_stars/tidy_proteomics_cycles.csv`
  - `analysis_exports/text_vs_notext_proteomics_grouped_stars/paired_stats_per_model.csv`
- rebuild source if cache missing:
  - `predictions/Text_vs_No_Text_experiment/multiclass/<run_name>/*.pkl`
  - `predictions/Text_vs_No_Text_experiment/no_text/<run_name>/*.pkl`

Generating script and MLflow:
- script: `/mnt/nas/PhD/Code/Ms_paint/Codebase/Feature_eval_script_250917.py`
- experiments:
  - `837307800117189045` (original: `251110_5x5cv_text_prot_multiclass_2`)
  - `368139422039444402` (original: `251110_5x5cv_no_text_prot_multiclass_2`)

Runs used in panel:
- text/proteomics:
  - `33c5803178a747079d8a06f28a533a08` (`sentence_BioBERT_lightgbm_Text_Embedding+Proteomics`)
  - `9a0109c12fa040629a98aa9eb916f9b0` (`sentence_BioBERT_pytorch_mlp_Text_Embedding+Proteomics`)
- no-text/proteomics:
  - `e008beaef911491fb11222bcbcf03935` (`no_text__lightgbm_Proteomics`)
  - `614dfc809f474cb7bc59bbde984f8524` (`no_text__pytorch_mlp_Proteomics`)

Parameter set (text runs):
- `Task=multiclass`
- `DataSplit=BaseOnly`
- `LLM_Model=BioBERT`
- `Augmentation=sentence`
- `Modalities=Text_Embedding+Proteomics`
- `CV_Splits=5`, `Seed_Splits=1`, `CV_Iterations=5`, `Base_Seed=42`
- `Folds_Manifest=fold_manifests/figure3d_text_multiclass_compound_folds.json`

Parameter set (no-text runs):
- `Task=no_text`
- `DataSplit=BaseOnly`
- `LLM_Model=None`
- `Augmentation=None`
- `Modalities=Proteomics`
- `CV_Splits=5`, `Seed_Splits=1`, `Base_Seed=42`

## Figure 4a - TabM feature-ablation boxplot + dot grid (PR AUC)

Notebook consumer:
- `code_repository/notebooks/figure4/figure4_combined.ipynb`
- loaded files:
  - `analysis_all_models_250917_5x5cv_feature_binary_binary/tabm/pr_auc/tabm_per_cycle_pr_auc.csv`
  - `analysis_all_models_250917_5x5cv_feature_binary_binary/tabm/pr_auc/tabm_pairwise_pr_auc_pvals_BH.csv`
  - optional: `analysis_all_models_250917_5x5cv_feature_binary_binary/tabm/pr_auc/tabm_friedman_pr_auc.csv`

Generating script and MLflow:
- script: `/mnt/nas/PhD/Code/Ms_paint/Codebase/Feature_eval_script_250917.py`
- experiment: `694949682285093199` (`250917_5x5cv_feature_binary`)
- run set used: all TabM binary runs with `LLM_Model=BioBERT`, `Augmentation=sentence`, `DataSplit=BaseOnly`
- total runs in this panel set: `31`

Fixed run parameters:
- `Task=binary`
- `DataSplit=BaseOnly`
- `Model_Type=tabm`
- `LLM_Model=BioBERT`
- `Augmentation=sentence`
- `CV_Splits=5`
- `Seed_Splits=1`
- `CV_Iterations=5`
- `Base_Seed=42`
- `Folds_Manifest=predictions/250917_5x5cv_feature_binary/folds/binary_compound_folds.json`

Variable run parameter:
- `Modalities` = all feature combinations over `{Proteomics, ECFP, CellPaint, MIC, GC}` with `Text_Embedding` fixed.

Run IDs and modalities for Figure 4a:
- `a0aa9033eaae4b34b449aeeb231d3aa1` (`sentence_BioBERT_tabm_Text_Embedding+CellPaint`; `Modalities=Text_Embedding+CellPaint`)
- `2d72fa2970ac40fc82313f5953db2294` (`sentence_BioBERT_tabm_Text_Embedding+CellPaint+GC`; `Modalities=Text_Embedding+CellPaint+GC`)
- `106836de2d184e89a44e9c6436edf1ab` (`sentence_BioBERT_tabm_Text_Embedding+CellPaint+MIC`; `Modalities=Text_Embedding+CellPaint+MIC`)
- `0d790da031434a2cbedd882fb2c3e624` (`sentence_BioBERT_tabm_Text_Embedding+CellPaint+MIC+GC`; `Modalities=Text_Embedding+CellPaint+MIC+GC`)
- `8143fb27138e494b956c74e2661bccb9` (`sentence_BioBERT_tabm_Text_Embedding+ECFP`; `Modalities=Text_Embedding+ECFP`)
- `d7fd963f24c74674891c8c874813d292` (`sentence_BioBERT_tabm_Text_Embedding+ECFP+CellPaint`; `Modalities=Text_Embedding+ECFP+CellPaint`)
- `337a2d433aed45dbb243e79598d7a803` (`sentence_BioBERT_tabm_Text_Embedding+ECFP+CellPaint+GC`; `Modalities=Text_Embedding+ECFP+CellPaint+GC`)
- `6da4a38fe0004dfc8647c075e24cd60d` (`sentence_BioBERT_tabm_Text_Embedding+ECFP+CellPaint+MIC`; `Modalities=Text_Embedding+ECFP+CellPaint+MIC`)
- `f40a56a320c54418aa7afc1cd0677ab1` (`sentence_BioBERT_tabm_Text_Embedding+ECFP+CellPaint+MIC+GC`; `Modalities=Text_Embedding+ECFP+CellPaint+MIC+GC`)
- `52e81b23099e46b19129ba9b7e2a5579` (`sentence_BioBERT_tabm_Text_Embedding+ECFP+GC`; `Modalities=Text_Embedding+ECFP+GC`)
- `51efdd3ed68241688888aad453bfa033` (`sentence_BioBERT_tabm_Text_Embedding+ECFP+MIC`; `Modalities=Text_Embedding+ECFP+MIC`)
- `b9cc7380c94341fd92ce0856a1e64525` (`sentence_BioBERT_tabm_Text_Embedding+ECFP+MIC+GC`; `Modalities=Text_Embedding+ECFP+MIC+GC`)
- `1345cacb3f274cc6a3e053bc26918c62` (`sentence_BioBERT_tabm_Text_Embedding+GC`; `Modalities=Text_Embedding+GC`)
- `db7c3f7a37ff4aadbfc90f79498c76af` (`sentence_BioBERT_tabm_Text_Embedding+MIC`; `Modalities=Text_Embedding+MIC`)
- `2ddf532ed36c486399367631e72d0b7a` (`sentence_BioBERT_tabm_Text_Embedding+MIC+GC`; `Modalities=Text_Embedding+MIC+GC`)
- `2924fb3b579d4d5a87332934873a84e6` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics`; `Modalities=Text_Embedding+Proteomics`)
- `1cf446493fed405cb342273a102bcae5` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics+CellPaint`; `Modalities=Text_Embedding+Proteomics+CellPaint`)
- `8532393b221a4af7af03bb426ec5eb1d` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics+CellPaint+GC`; `Modalities=Text_Embedding+Proteomics+CellPaint+GC`)
- `09103ec5527145ed80fec90f7223a929` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics+CellPaint+MIC`; `Modalities=Text_Embedding+Proteomics+CellPaint+MIC`)
- `ce88da232d214a7ba64000110140cfbb` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics+CellPaint+MIC+GC`; `Modalities=Text_Embedding+Proteomics+CellPaint+MIC+GC`)
- `e97bb23d6b8548feb2787cebb4449e72` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP`; `Modalities=Text_Embedding+Proteomics+ECFP`)
- `2c241af67de34a9593a80391f12da8c2` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint`; `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint`)
- `9207e7049d3f4e11bdbf50c35e356b45` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+GC`; `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint+GC`)
- `c1c28502d1eb4b73a52705b25b6d12b4` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC`; `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint+MIC`)
- `2fa80c27dc814e759c29ded5e138aa5b` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`; `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`)
- `9da1838f2428492db7d7025de7824585` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+GC`; `Modalities=Text_Embedding+Proteomics+ECFP+GC`)
- `d2b529f3ccbf497782906b043a41479f` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+MIC`; `Modalities=Text_Embedding+Proteomics+ECFP+MIC`)
- `1e9288bd14e44dd0ac755cb16b265e7c` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+MIC+GC`; `Modalities=Text_Embedding+Proteomics+ECFP+MIC+GC`)
- `c6ccae4bb95a457980844603798df9e9` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics+GC`; `Modalities=Text_Embedding+Proteomics+GC`)
- `b2337256f08b4c41bf336afdcce3a998` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics+MIC`; `Modalities=Text_Embedding+Proteomics+MIC`)
- `1ab1c66ef10a43a18811f81523abd16f` (`sentence_BioBERT_tabm_Text_Embedding+Proteomics+MIC+GC`; `Modalities=Text_Embedding+Proteomics+MIC+GC`)

## Figure 4b - Grouped model comparison (global-best style, PR AUC)

Notebook consumer:
- `code_repository/notebooks/figure4/figure4_combined.ipynb`
- internal selector in plotting function:
  - models: `rf, lightgbm, xgboost, mlp, pytorch_mlp, tabm`
  - feature sets:
    - `CellPaint+ECFP+Proteomics`
    - `CellPaint+ECFP+MIC+Proteomics`
    - `CellPaint+ECFP+GC+MIC+Proteomics`
- source folder: `analysis_all_models_250917_5x5cv_feature_binary_binary`

Generating script and MLflow:
- script: `/mnt/nas/PhD/Code/Ms_paint/Codebase/Feature_eval_script_250917.py`
- experiment: `694949682285093199`
- run set used: `18` runs (6 models x 3 feature sets)

Fixed run parameters:
- `Task=binary`, `DataSplit=BaseOnly`, `LLM_Model=BioBERT`, `Augmentation=sentence`
- `CV_Splits=5`, `Seed_Splits=1`, `CV_Iterations=5`, `Base_Seed=42`
- `Folds_Manifest=predictions/250917_5x5cv_feature_binary/folds/binary_compound_folds.json`

Run IDs for Figure 4b:
- `ee4c20d25a574ba7bc6eb3ba6ad345ba` (`model=lightgbm`, `feature_group=CP+ECFP+GC+MIC+Prot`, `run=sentence_BioBERT_lightgbm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`)
- `51491b4a6a614a6a9d52ab5a7096eb71` (`model=lightgbm`, `feature_group=CP+ECFP+MIC+Prot`, `run=sentence_BioBERT_lightgbm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint+MIC`)
- `a78d746ff40f4efb9207b77f20269406` (`model=lightgbm`, `feature_group=CP+ECFP+Prot`, `run=sentence_BioBERT_lightgbm_Text_Embedding+Proteomics+ECFP+CellPaint`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint`)
- `e19bc4904f3a4aff978166d960d9d1f2` (`model=mlp`, `feature_group=CP+ECFP+GC+MIC+Prot`, `run=sentence_BioBERT_mlp_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`)
- `808c80ea3fef4055be0ea564e7a979b7` (`model=mlp`, `feature_group=CP+ECFP+MIC+Prot`, `run=sentence_BioBERT_mlp_Text_Embedding+Proteomics+ECFP+CellPaint+MIC`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint+MIC`)
- `b6cbde5a77b047e8b4916cf596b83cb5` (`model=mlp`, `feature_group=CP+ECFP+Prot`, `run=sentence_BioBERT_mlp_Text_Embedding+Proteomics+ECFP+CellPaint`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint`)
- `d36b99758dc34b6b93a2eeb0de02ff41` (`model=pytorch_mlp`, `feature_group=CP+ECFP+GC+MIC+Prot`, `run=sentence_BioBERT_pytorch_mlp_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`)
- `f81e6594b15d41e7a959d1a7fbf2c75e` (`model=pytorch_mlp`, `feature_group=CP+ECFP+MIC+Prot`, `run=sentence_BioBERT_pytorch_mlp_Text_Embedding+Proteomics+ECFP+CellPaint+MIC`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint+MIC`)
- `d360ef5d29ea4a548de770a99afe01cf` (`model=pytorch_mlp`, `feature_group=CP+ECFP+Prot`, `run=sentence_BioBERT_pytorch_mlp_Text_Embedding+Proteomics+ECFP+CellPaint`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint`)
- `01a920fe05ef43daaaf69dca15ce8262` (`model=rf`, `feature_group=CP+ECFP+GC+MIC+Prot`, `run=sentence_BioBERT_rf_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`)
- `d94ca9fda7e643188976eadb7a64d158` (`model=rf`, `feature_group=CP+ECFP+MIC+Prot`, `run=sentence_BioBERT_rf_Text_Embedding+Proteomics+ECFP+CellPaint+MIC`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint+MIC`)
- `73bddc335e76468da275ba10a4019584` (`model=rf`, `feature_group=CP+ECFP+Prot`, `run=sentence_BioBERT_rf_Text_Embedding+Proteomics+ECFP+CellPaint`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint`)
- `2fa80c27dc814e759c29ded5e138aa5b` (`model=tabm`, `feature_group=CP+ECFP+GC+MIC+Prot`, `run=sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`)
- `c1c28502d1eb4b73a52705b25b6d12b4` (`model=tabm`, `feature_group=CP+ECFP+MIC+Prot`, `run=sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint+MIC`)
- `2c241af67de34a9593a80391f12da8c2` (`model=tabm`, `feature_group=CP+ECFP+Prot`, `run=sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint`)
- `2d3026d3455b46e2b28cba0545fbeda0` (`model=xgboost`, `feature_group=CP+ECFP+GC+MIC+Prot`, `run=sentence_BioBERT_xgboost_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`)
- `7df8fd66d4b44d48948eeb36fe8ed802` (`model=xgboost`, `feature_group=CP+ECFP+MIC+Prot`, `run=sentence_BioBERT_xgboost_Text_Embedding+Proteomics+ECFP+CellPaint+MIC`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint+MIC`)
- `0e3cd9041e514c35ac3b6adbaa0c4309` (`model=xgboost`, `feature_group=CP+ECFP+Prot`, `run=sentence_BioBERT_xgboost_Text_Embedding+Proteomics+ECFP+CellPaint`, `Modalities=Text_Embedding+Proteomics+ECFP+CellPaint`)

## Figure 4c - Exact Shapley (PR AUC)

Notebook consumer:
- `code_repository/notebooks/figure4/figure4_combined.ipynb`
- loaded file: `analysis_tabm_exact_shap/tabm_shapley_exact_pr_auc.csv`

Generation linkage:
- this Shapley output is tied to the full-feature TabM configuration from Figure 4a/4b:
  - run ID: `2fa80c27dc814e759c29ded5e138aa5b`
  - run name: `sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`
  - script: `/mnt/nas/PhD/Code/Ms_paint/Codebase/Feature_eval_script_250917.py`
- exact Shapley computation itself is a downstream analysis artifact in `analysis_tabm_exact_shap` (not logged as a separate MLflow run in the tracked set above).

## Figure 4f - Temafloxacin proteomics ablation effect

Notebook consumer:
- `code_repository/notebooks/figure4/figure4_combined.ipynb`
- function `plot_compound_proteomics_effect(ax_f, 'JH27', ...)`
- source folder pattern:
  - `predictions/251028_tabm_LOO/binary_loo_ablation/sentence_BioBERT_tabm_Text_Embedding+ECFP+CellPaint+MIC+GC/*.pkl`
  - `predictions/251028_tabm_LOO/binary_loo_ablation/sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC/*.pkl`

Generating script and MLflow:
- script: `/mnt/nas/PhD/Code/Ms_paint/Codebase/LOO_eval_script_251006.py`
- experiment: `879354154507378166` (`251028_tabm_LOO`)

Run parameters:
- with proteomics run ID: `0d54cd002bd7434189fa13eb021c7bf1`
- no-proteomics run IDs (duplicate run names in MLflow history):
  - `81c765474e0d470aa1fc545fc839c695` (later)
  - `c8f915672d1449a3a0905e4ff6e4cc4f` (earlier)
- shared params:
  - `Task=binary_loo_ablation`
  - `DataSplit=LOO`
  - `Model_Type=tabm`
  - `LLM_Model=BioBERT`
  - `Augmentation=sentence`
  - `Base_Seed=42`
  - `Seed_Repeats=5`
  - `mc_passes=0`
  - `Return_Experts=True`

## Figure 4h - Gepotidacin proteomics ablation effect

Notebook consumer:
- `code_repository/notebooks/figure4/figure4_combined.ipynb`
- function `plot_compound_proteomics_effect(ax_h, 'JH32', ...)`
- data provenance is identical to Figure 4f (same two run folders; different `compound_id`).

## Figure 5b - eta-LOO uncertainty coverage vs error recall

Notebook consumer:
- `code_repository/notebooks/figure5/figure5_combined.ipynb`
- loaded files:
  - `analysis_exports/uncertainty_estimator/eta_loo_overall_summary.csv`
  - `analysis_exports/uncertainty_estimator/eta_loo_per_compound_metrics.csv`
- plotting behavior:
  - panel b excludes composite OR methods:
    - `Small margin OR high exp. entropy`
    - `Low MSP OR high pred. entropy`

Upstream generating runs and script:
- script: `/mnt/nas/PhD/Code/Ms_paint/Codebase/LOO_eval_script_251006.py`
- experiments used to create prediction inputs aggregated in this panel:
  - `879354154507378166` (`251028_tabm_LOO`)
  - `603034844447165657` (`251029_tabm_LOSO`)
- run IDs (with/without proteomics for both split types):
  - LOO + proteomics: `0d54cd002bd7434189fa13eb021c7bf1`
  - LOO + no-proteomics: `81c765474e0d470aa1fc545fc839c695` (duplicate historical alt: `c8f915672d1449a3a0905e4ff6e4cc4f`)
  - LOSO + proteomics: `f8e68a5edff341e289e8281a5fe67f1a`
  - LOSO + no-proteomics: `8899f1050e5649d1a079d02c4a495a65`

Shared run params (LOO/LOSO sets):
- `Model_Type=tabm`
- `LLM_Model=BioBERT`
- `Augmentation=sentence`
- `Base_Seed=42`
- `Seed_Repeats=5`
- `mc_passes=0`
- `Return_Experts=True`
- task/split vary:
  - LOO: `Task=binary_loo_ablation`, `DataSplit=LOO`
  - LOSO: `Task=binary_loso_ablation`, `DataSplit=LOSO`

## Figure 5c - pairwise significance vs best uncertainty method

Notebook consumer:
- `code_repository/notebooks/figure5/figure5_combined.ipynb`
- loaded file:
  - `analysis_exports/uncertainty_estimator/eta_loo_pairwise_stats_BH.csv`
- if missing, panel C recomputes pairwise tests in-notebook from `wide_clean` built from `eta_loo_per_compound_metrics.csv`.
- plotting behavior:
  - panel c uses `primary_metric="err_recall"` and excludes the same two composite OR methods as panel b.
  - pairwise multiplicity correction is Holm (stored in `p_adj_Holm`; `p_adj_BH` is retained as a compatibility alias).

Upstream run provenance:
- identical run set and parameters as Figure 5b.

## Figure 5i - meta-model uncertainty across compounds

Notebook consumer:
- `code_repository/notebooks/figure5/figure5_combined.ipynb`
- runtime object construction:
  - `UncertaintyEstimator(pred_root='predictions/251103_special_prediction_tabM_ensemble/Special_v1/', meta_dir='models/uncertainty_estimator/', out_dir='predictions/251103_special_prediction_tabM_ensemble/Special_v1/uncertainty_meta_apply', abs_min_score_floor=0.05, score_floor_factor=0.1, margin_floor=0.01)`
- plotted compounds: `JH68`, `JH51`, `JH53`, `JH66`, `JH67` (mapped to names in notebook).

Script linkage:
- panel source uses `Codebase/uncertainty_estimator.py` API behavior.
- supporting prediction inputs come from `pred_root` artifacts.

MLflow note:
- this `Special_v1` artifact set is not directly represented as a dedicated MLflow run in the experiments enumerated above.

## Figure 5j - top-3 predicted MoAs per compound

Notebook consumer:
- `code_repository/notebooks/figure5/figure5_combined.ipynb`
- inputs:
  - uncertainty outputs from the same `UncertaintyEstimator` run as panel i
  - `pairwise_csv = pred_root / 'pairwise_predictions.csv'`

Script linkage and parameters:
- same as Figure 5i.

## Figure 5k - waterfall contributions (Cystobactamid / JH68)

Notebook consumer:
- `code_repository/notebooks/figure5/figure5_combined.ipynb`
- function call: `plot_waterfall_ax(ax_k, ue_all_modalities, 'JH68', ...)`
- uses:
  - uncertainty output table (`ue.out`)
  - logistic-risk pipeline extracted from `meta_dir` via `ue._extract_inner_lr_pipeline()`

Script linkage and parameters:
- same `UncertaintyEstimator` configuration as Figure 5i.

## Figure 5m - waterfall contributions (Nitroxoline / JH53)

Notebook consumer:
- `code_repository/notebooks/figure5/figure5_combined.ipynb`
- function call: `plot_waterfall_ax(ax_m, ue_all_modalities, 'JH53', ...)`

Script linkage and parameters:
- identical to Figure 5k; only `compound_id` differs.

---

## Quick verification commands

Example: verify generating script for a run:
- `cat /mnt/jh/delta/code/pkg/mlflow/jh/mlruns/<exp_id>/<run_id>/tags/mlflow.source.name`

Example: list run parameters:
- `ls /mnt/jh/delta/code/pkg/mlflow/jh/mlruns/<exp_id>/<run_id>/params`
- `cat /mnt/jh/delta/code/pkg/mlflow/jh/mlruns/<exp_id>/<run_id>/params/<ParamName>`

---

## End-to-end regeneration (results tables, not only figure rendering)

Run from repository root: `code_repository/`.

### Figure 3c (panel c data)

1. Generate raw prediction artifacts:
```bash
PYTHONPATH=scripts/Data_generation \
python scripts/Data_generation/LLM_eval_script.py \
  --exp_name LLM_augmentation_experiment \
  --task binary \
  --cv_splits 5 \
  --seed_splits 1 \
  --cv_iterations 5 \
  --base_seed 42 \
  --folds_manifest fold_manifests/figure3c_compound_folds.json
```
2. Generate analysis tables consumed by `figure3_combined.ipynb`:
```bash
jupyter nbconvert --to notebook --execute notebooks/figure3/figure3_llm_parametric_tests.ipynb --inplace
```
3. Expected result file for panel c:
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/rm_anova_input.csv`

### Figure 3d (panel d data)

1. Generate text/proteomics and no-text/proteomics prediction artifacts:
```bash
PYTHONPATH=scripts/Data_generation \
python scripts/Data_generation/Feature_eval_script.py \
  --exp_name Text_vs_No_Text_experiment \
  --task multiclass \
  --cv_splits 5 \
  --seed_splits 1 \
  --cv_iterations 5 \
  --base_seed 42 \
  --folds_manifest fold_manifests/figure3d_text_multiclass_compound_folds.json

PYTHONPATH=scripts/Data_generation \
python scripts/Data_generation/Feature_eval_script.py \
  --exp_name Text_vs_No_Text_experiment \
  --task no_text \
  --cv_splits 5 \
  --seed_splits 1 \
  --cv_iterations 5 \
  --base_seed 42 \
  --folds_manifest fold_manifests/figure3d_no_text_compound_folds.json
```
2. Generate grouped stats tables:
```bash
jupyter nbconvert --to notebook --execute notebooks/figure3/figure3_text_augmentation_effects.ipynb --inplace
```
3. Expected result files for panel d:
- `analysis_exports/text_vs_notext_proteomics_grouped_stars/tidy_proteomics_cycles.csv`
- `analysis_exports/text_vs_notext_proteomics_grouped_stars/paired_stats_per_model.csv`

### Figure 4a and 4b (panel a,b data)

1. Generate TabM ablation runs (31 feature combinations for panel a):
```bash
python scripts/Data_generation/Feature_eval_script_250917.py \
  --exp_name 250917_5x5cv_feature_binary \
  --task binary \
  --cv_splits 5 \
  --seed_splits 1 \
  --cv_iterations 5 \
  --base_seed 42 \
  --preset fig4_tabm_ablation \
  --folds_manifest predictions/250917_5x5cv_feature_binary/folds/binary_compound_folds.json
```
2. Generate cross-model top-3 feature-set runs (panel b):
```bash
python scripts/Data_generation/Feature_eval_script_250917.py \
  --exp_name 250917_5x5cv_feature_binary \
  --task binary \
  --cv_splits 5 \
  --seed_splits 1 \
  --cv_iterations 5 \
  --base_seed 42 \
  --preset fig4_crossmodel_top3 \
  --folds_manifest predictions/250917_5x5cv_feature_binary/folds/binary_compound_folds.json
```
3. Aggregate predictions into result tables:
```bash
jupyter nbconvert --to notebook --execute notebooks/figure4/figure4_feature_replot.ipynb --inplace
```
4. Expected result files:
- `analysis_all_models_250917_5x5cv_feature_binary_binary/tabm/pr_auc/tabm_per_cycle_pr_auc.csv`
- `analysis_all_models_250917_5x5cv_feature_binary_binary/tabm/pr_auc/tabm_pairwise_pr_auc_pvals_BH.csv`
- `analysis_all_models_250917_5x5cv_feature_binary_binary/cross_model_on_topsets/` (panel b source tables)

### Figure 4c (panel c data)

1. Ensure full TabM prediction run exists:
- `predictions/250917_5x5cv_feature_binary/binary/sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC/`
2. Compute exact Shapley table:
```bash
jupyter nbconvert --to notebook --execute notebooks/figure4/figure4_shapley_analysis.ipynb --inplace
```
3. Expected result file:
- `analysis_tabm_exact_shap/tabm_shapley_exact_pr_auc.csv`

### Figure 4f and 4h (panel f,h data)

1. Generate LOO ablation predictions (with and without proteomics):
```bash
python scripts/Data_generation/LOO_eval_script_251006.py \
  --exp_name 251028_tabm_LOO \
  --task binary_loo_ablation \
  --seed_repeats 5 \
  --base_seed 42 \
  --mc_passes 0
```
2. Optional per-compound stats export:
```bash
jupyter nbconvert --to notebook --execute notebooks/figure4/figure4_proteomics_ablation_effects.ipynb --inplace
```
3. Panel `f/h` in `figure4_combined.ipynb` are built directly from:
- `predictions/251028_tabm_LOO/binary_loo_ablation/sentence_BioBERT_tabm_Text_Embedding+ECFP+CellPaint+MIC+GC/`
- `predictions/251028_tabm_LOO/binary_loo_ablation/sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC/`

### Figure 5b and 5c (panel b,c data)

1. Ensure LOO and LOSO prediction runs exist:
```bash
python scripts/Data_generation/LOO_eval_script_251006.py \
  --exp_name 251028_tabm_LOO \
  --task binary_loo_ablation \
  --seed_repeats 5 \
  --base_seed 42 \
  --mc_passes 0

python scripts/Data_generation/LOO_eval_script_251006.py \
  --exp_name 251029_tabm_LOSO \
  --task binary_loso_ablation \
  --seed_repeats 5 \
  --base_seed 42 \
  --mc_passes 0
```
2. Build uncertainty method comparison tables:
```bash
jupyter nbconvert --to notebook --execute notebooks/figure5/figure5_uncertainty_comparison.ipynb --inplace
```
3. Expected result files:
- `analysis_exports/uncertainty_estimator/eta_loo_overall_summary.csv`
- `analysis_exports/uncertainty_estimator/eta_loo_per_compound_metrics.csv`
- `analysis_exports/uncertainty_estimator/eta_loo_pairwise_stats_BH.csv`

### Figure 5i, 5j, 5k, 5m (panel i,j,k,m data)

1. Train base TabM ensemble and generate special-set predictions:
```bash
python scripts/Data_generation/train_primary_tabm.py [args used in your training run]
python scripts/Data_generation/predict_primary_tabm.py [args used for Special_v1]
```
2. Apply meta uncertainty model and export compound-level outputs:
```bash
jupyter nbconvert --to notebook --execute notebooks/figure5/figure5_external_validation.ipynb --inplace
```
3. Panel `i/j/k/m` consume:
- `predictions/251103_special_prediction_tabM_ensemble/Special_v1/`
- `predictions/251103_special_prediction_tabM_ensemble/Special_v1/uncertainty_meta_apply/`
- `models/uncertainty_estimator/`
