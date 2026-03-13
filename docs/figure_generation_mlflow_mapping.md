# Figure generation provenance (MLflow)

MLflow root
- `/mnt/jh/delta/code/pkg/mlflow/jh/mlruns`

How to identify the generating script for a run
- `mlruns/<experiment_id>/<run_id>/tags/mlflow.source.name`

Scripts copied into `code_repository/scripts/Data_generation`
- `LLM_eval_script.py`
- `Feature_eval_script.py`
- `LOO_eval_script.py`

Figure 2
- Source data are preprocessed outputs under `Data/Base/*.pkl` in the Ms_paint workspace. No MLflow experiment was used for these data.

Figure 3
- Panel c: LLM/Aug PR AUC by method
- MLflow experiment: `125845022789070498` (`LLM_augmentation_experiment`)
- Script: `/mnt/nas/PhD/Code/Ms_paint/Codebase/LLM_eval_script_250916.py`
- Runs
- `semantic_BERT_pytorch_mlp_Text_Embedding+Proteomics` -> `1195cfa99814478cafd911f8dfddf8dc`
- `semantic_BioBERT_pytorch_mlp_Text_Embedding+Proteomics` -> `e086e399acc745a0a12161edd198967b`
- `semantic_PubChemDeBERTa_pytorch_mlp_Text_Embedding+Proteomics` -> `b18014ed683b487c8f82fef2386ae0c9`
- `sentence_BERT_pytorch_mlp_Text_Embedding+Proteomics` -> `f23bff90d80746b9b22cc5fcda7472e9`
- `sentence_BioBERT_pytorch_mlp_Text_Embedding+Proteomics` -> `385b7399ab3c48d3b6f2332160291e2b`
- `sentence_PubChemDeBERTa_pytorch_mlp_Text_Embedding+Proteomics` -> `39783464e0b14bfda78f28252dac298d`
- `word_BERT_pytorch_mlp_Text_Embedding+Proteomics` -> `e9028937c1954f319cd54295454096bc`
- `word_BioBERT_pytorch_mlp_Text_Embedding+Proteomics` -> `691d0469fc96404983648c06c756d579`
- `word_PubChemDeBERTa_pytorch_mlp_Text_Embedding+Proteomics` -> `9cca8210eec24de79c452bdbff606c43`

- Panel d: Text embeddings vs no text (LightGBM and PyTorch-MLP)
- MLflow experiments
- `837307800117189045` (original: `251110_5x5cv_text_prot_multiclass_2`, mirrored in repo as `Text_vs_No_Text_experiment/multiclass`)
- `368139422039444402` (original: `251110_5x5cv_no_text_prot_multiclass_2`, mirrored in repo as `Text_vs_No_Text_experiment/no_text`)
- Script: `/mnt/nas/PhD/Code/Ms_paint/Codebase/Feature_eval_script_250917.py`
- Runs
- `sentence_BioBERT_lightgbm_Text_Embedding+Proteomics` -> `33c5803178a747079d8a06f28a533a08`
- `sentence_BioBERT_pytorch_mlp_Text_Embedding+Proteomics` -> `9a0109c12fa040629a98aa9eb916f9b0`
- `no_text__lightgbm_Proteomics` -> `e008beaef911491fb11222bcbcf03935`
- `no_text__pytorch_mlp_Proteomics` -> `614dfc809f474cb7bc59bbde984f8524`

Figure 4
- Panels a, b, c: feature set performance, model comparisons, and exact Shapley
- MLflow experiment: `694949682285093199` (`250917_5x5cv_feature_binary`)
- Script: `/mnt/nas/PhD/Code/Ms_paint/Codebase/Feature_eval_script_250917.py`
- Runs (full modality set)
- `sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC` -> `2fa80c27dc814e759c29ded5e138aa5b`
- `sentence_BioBERT_pytorch_mlp_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC` -> `d36b99758dc34b6b93a2eeb0de02ff41`
- `sentence_BioBERT_lightgbm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC` -> `ee4c20d25a574ba7bc6eb3ba6ad345ba`
- `sentence_BioBERT_xgboost_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC` -> `2d3026d3455b46e2b28cba0545fbeda0`
- `sentence_BioBERT_mlp_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC` -> `e19bc4904f3a4aff978166d960d9d1f2`
- `sentence_BioBERT_rf_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC` -> `01a920fe05ef43daaaf69dca15ce8262`

- Panels f, h: LOO proteomics ablation (Temafloxacin, Gepotidacin)
- MLflow experiment: `879354154507378166` (`251028_tabm_LOO`)
- Script: `/mnt/nas/PhD/Code/Ms_paint/Codebase/LOO_eval_script_251006.py`
- Runs
- `sentence_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC` -> `0d54cd002bd7434189fa13eb021c7bf1`
- `sentence_BioBERT_tabm_Text_Embedding+ECFP+CellPaint+MIC+GC` -> `81c765474e0d470aa1fc545fc839c695`
- `sentence_BioBERT_tabm_Text_Embedding+ECFP+CellPaint+MIC+GC` -> `c8f915672d1449a3a0905e4ff6e4cc4f`

Figure 5
- Panels b, c: uncertainty method comparison
- MLflow experiments
- `879354154507378166` (`251028_tabm_LOO`) run `0d54cd002bd7434189fa13eb021c7bf1`
- `603034844447165657` (`251029_tabm_LOSO`) run `f8e68a5edff341e289e8281a5fe67f1a`
- Script for both runs: `/mnt/nas/PhD/Code/Ms_paint/Codebase/LOO_eval_script_251006.py`

- Panels i, j, k, m: external validation compounds
- Data products are under `predictions/251103_special_prediction_tabM_ensemble/Special_v1/` and are not tracked by MLflow.

Figure 6
- External validation data products are under `predictions/251126_external_data`, `predictions/251208_eclipse_data`, and `predictions/251209_eclipse_data` and are not tracked by MLflow.
