# Figure 3 Data Generation Runs

This file documents the exact CLI commands used in `code_repository` to regenerate data for Figure 3.

Run from:

```bash
cd /mnt/jh/nas/PhD/Code/Ms_paint/code_repository
```

## Figure 3c (LLM/Augmentation experiment)

### Prediction generation command

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

### Outputs produced

- `predictions/LLM_augmentation_experiment/binary/semantic_BERT_pytorch_mlp_Text_Embedding+Proteomics/`
- `predictions/LLM_augmentation_experiment/binary/semantic_BioBERT_pytorch_mlp_Text_Embedding+Proteomics/`
- `predictions/LLM_augmentation_experiment/binary/semantic_PubChemDeBERTa_pytorch_mlp_Text_Embedding+Proteomics/`
- `predictions/LLM_augmentation_experiment/binary/sentence_BERT_pytorch_mlp_Text_Embedding+Proteomics/`
- `predictions/LLM_augmentation_experiment/binary/sentence_BioBERT_pytorch_mlp_Text_Embedding+Proteomics/`
- `predictions/LLM_augmentation_experiment/binary/sentence_PubChemDeBERTa_pytorch_mlp_Text_Embedding+Proteomics/`
- `predictions/LLM_augmentation_experiment/binary/word_BERT_pytorch_mlp_Text_Embedding+Proteomics/`
- `predictions/LLM_augmentation_experiment/binary/word_BioBERT_pytorch_mlp_Text_Embedding+Proteomics/`
- `predictions/LLM_augmentation_experiment/binary/word_PubChemDeBERTa_pytorch_mlp_Text_Embedding+Proteomics/`
- `results/LLM_augmentation_experiment.pkl`

### Analysis/export command (for Figure 3c inputs + supplementary metrics)

```bash
jupyter nbconvert --to notebook --execute notebooks/figure3/figure3_llm_parametric_tests.ipynb --inplace
```

### Analysis outputs produced

- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/resample_metrics.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/rm_anova_input.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/method_descriptives.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/anova_summary.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/anova_pr_auc.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/anova_roc_auc.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/anova_mcc.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/tukey_pr_auc_pairs.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/tukey_pr_auc_means.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/tukey_pr_auc_mean_diffs.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/tukey_pr_auc_pvals_matrix.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/tukey_roc_auc_pairs.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/tukey_roc_auc_means.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/tukey_roc_auc_mean_diffs.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/tukey_roc_auc_pvals_matrix.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/tukey_mcc_pairs.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/tukey_mcc_means.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/tukey_mcc_mean_diffs.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/tukey_mcc_pvals_matrix.csv`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/methods_pairwise_BH_pr_auc.png`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/methods_pairwise_BH_roc_auc.png`
- `analysis_exports/LLM_augmentation_experiment_binary_pytorch_mlp_Text_Embedding-Proteomics/methods_pairwise_BH_mcc.png`

## Figure 3d (Text vs No-Text proteomics)

Use one experiment root with two task runs:

- `task=multiclass` (with text)
- `task=no_text` (without text)

### Prediction generation commands

```bash
PYTHONPATH=scripts/Data_generation \
python scripts/Data_generation/Feature_eval_script.py \
  --exp_name Text_vs_No_Text_experiment \
  --task multiclass \
  --cv_splits 5 \
  --seed_splits 1 \
  --cv_iterations 5 \
  --base_seed 42 \
  --folds_manifest fold_manifests/figure3d_text_compound_folds.json
```

```bash
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

### Expected prediction folders for Figure 3d

- `predictions/Text_vs_No_Text_experiment/multiclass/sentence_BioBERT_lightgbm_Text_Embedding+Proteomics/`
- `predictions/Text_vs_No_Text_experiment/multiclass/sentence_BioBERT_pytorch_mlp_Text_Embedding+Proteomics/`
- `predictions/Text_vs_No_Text_experiment/no_text/no_text__lightgbm_Proteomics/`
- `predictions/Text_vs_No_Text_experiment/no_text/no_text__pytorch_mlp_Proteomics/`

### Analysis/export command for Figure 3d tables

`notebooks/figure3/figure3_text_augmentation_effects.ipynb` now points to:

- `predictions/Text_vs_No_Text_experiment/multiclass`
- `predictions/Text_vs_No_Text_experiment/no_text`

Then execute:

```bash
jupyter nbconvert --to notebook --execute notebooks/figure3/figure3_text_augmentation_effects.ipynb --inplace
```

### Expected Figure 3d table outputs

- `analysis_exports/text_vs_notext_proteomics_grouped_stars/tidy_proteomics_cycles.csv`
- `analysis_exports/text_vs_notext_proteomics_grouped_stars/paired_stats_per_model.csv`
