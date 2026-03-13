# Figure 5 Data Generation Runs (Canonical Semantic Setup)

Run from:

```bash
cd /mnt/jh/nas/PhD/Code/Ms_paint/code_repository
```

## Figure 5b/5c (eta-LOO uncertainty method comparison)

Panels `b` and `c` are built from LOO + LOSO prediction folders for the same TabM run:

- `semantic_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC`

### Prediction generation commands

```bash
PYTHONPATH=scripts/Data_generation \
python3 scripts/Data_generation/LOO_eval_script.py \
  --exp_name Uncertainty_experiment_semantic \
  --task binary_loo_ablation \
  --seed_repeats 5 \
  --base_seed 42 \
  --mc_passes 0
```

```bash
PYTHONPATH=scripts/Data_generation \
python3 scripts/Data_generation/LOO_eval_script.py \
  --exp_name Uncertainty_experiment_semantic \
  --task binary_loso_ablation \
  --seed_repeats 5 \
  --base_seed 42 \
  --mc_passes 0
```

### Prediction outputs used by the Figure 5 uncertainty notebook

- `predictions/Uncertainty_experiment_semantic/binary_loo_ablation/semantic_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC/`
- `predictions/Uncertainty_experiment_semantic/binary_loso_ablation/semantic_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC/`

### Analysis/export command (eta-LOO summaries + model export)

```bash
jupyter nbconvert --to notebook --execute notebooks/figure5/figure5_uncertainty_comparison.ipynb --inplace
```

### Analysis outputs used by Figure 5 combined

- `analysis_exports/uncertainty_estimator/eta_loo_overall_summary.csv`
- `analysis_exports/uncertainty_estimator/eta_loo_per_compound_metrics.csv`
- `analysis_exports/uncertainty_estimator/eta_loo_pairwise_stats_Holm.csv`
- `analysis_exports/uncertainty_estimator/eta_loo_pairwise_stats_BH.csv`
- `analysis_exports/uncertainty_estimator/eta_loo_omnibus_summary.txt`

### Deployed uncertainty model artifact

- `models/uncertainty_estimator/meta_model.joblib`
- `models/uncertainty_estimator/meta_artifact.json`

## Figure 5i/5j/5k/5m (external validation + uncertainty application)

Panels `i/j/k/m` are generated inside `notebooks/figure5/figure5_combined.ipynb` from:

- `predictions/Figure5_external_validation/Special_v1/`
- `models/uncertainty_estimator/`

### (Optional) Train primary TabM ensemble if model bundle is not present

```bash
PYTHONPATH=scripts/Data_generation \
python3 scripts/Data_generation/train_primary_tabm.py \
  --base_dir Data/Base \
  --llm_model BioBERT \
  --aug_type semantic \
  --all_modalities \
  --base_seed 42 \
  --num_seeds 5 \
  --export_dir models/primary_tabm_semantic_biobert_all_modalities
```

### Generate external-set predictions

```bash
PYTHONPATH=scripts/Data_generation \
python3 scripts/Data_generation/predict_primary_tabm.py \
  --model_dir models/primary_tabm_semantic_biobert_all_modalities \
  --base_dir Data/Special \
  --out_dir predictions/Figure5_external_validation/Special_v1 \
  --return_experts \
  --mc_passes 0
```

## Figure 6 threshold adaptation (same meta-model, proteomics target domain)

The base uncertainty model is retained (`models/uncertainty_estimator/`) and only the threshold is recalibrated on:

- `predictions/Uncertainty_experiment_semantic_proteomics/binary_loo_ablation/semantic_BioBERT_tabm_Text_Embedding+Proteomics/`
- `predictions/Uncertainty_experiment_semantic_proteomics/binary_loso_ablation/semantic_BioBERT_tabm_Text_Embedding+Proteomics/`

Run:

```bash
jupyter nbconvert --to notebook --execute notebooks/figure6/figure6_uncertainty_estimator_calibration.ipynb --inplace
```

This writes:

- `models/uncertainty_estimator_adapted/meta_model.joblib`
- `models/uncertainty_estimator_adapted/meta_artifact.json`

## Figure 5/6 combined plot generation

```bash
jupyter nbconvert --to notebook --execute notebooks/figure5/figure5_combined.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/figure6/figure6_combined.ipynb --inplace
```

Expected figure export locations:

- `images/figure5/figure5_combined.(png|pdf|svg)`
- `images/figure6/figure6_combined.(png|pdf|svg)`

## Notes

- The Figure 5 uncertainty comparison excludes the deprecated `Small margin` method.
- Canonical uncertainty artifact names are now:
  - baseline: `models/uncertainty_estimator`
  - adapted for Figure 6 proteomics-only transfer: `models/uncertainty_estimator_adapted`
