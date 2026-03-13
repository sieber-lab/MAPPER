# Traceability: Figures -> Code -> Artifacts

This document maps the current manuscript figures to the notebooks and artifact roots used in the final pipeline.

## General note

Tracked in this public repository:

- code
- LaTeX sources
- curated analysis exports
- figure assets

Often local-only and ignored from public git history:

- `predictions/`
- `models/`
- `results/`

## Figure 2

- Notebooks:
  - `notebooks/figure2/figure2_data_overview.ipynb`
  - `notebooks/figure2/figure2_combined.ipynb`
- Inputs:
  - `Data/Base/X_proteomics.pkl`
  - `Data/Base/X_ecfp.pkl`
  - `Data/Base/X_cellpaint.pkl`
  - `Data/Base/X_mic.pkl`
  - `Data/Base/X_gc.pkl`
  - `Data/Base/y_labels.pkl`
- Final asset:
  - `images/main_figures/figure2.pdf`

## Figure 3

- Notebooks:
  - `notebooks/figure3/figure3_llm_parametric_tests.ipynb`
  - `notebooks/figure3/figure3_text_augmentation_effects.ipynb`
  - `notebooks/figure3/figure3_combined.ipynb`
- Inputs:
  - `predictions/LLM_augmentation_experiment/binary/*`
  - `predictions/Text_vs_No_Text_experiment/multiclass/*`
  - `predictions/Text_vs_No_Text_experiment/no_text/*`
  - `analysis_exports/LLM_augmentation_experiment*`
  - `analysis_exports/text_vs_notext_experiment/*`
  - `analysis_exports/text_vs_notext_proteomics_grouped_stars/*`
- Final asset:
  - `images/main_figures/figure3.pdf`
- Supplementary asset:
  - `images/supplementary/fig3_SI.pdf`

## Figure 4

- Notebooks:
  - `notebooks/figure4/figure4_best_feature_combination.ipynb`
  - `notebooks/figure4/figure4_feature_replot.ipynb`
  - `notebooks/figure4/figure4_shapley_analysis.ipynb`
  - `notebooks/figure4/figure4_structural_novelty_modality_jumps.ipynb`
  - `notebooks/figure4/figure4_combined.ipynb`
- Inputs:
  - `predictions/Feature_augmentation_experiment/binary/semantic_BioBERT_*`
  - `analysis_exports/Feature_augmentation_experiment/*`
  - `analysis_exports/figure4_novelty_modality_jumps/*`
- Final asset:
  - `images/main_figures/figure4.pdf`
- Supplementary asset:
  - `images/supplementary/fig4_SI.pdf`

## Figure 5

- Notebooks:
  - `notebooks/figure5/figure5_uncertainty_comparison.ipynb`
  - `notebooks/figure5/figure5_combined.ipynb`
- Inputs:
  - `predictions/Uncertainty_experiment_semantic/binary_loo_ablation/*`
  - `predictions/Uncertainty_experiment_semantic/binary_loso_ablation/*`
  - `predictions/Figure5_external_validation/Special_v1/*`
  - `analysis_exports/uncertainty_estimator/*`
  - `models/uncertainty_estimator/*`
- Final asset:
  - `images/main_figures/figure5.pdf`

## Figure 6

- Notebooks:
  - `notebooks/figure6/figure6_uncertainty_estimator_calibration.ipynb`
  - `notebooks/figure6/figure6_eclipse_instrument.ipynb`
  - `notebooks/figure6/figure6_external_lab.ipynb`
  - `notebooks/figure6/figure6_combined.ipynb`
- Inputs:
  - `predictions/eclipse_data/non_sig_reduced/*`
  - `predictions/eclipse_data/sig_reduced/*`
  - `predictions/external_data/non_sig_reduced/*`
  - `predictions/external_data/sig_reduced/*`
  - `predictions/Uncertainty_experiment_semantic_proteomics/*`
  - `models/uncertainty_estimator_adapted/*`
- Final asset:
  - `images/main_figures/figure6.pdf`

## Supplementary statistical summaries

Manuscript-facing statistical summary CSVs are stored under:

- `analysis_exports/supplementary_stats/`

They can be regenerated from the checked-in current analysis state using:

- `notebooks/supplementary/regenerate_supplementary_stats.ipynb`
