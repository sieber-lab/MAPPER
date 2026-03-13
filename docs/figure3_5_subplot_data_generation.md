# Figure 3-5 Regeneration Notes

This runbook summarizes the current regeneration order for the figure-analysis notebooks used in the manuscript.

## Figure 3

Run in this order:

1. `notebooks/figure3/figure3_llm_parametric_tests.ipynb`
2. `notebooks/figure3/figure3_text_augmentation_effects.ipynb`
3. `notebooks/figure3/figure3_combined.ipynb`

Primary inputs:

- `predictions/LLM_augmentation_experiment/binary/*`
- `predictions/Text_vs_No_Text_experiment/multiclass/*`
- `predictions/Text_vs_No_Text_experiment/no_text/*`

Primary exports consumed by the combined notebook:

- `analysis_exports/LLM_augmentation_experiment*`
- `analysis_exports/text_vs_notext_experiment/*`
- `analysis_exports/text_vs_notext_proteomics_grouped_stars/*`

## Figure 4

Run in this order:

1. `notebooks/figure4/figure4_best_feature_combination.ipynb`
2. `notebooks/figure4/figure4_feature_replot.ipynb`
3. `notebooks/figure4/figure4_shapley_analysis.ipynb`
4. `notebooks/figure4/figure4_structural_novelty_modality_jumps.ipynb`
5. `notebooks/figure4/figure4_combined.ipynb`

Primary inputs:

- `predictions/Feature_augmentation_experiment/binary/semantic_BioBERT_*`

Primary exports consumed by the combined notebook:

- `analysis_exports/Feature_augmentation_experiment/*`
- `analysis_exports/figure4_novelty_modality_jumps/*`

## Figure 5

Run in this order:

1. `notebooks/figure5/figure5_uncertainty_comparison.ipynb`
2. `notebooks/figure5/figure5_combined.ipynb`

Primary inputs:

- `predictions/Uncertainty_experiment_semantic/binary_loo_ablation/*`
- `predictions/Uncertainty_experiment_semantic/binary_loso_ablation/*`
- `predictions/Figure5_external_validation/Special_v1/*`

Primary exports consumed by the combined notebook:

- `analysis_exports/uncertainty_estimator/*`
- `models/uncertainty_estimator/*`

## Supplementary stats refresh

If the figure-analysis notebooks are rerun and manuscript-facing SI tables should be refreshed too, also run:

- `notebooks/supplementary/supp_table_s7_ci.ipynb`
- `notebooks/supplementary/regenerate_supplementary_stats.ipynb`

## Outputs

Combined manuscript figures:

- `images/main_figures/figure3.pdf`
- `images/main_figures/figure4.pdf`
- `images/main_figures/figure5.pdf`

Supplementary summary CSVs:

- `analysis_exports/supplementary_stats/*`
