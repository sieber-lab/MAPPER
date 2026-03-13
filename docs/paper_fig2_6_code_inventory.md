# Paper Figures 2-6: Current Code Inventory

This file lists the code and artifact roots required to regenerate the current manuscript figures.

## Scope

- Main manuscript Figures 2-6
- Supplementary Figures S1-S2
- Manuscript-facing statistical summary CSVs under `analysis_exports/supplementary_stats/`

## Important public/private note

This public repository tracks the code, curated exports, and figure assets. Large runtime artifact roots such as `predictions/`, `models/`, and `results/` may exist only in a local workspace and are ignored in public git history.

## Core scripts

- `scripts/Data_generation/LLM_eval_script.py`
- `scripts/Data_generation/Feature_eval_script.py`
- `scripts/Data_generation/LOO_eval_script.py`
- `scripts/Data_generation/train_primary_tabm.py`
- `scripts/Data_generation/predict_primary_tabm.py`
- `scripts/uncertainty/uncertainty_estimator.py`

Supporting modules:

- `scripts/Data_generation/Codebase/eval_utils.py`
- `scripts/Data_generation/Codebase/train_utils.py`
- `scripts/Data_generation/Codebase/tabm_reference.py`

## Preprocessing notebooks/scripts

- `notebooks/data_processing/run_preprocessing.py`
- `notebooks/data_processing/prepare_significant_proteomics_data.ipynb`
- `notebooks/data_processing/eclipse_data_processing.ipynb`
- `notebooks/data_processing/external_proteomics_processing.ipynb`

## Canonical figure notebooks

Figure 2:

- `notebooks/figure2/figure2_data_overview.ipynb`
- `notebooks/figure2/figure2_combined.ipynb`

Figure 3:

- `notebooks/figure3/figure3_llm_parametric_tests.ipynb`
- `notebooks/figure3/figure3_text_augmentation_effects.ipynb`
- `notebooks/figure3/figure3_combined.ipynb`

Figure 4:

- `notebooks/figure4/figure4_best_feature_combination.ipynb`
- `notebooks/figure4/figure4_feature_replot.ipynb`
- `notebooks/figure4/figure4_shapley_analysis.ipynb`
- `notebooks/figure4/figure4_structural_novelty_modality_jumps.ipynb`
- `notebooks/figure4/figure4_combined.ipynb`

Figure 5:

- `notebooks/figure5/figure5_uncertainty_comparison.ipynb`
- `notebooks/figure5/figure5_combined.ipynb`

Figure 6:

- `notebooks/figure6/figure6_uncertainty_estimator_calibration.ipynb`
- `notebooks/figure6/figure6_eclipse_instrument.ipynb`
- `notebooks/figure6/figure6_external_lab.ipynb`
- `notebooks/figure6/figure6_combined.ipynb`

Supplementary:

- `notebooks/supplementary/supp_string_analysis_preprocess.ipynb`
- `notebooks/supplementary/supp_string_tabm_predictions.ipynb`
- `notebooks/supplementary/supp_table_s7_ci.ipynb`
- `notebooks/supplementary/regenerate_supplementary_stats.ipynb`

## Local artifact roots used by the notebooks

- `Data/Base/*.pkl`
- `Data/Special/*.pkl`
- `Data/eclipse_data/*`
- `Data/external_data/*`
- `Embeddings/*`
- `analysis_exports/Feature_augmentation_experiment/*`
- `analysis_exports/LLM_augmentation_experiment*`
- `analysis_exports/text_vs_notext_experiment/*`
- `analysis_exports/text_vs_notext_proteomics_grouped_stars/*`
- `analysis_exports/uncertainty_estimator/*`
- `analysis_exports/figure4_novelty_modality_jumps/*`
- `analysis_exports/supplementary_stats/*`

Local-only runtime roots often required for full reruns:

- `predictions/LLM_augmentation_experiment/*`
- `predictions/Text_vs_No_Text_experiment/*`
- `predictions/Feature_augmentation_experiment/*`
- `predictions/Uncertainty_experiment_semantic/*`
- `predictions/Uncertainty_experiment_semantic_proteomics/*`
- `predictions/Figure5_external_validation/Special_v1/*`
- `predictions/eclipse_data/{non_sig_reduced,sig_reduced}/*`
- `predictions/external_data/{non_sig_reduced,sig_reduced}/*`
- `models/uncertainty_estimator/*`
- `models/uncertainty_estimator_adapted/*`

## Non-canonical notebooks

The following notebooks are kept locally for exploratory history but are not part of the current canonical figure inventory:

- `notebooks/figure2/figure2_proteomics_operator_coloring.ipynb`
- `notebooks/figure4/figure4_compound_proteomics_comparison.ipynb`
- `notebooks/figure4/figure4_proteomics_ablation_effects.ipynb`
- `notebooks/figure4/figure4_extended_same_class_pairs.ipynb`
