# Paper Figures 2-6: Required Reproducibility Code (Current)

This file lists the code and artifact roots required to regenerate the current Figures 2-6 in this repository.

## 1) Core runner scripts

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

## 2) Data-processing notebooks/scripts used by figures

- `notebooks/data_processing/run_preprocessing.py`
- `notebooks/data_processing/prepare_significant_proteomics_data.ipynb`
- `notebooks/data_processing/eclipse_data_processing.ipynb`
- `notebooks/data_processing/external_proteomics_processing.ipynb`

## 3) Figure notebooks

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
- `notebooks/figure6/figure6_external_lab.ipynb`
- `notebooks/figure6/figure6_eclipse_instrument.ipynb`
- `notebooks/figure6/figure6_combined.ipynb`

## 4) Canonical artifact roots used by final figures

- `predictions/LLM_augmentation_experiment/binary/*`
- `predictions/Text_vs_No_Text_experiment/multiclass/*`
- `predictions/Text_vs_No_Text_experiment/no_text/*`
- `predictions/Feature_augmentation_experiment/binary/semantic_BioBERT_*`
- `predictions/Uncertainty_experiment_semantic/binary_loo_ablation/*`
- `predictions/Uncertainty_experiment_semantic/binary_loso_ablation/*`
- `predictions/Uncertainty_experiment_semantic_proteomics/*` (Figure 6 threshold recalibration source)
- `predictions/Figure5_external_validation/Special_v1/*`
- `predictions/eclipse_data/{non_sig_reduced,sig_reduced}/*`
- `predictions/external_data/{non_sig_reduced,sig_reduced}/*`
- `analysis_exports/Feature_augmentation_experiment/*`
- `analysis_exports/uncertainty_estimator/*`
- `models/uncertainty_estimator/*`
- `models/uncertainty_estimator_adapted/*`

## 5) Canonical augmentation/model choice

- Downstream pipeline (Figures 4-6): `semantic` text augmentation + `BioBERT`.
- Figure 3 retains the explicit augmentation/encoder comparison panel.

## 6) Legacy artifacts no longer canonical

The following are historical and not required for final figure regeneration:
- `analysis_exports/uncertainty_fair_eta*`
- `models/uncertainty_fair_meta*`
- sentence-only downstream uncertainty paths
- FC-threshold exploratory outputs (moved under `redundancy/fc-test/`)
