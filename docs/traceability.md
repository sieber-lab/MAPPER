# Traceability: Manuscript Figures -> Code -> Data (Canonical, Semantic)

This document maps main manuscript figures to the notebooks, input artifacts, and generator scripts used in the current final pipeline.

## Figure 2 (dataset overview)
- Notebook: `notebooks/figure2/figure2_data_overview.ipynb`
- Inputs:
  - `Data/Base/X_proteomics.pkl`, `X_ecfp.pkl`, `X_cellpaint.pkl`, `X_mic.pkl`, `X_gc.pkl`, `y_labels.pkl`
- Upstream generation:
  - `notebooks/data_processing/run_preprocessing.py`
  - `scripts/Preprocessing/*`

## Figure 3 (text augmentation and text-vs-no-text)
- Notebooks:
  - `notebooks/figure3/figure3_llm_parametric_tests.ipynb`
  - `notebooks/figure3/figure3_text_augmentation_effects.ipynb`
- Inputs:
  - `predictions/LLM_augmentation_experiment/binary/*`
  - `predictions/Text_vs_No_Text_experiment/multiclass/*`
  - `predictions/Text_vs_No_Text_experiment/no_text/*`
- Canonical downstream choice from this benchmark:
  - augmentation `semantic`, encoder `BioBERT`

## Figure 4 (feature combinations + Shapley + structural novelty jumps)
- Notebooks:
  - `notebooks/figure4/figure4_best_feature_combination.ipynb`
  - `notebooks/figure4/figure4_feature_replot.ipynb`
  - `notebooks/figure4/figure4_shapley_analysis.ipynb`
  - `notebooks/figure4/figure4_structural_novelty_modality_jumps.ipynb`
  - `notebooks/figure4/figure4_combined.ipynb`
- Inputs:
  - `predictions/Feature_augmentation_experiment/binary/semantic_BioBERT_*`
  - `analysis_exports/Feature_augmentation_experiment/*`

## Figure 5 (uncertainty model development + special compounds)
- Notebooks:
  - `notebooks/figure5/figure5_uncertainty_comparison.ipynb`
  - `notebooks/figure5/figure5_combined.ipynb`
- Inputs:
  - `predictions/Uncertainty_experiment_semantic/binary_loo_ablation/semantic_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC/`
  - `predictions/Uncertainty_experiment_semantic/binary_loso_ablation/semantic_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC/`
  - `predictions/Figure5_external_validation/Special_v1/`
- Exported model artifact:
  - `models/uncertainty_estimator/`

## Figure 6 (cross-instrument and external-lab transfer)
- Notebooks:
  - `notebooks/figure6/figure6_uncertainty_estimator_calibration.ipynb`
  - `notebooks/figure6/figure6_eclipse_instrument.ipynb`
  - `notebooks/figure6/figure6_external_lab.ipynb`
  - `notebooks/figure6/figure6_combined.ipynb`
- Inputs:
  - `predictions/eclipse_data/non_sig_reduced/`, `predictions/eclipse_data/sig_reduced/`
  - `predictions/external_data/non_sig_reduced/`, `predictions/external_data/sig_reduced/`
  - LOO/LOSO recalibration source: `predictions/Uncertainty_experiment_semantic_proteomics/*`
- Exported adapted artifact:
  - `models/uncertainty_estimator_adapted/`

## Notes
- Legacy sentence-based uncertainty artifacts (`uncertainty_fair_*`, `uncertainty_fair_meta`) are deprecated and not used in the final figure notebooks.
- FC-threshold exploratory analysis was moved to `redundancy/fc-test/` and is not part of the canonical manuscript pipeline.
