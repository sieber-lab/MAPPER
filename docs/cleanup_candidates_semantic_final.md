# Cleanup Candidates After Semantic-Finalization

This file lists artifacts that are not part of the current canonical pipeline
(Figures 3-6 with semantic downstream and `uncertainty_estimator` / `uncertainty_estimator_adapted`).

## Keep (canonical)

- `predictions/LLM_augmentation_experiment/`
- `predictions/Text_vs_No_Text_experiment/`
- `predictions/Feature_augmentation_experiment/`
- `predictions/Uncertainty_experiment_semantic/`
- `predictions/Uncertainty_experiment_semantic_proteomics/`
- `predictions/Figure5_external_validation/Special_v1/`
- `predictions/eclipse_data/{non_sig_reduced,sig_reduced}/`
- `predictions/external_data/{non_sig_reduced,sig_reduced}/`
- `models/tabM_all_modalities/`
- `models/tabM_proteomics_external_full/`
- `models/tabM_proteomics_external_sig/`
- `models/tabM_proteomics_red_eclipse/`
- `models/tabM_proteomics_sig_red_eclipse/`
- `models/uncertainty_estimator/`
- `models/uncertainty_estimator_adapted/`

## Candidate removal (legacy / exploratory)

- Sentence-specific downstream model bundle:
  - `models/tabM_all_modalities_sentence/`
- Obsolete uncertainty experiment roots from pre-final comparisons:
  - `predictions/Uncertainty_experiment/`
  - `predictions/Uncertainty_experiment_word/`
  - `predictions/Proteomics_ablation_study/` (if no longer needed for any archived figure)
- Intermediate uncertainty transfer experiments not used in final main-text figures:
  - `predictions/Uncertainty_experiment_semantic_ecfp_cp/`
  - `predictions/Uncertainty_experiment_semantic_prot_ecfp/`
  - `predictions/Uncertainty_experiment_semantic_prot_ecfp_cp/`
- Legacy/sandbox prediction outputs:
  - `predictions/testing/`
  - `predictions/external_data/*/uncertainty_meta_apply_semantic_proteomics_meta/`
  - `predictions/external_data/*/uncertainty_meta_apply_uncertainty_optimized_*`
  - `predictions/external_data/*/uncertainty_meta_apply_uncertainty_constrained_*`
  - `predictions/external_data/sig_reduced/uncertainty_meta_apply_semantic01_recal_semprot/`
- Optional duplicate special-set run:
  - `predictions/Figure5_external_validation/Special_semantic_prot_ecfp_cp_v1/`

## Notebook/code cleanup candidates

- Any archived FC-threshold exploratory notebooks/scripts now moved to:
  - `redundancy/fc-test/`
- Any sentence-specific uncertainty notebooks retained only for history.

## Notes

- Remove candidate roots only after confirming no figure notebook references them.
- If reproducibility of historical iterations is needed, archive first under `redundancy/` instead of deletion.
