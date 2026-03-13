# MAPPER

Public code and analysis snapshot for the antibiotic mode-of-action prediction project.

This repository contains the code, curated analysis exports, and figure assets used for the current paper and Supplementary Information.

## Repository layout

- `notebooks/` – figure, preprocessing, and supplementary notebooks
- `scripts/` – reusable generation and preprocessing scripts
- `Data/` – curated input matrices and supporting input files used by the notebooks
- `Embeddings/` – precomputed text embedding artifacts
- `analysis_exports/` – manuscript-facing summary exports and figure-analysis CSVs
- `images/` – exported figure assets
- `docs/` – code-to-figure mapping and reproducibility notes

## Local-only large artifacts

The working project also uses large prediction, model, and result caches under:

- `predictions/`
- `models/`
- `results/`

These are intentionally ignored in the public git history. If they are present in a local workspace, the notebooks and docs may refer to them as canonical runtime artifact roots.

## Canonical figure notebooks

Figures 2-6:

- `notebooks/figure2/figure2_data_overview.ipynb`
- `notebooks/figure2/figure2_combined.ipynb`
- `notebooks/figure3/figure3_llm_parametric_tests.ipynb`
- `notebooks/figure3/figure3_text_augmentation_effects.ipynb`
- `notebooks/figure3/figure3_combined.ipynb`
- `notebooks/figure4/figure4_best_feature_combination.ipynb`
- `notebooks/figure4/figure4_feature_replot.ipynb`
- `notebooks/figure4/figure4_shapley_analysis.ipynb`
- `notebooks/figure4/figure4_structural_novelty_modality_jumps.ipynb`
- `notebooks/figure4/figure4_combined.ipynb`
- `notebooks/figure5/figure5_uncertainty_comparison.ipynb`
- `notebooks/figure5/figure5_combined.ipynb`
- `notebooks/figure6/figure6_uncertainty_estimator_calibration.ipynb`
- `notebooks/figure6/figure6_eclipse_instrument.ipynb`
- `notebooks/figure6/figure6_external_lab.ipynb`
- `notebooks/figure6/figure6_combined.ipynb`

Supplementary utilities:

- `notebooks/supplementary/supp_string_analysis_preprocess.ipynb`
- `notebooks/supplementary/supp_string_tabm_predictions.ipynb`
- `notebooks/supplementary/supp_table_s7_ci.ipynb`
- `notebooks/supplementary/regenerate_supplementary_stats.ipynb`

## Reproducibility notes

- Main manuscript and SI figure numbering/inputs are documented in `docs/traceability.md`.
- Required code and local artifact roots for Figures 2-6 are summarized in `docs/paper_fig2_6_code_inventory.md`.
