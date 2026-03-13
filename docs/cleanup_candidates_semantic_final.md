# Cleanup Status After Public Snapshot Preparation

This file records the cleanup decisions applied while preparing the public repository snapshot.

## Removed from the local working tree

- `analysis_exports/supplementary_stats_backup_*`
- `analysis_exports/supplementary_stats_regenerated/`
- `redundancy/`
- `tmp/`
- empty `figures/`
- `similarity_vs_pool.csv`
- `Thumbs.db` files

## Kept locally but marked as non-canonical

Ignored in `.gitignore`:

- `notebooks/figure2/figure2_proteomics_operator_coloring.ipynb`
- `notebooks/figure4/figure4_compound_proteomics_comparison.ipynb`
- `notebooks/figure4/figure4_proteomics_ablation_effects.ipynb`
- `notebooks/figure4/figure4_extended_same_class_pairs.ipynb`
- duplicate image exports (`png/svg`) under `images/figure2/` to `images/figure6/`
- duplicate image exports (`png/svg`) under `images/supplementary/`
- `images/figure5/figure5_combined.afdesign`

## Still intentionally kept

Canonical code and manuscript assets:

- `latex/`
- canonical figure notebooks in `notebooks/figure2/` to `notebooks/figure6/`
- supplementary utility notebooks under `notebooks/supplementary/`
- `analysis_exports/`
- `Data/`
- `Embeddings/`
- `images/main_figures/`
- `images/supplementary/fig3_SI.pdf`
- `images/supplementary/fig4_SI.pdf`

Local-only large runtime roots may still exist in a private workspace but are ignored in public git history:

- `predictions/`
- `models/`
- `results/`
