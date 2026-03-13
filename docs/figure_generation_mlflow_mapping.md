# Figure Generation and MLflow Provenance

The original project used local MLflow tracking during development. Those MLflow stores are not part of this public repository.

## Current recommendation

For public reproducibility, treat the checked-in notebooks, exported CSVs, and local artifact manifests as the primary provenance layer:

- notebooks under `notebooks/figure*/`
- curated exports under `analysis_exports/`
- prediction manifests under local `predictions/*` roots
- model manifests under local `models/*` roots

## What is intentionally not portable here

- private MLflow tracking directories from the original workspace
- absolute MLflow source paths from the original machine
- run-ID-level provenance tied to private development storage

## If local MLflow history still exists in a private workspace

Useful lookups are:

- `tags/mlflow.source.name` for the generating script
- `params/*` for recorded run parameters
- `metrics/*` for tracked scalar outputs

For the public repository, however, the figure regeneration docs in:

- `docs/paper_fig2_6_code_inventory.md`
- `docs/traceability.md`
- `docs/figure3_5_subplot_data_generation.md`

should be treated as the canonical high-level provenance notes.
