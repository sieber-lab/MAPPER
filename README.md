# Paper Code Repository

This folder is a curated, publication-ready snapshot of the code used to generate the figures and supplementary analyses in the manuscript. It lives alongside the original project; **all original files remain untouched**.

## Structure

- `notebooks/` – figure- and supplement-specific notebooks (one per plot where possible)
- `src/` – reusable code for preprocessing, modeling, evaluation, and uncertainty
- `docs/` – traceability notes and figure-to-code mapping
- `images/` – exported figures (PNG/PDF/SVG), organized by figure

## How to run notebooks

Notebooks include a small bootstrap cell that:
- finds the project root containing `Data/` and `predictions/`
- `chdir`s to that root so original relative paths still work
- adds `code_repository/src` to `sys.path`
They also call a shared style helper that enforces consistent plot formatting and saves outputs to `code_repository/images/<figure>/` as PNG/PDF/SVG.

If your data/predictions live elsewhere, update the bootstrap cell or set the working directory manually.

## Pipeline overview

1) **Data processing**
   - `src/data_processing/run_preprocessing.py` orchestrates proteomics, chemistry, cell-painting, and text-embedding pipelines
   - Produces `Data/Base/*.pkl` and `Embeddings/*_embeddings.pkl`

2) **Model evaluation / prediction generation**
   - `src/analysis/llm_eval.py` (LLM + augmentation comparisons; Figure 3 inputs)
   - `src/analysis/feature_eval.py` (feature combination comparisons; Figure 4 inputs)
   - `src/analysis/loo_eval.py` and `src/analysis/loo_eval_loso.py` (LOO/LOSO ablations; Figure 5 inputs)
   - `src/modeling/train_primary_tabm.py` + `src/modeling/predict_primary_tabm.py` (external set predictions; Figures 5–6 inputs)

3) **Plotting / analysis**
   - `notebooks/figure*/` and `notebooks/supplementary/`

See `docs/traceability.md` for a detailed mapping from figures to notebooks and upstream data generation.

## Notes

- This repo does **not** include large data files, model weights, or prediction artifacts.
- Relative paths in notebooks/scripts assume the original top-level folders still exist (e.g., `Data/`, `predictions/`, `Embeddings/`).
- External dependencies (RDKit, PyTorch, Lightning, XGBoost, etc.) are required; assemble your environment accordingly.
