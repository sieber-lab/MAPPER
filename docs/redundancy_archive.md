# Redundancy Archive

This note documents files moved to the local backup folder `redundancy/`.

## Why these files were archived

These modules were identified as legacy or redundant for the active Figure 2-6 publication pipeline. Current figure notebooks and active data-generation scripts do not require them in their original active import paths.

Preprocessing scripts and supplementary notebooks were intentionally kept in place.

## Archived files

- `notebooks/analysis/feature_eval.py`
- `notebooks/analysis/llm_eval.py`
- `notebooks/analysis/loo_eval.py`
- `notebooks/analysis/loo_eval_loso.py`
- `notebooks/analysis/text_only_eval.py`
- `notebooks/modeling/eval_utils.py`
- `notebooks/modeling/eval_utils_legacy.py`
- `notebooks/modeling/eval_utils_loco.py`
- `notebooks/modeling/tabm_reference.py`
- `notebooks/modeling/train_utils.py`
- `scripts/Data_generation/Codebase/eval_utils3.py`
- `scripts/Data_generation/Codebase/eval_utils5.py`

All files were moved to mirrored paths under `redundancy/`, for example:

- `notebooks/analysis/feature_eval.py` -> `redundancy/notebooks/analysis/feature_eval.py`

## Restore instructions

To restore a file to its original active location:

```bash
mv redundancy/<original/path> <original/path>
```

Example:

```bash
mv redundancy/notebooks/analysis/feature_eval.py notebooks/analysis/feature_eval.py
```

## Git behavior

`redundancy/` is ignored via `.gitignore`, so archived backup content stays local and is not uploaded.
