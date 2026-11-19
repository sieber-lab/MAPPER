"""The fixed model-family selection policies evaluated by BBC-CV."""

import numpy as np
from .configuration_selection import model_family

MODELS = ("xgboost", "lightgbm", "tabm", "pytorch_mlp", "mlp", "rf")
DEPLOYED = "semantic_BioBERT_tabm_Text_Embedding+Proteomics+ECFP+CellPaint+MIC+GC"


def policies(names: list[str]) -> dict[str, np.ndarray]:
    result = {"all_configurations": np.arange(len(names))}
    for model in MODELS:
        indices = np.array([i for i, name in enumerate(names) if model_family(name) == model])
        if len(indices):
            result[f"within_{model}"] = indices
    if DEPLOYED not in names:
        raise ValueError("Deployed full-feature TabM configuration missing")
    result["fixed_deployed_tabm"] = np.array([names.index(DEPLOYED)])
    return result
