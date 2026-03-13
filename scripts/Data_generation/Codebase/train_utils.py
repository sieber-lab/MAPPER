# train_utils.py
from __future__ import annotations
import os, json, pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Reuse shared feature/backbone helpers from unified eval utils
from Codebase.eval_utils import build_features, generate_backbone_df


# -----------------------------
# Data loading / preparation
# -----------------------------
def load_base_data(
    base_dir: str | Path = "Data/Base",
    emb_sem: str | Path = "Embeddings/250808_semanticAugmentation_embeddings.pkl",
    emb_sentence: str | Path = "Embeddings/250808_sentenceAugmentation_embeddings.pkl",
    emb_word: str | Path = "Embeddings/250808_wordAugmentation_embeddings.pkl",
) -> dict:
    """
    Loads the same pickles used in your evaluation pipelines.
    Returns a dict with matrices/labels and the 3 text-embedding tables.
    """
    base_dir = Path(base_dir)
    with open(base_dir / "X_proteomics.pkl", "rb") as f: X_prot = pickle.load(f)
    with open(base_dir / "y_labels.pkl",    "rb") as f: y_lbl  = pickle.load(f)
    with open(base_dir / "X_cellpaint.pkl", "rb") as f: X_cp   = pickle.load(f)
    with open(base_dir / "X_ecfp.pkl",      "rb") as f: X_ecfp = pickle.load(f)
    with open(base_dir / "X_mic.pkl",       "rb") as f: X_mic  = pickle.load(f)
    with open(base_dir / "X_gc.pkl",        "rb") as f: X_gc   = pickle.load(f)

    # Consistent dtypes
    if isinstance(y_lbl, dict): y_lbl = pd.Series(y_lbl)
    for df in (X_prot, X_cp, X_ecfp, X_mic, X_gc):
        df.index = df.index.astype(str)
    y_lbl.index = y_lbl.index.astype(str)

    # Load text embeddings
    emb_sem_df  = pd.read_pickle(emb_sem)
    emb_sent_df = pd.read_pickle(emb_sentence)
    emb_word_df = pd.read_pickle(emb_word)

    return dict(
        X_proteomics=X_prot, X_cellpaint=X_cp, X_ecfp=X_ecfp, X_mic=X_mic, X_gc=X_gc,
        y_labels=y_lbl,
        text_embeddings=dict(semantic=emb_sem_df, sentence=emb_sent_df, word=emb_word_df),
    )

from typing import List, Tuple, Dict
import pickle
import pandas as pd
from pathlib import Path

def load_inference_data(
    base_dir: str | Path = "Data/Base",
    emb_sem: str | Path = "Embeddings/250808_semanticAugmentation_embeddings.pkl",
    emb_sentence: str | Path = "Embeddings/250808_sentenceAugmentation_embeddings.pkl",
    emb_word: str | Path = "Embeddings/250808_wordAugmentation_embeddings.pkl",
    selected_modalities: List[str] | Tuple[str, ...] = (),
    aug_type: str = "semantic",
) -> dict:
    """
    Inference-only loader:
      - Loads ONLY the selected modalities.
      - Always loads y_labels and coerces it to a Series with string index.
      - Loads ONLY the requested embedding table (aug_type).
      - Returns same keys as load_base_data (unused modalities are empty DFs with the correct index).
    """
    base_dir = Path(base_dir)

    # --- y_labels (coerce to Series) ---
    y_path = base_dir / "y_labels.pkl"
    if not y_path.exists():
        raise FileNotFoundError(f"Missing y_labels file: {y_path}")
    with open(y_path, "rb") as f:
        y_lbl = pickle.load(f)

    # Coerce to Series (robust to dict/DataFrame/Series)
    if isinstance(y_lbl, dict):
        y_lbl = pd.Series(y_lbl)
    elif isinstance(y_lbl, pd.DataFrame):
        if "MoA" in y_lbl.columns:
            y_lbl = y_lbl["MoA"]
        elif y_lbl.shape[1] == 1:
            y_lbl = y_lbl.iloc[:, 0]
        else:
            raise ValueError(
                f"y_labels DataFrame has multiple columns {list(y_lbl.columns)}; "
                f"expected a single column or a column named 'MoA'."
            )
    elif not isinstance(y_lbl, pd.Series):
        # last resort: squeeze if possible
        try:
            y_lbl = pd.Series(y_lbl)
        except Exception as e:
            raise TypeError(f"Unsupported y_labels type: {type(y_lbl)}") from e

    y_lbl.index = y_lbl.index.astype(str)
    y_lbl = y_lbl.astype("object")

    # Helper for empty DF with proper index
    def _empty_df():
        return pd.DataFrame(index=y_lbl.index.copy())

    # Initialize all modality frames as empty placeholders
    frames = {
        "X_proteomics": _empty_df(),
        "X_cellpaint":  _empty_df(),
        "X_ecfp":       _empty_df(),
        "X_mic":        _empty_df(),
        "X_gc":         _empty_df(),
    }

    # Only load selected modalities
    mod2file = {
        "Proteomics": ("X_proteomics.pkl", "X_proteomics"),
        "ECFP":       ("X_ecfp.pkl",      "X_ecfp"),
        "CellPaint":  ("X_cellpaint.pkl", "X_cellpaint"),
        "MIC":        ("X_mic.pkl",       "X_mic"),
        "GC":         ("X_gc.pkl",        "X_gc"),
    }
    selected_set = set(selected_modalities or [])
    for mod_name, (fname, outkey) in mod2file.items():
        if mod_name in selected_set:
            p = base_dir / fname
            if not p.exists():
                raise FileNotFoundError(
                    f"Selected modality '{mod_name}' is required but file not found: {p}"
                )
            with open(p, "rb") as f:
                df = pickle.load(f)
            df.index = df.index.astype(str)
            frames[outkey] = df

    # Load only the requested embedding table
    a = str(aug_type).strip().lower()
    if a == "semantic":
        emb_path = Path(emb_sem)
    elif a == "sentence":
        emb_path = Path(emb_sentence)
    elif a == "word":
        emb_path = Path(emb_word)
    else:
        raise ValueError(f"Invalid aug_type: {aug_type} (expected 'semantic'|'sentence'|'word')")
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file for aug_type='{a}' not found: {emb_path}")
    emb_df = pd.read_pickle(emb_path)

    return dict(
        X_proteomics=frames["X_proteomics"],
        X_cellpaint=frames["X_cellpaint"],
        X_ecfp=frames["X_ecfp"],
        X_mic=frames["X_mic"],
        X_gc=frames["X_gc"],
        y_labels=y_lbl,
        text_embeddings={a: emb_df},
    )




def make_feature_dicts(
    X_proteomics: pd.DataFrame, X_ecfp: pd.DataFrame, X_cellpaint: pd.DataFrame,
    X_mic: pd.DataFrame, X_gc: pd.DataFrame
) -> Dict[str, dict]:
    """Convert each modality to {SampleID -> {col -> value}} maps."""
    return {
        "Proteomics": X_proteomics.to_dict(orient="index"),
        "ECFP":       X_ecfp.to_dict(orient="index"),
        "CellPaint":  X_cellpaint.to_dict(orient="index"),
        "MIC":        X_mic.to_dict(orient="index"),
        "GC":         X_gc.to_dict(orient="index"),
    }


def attach_modalities_to_backbone(backbone_df: pd.DataFrame, feature_dicts: Dict[str, dict]) -> pd.DataFrame:
    """Add modality arrays as columns to the pairwise backbone rows."""
    out = backbone_df.copy()
    for mod in feature_dicts:
        out[mod] = out["SampleID"].map(lambda s: np.array(list(feature_dicts[mod][s].values())))
    return out


def modality_dims_from_frames(
    X_proteomics: pd.DataFrame, X_ecfp: pd.DataFrame, X_cellpaint: pd.DataFrame,
    X_mic: pd.DataFrame, X_gc: pd.DataFrame,
    text_dim: int = 768
) -> Dict[str, int]:
    return {
        "Text_Embedding": int(text_dim),
        "Proteomics":     int(X_proteomics.shape[1]),
        "ECFP":           int(X_ecfp.shape[1]),
        "CellPaint":      int(X_cellpaint.shape[1]),
        "MIC":            int(X_mic.shape[1]),
        "GC":             int(X_gc.shape[1]),
    }


def assemble_pairwise_training_table(
    text_df: pd.DataFrame, llm_model: str, y_labels: pd.Series, samples: pd.Index,
    feature_dicts: Dict[str, dict], selected_modalities: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build the pairwise backbone (Label is the target), attach modalities,
    and stack selected modalities + Text_Embedding into X; return (backbone_df, X, y).
    """
    # 1) backbone across all samples and all candidate MoAs from the chosen text table
    backbone_df = generate_backbone_df(text_df, llm_model, y_labels, samples)

    # 2) attach modality arrays per row
    backbone_df = attach_modalities_to_backbone(backbone_df, feature_dicts)

    # 3) ensure Text_Embedding exists and is first modality
    if "Text_Embedding" not in selected_modalities:
        selected_modalities = ["Text_Embedding"] + list(selected_modalities)

    # 4) stack features
    X = build_features(backbone_df, selected_modalities)  # np.ndarray
    y = backbone_df["Label"].astype(int).values

    return backbone_df, X, y


# -----------------------------
# Export
# -----------------------------
def export_model_bundle(
    export_dir: str | Path,
    model_obj,
    config: dict,
    feature_order: List[str],
    modality_dims: Dict[str, int],
):
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # model
    with open(export_dir / "model.pkl", "wb") as f:
        pickle.dump(model_obj, f)

    # metadata
    with open(export_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # feature order & dims (critical for inference)
    with open(export_dir / "feature_order.json", "w") as f:
        json.dump({"modalities": feature_order}, f, indent=2)

    with open(export_dir / "modality_dims.json", "w") as f:
        json.dump(modality_dims, f, indent=2)
