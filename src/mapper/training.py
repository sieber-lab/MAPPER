"""Load curated feature matrices and assemble the recorded text-matching inputs."""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from .evaluation import build_features, generate_backbone_df


def load_base_data(
    base_dir: str | Path = "data/inhouse",
    emb_sem: str | Path = "data/embeddings/250808_semanticAugmentation_embeddings.pkl",
    emb_sentence: str | Path = "data/embeddings/250808_sentenceAugmentation_embeddings.pkl",
    emb_word: str | Path = "data/embeddings/250808_wordAugmentation_embeddings.pkl",
) -> dict:
    """
    Loads the same pickles used in your evaluation pipelines.
    Returns a dict with matrices/labels and the 3 text-embedding tables.
    """
    base_dir = Path(base_dir)
    with open(base_dir / "X_proteomics.pkl", "rb") as f:
        X_prot = pickle.load(f)
    with open(base_dir / "y_labels.pkl", "rb") as f:
        y_lbl = pickle.load(f)
    with open(base_dir / "X_infoalign.pkl", "rb") as f:
        X_cp = pickle.load(f)
    with open(base_dir / "X_ecfp.pkl", "rb") as f:
        X_ecfp = pickle.load(f)
    with open(base_dir / "X_mic.pkl", "rb") as f:
        X_mic = pickle.load(f)
    with open(base_dir / "X_gc.pkl", "rb") as f:
        X_gc = pickle.load(f)

    # Consistent dtypes
    if isinstance(y_lbl, dict):
        y_lbl = pd.Series(y_lbl)
    for df in (X_prot, X_cp, X_ecfp, X_mic, X_gc):
        df.index = df.index.astype(str)
    y_lbl.index = y_lbl.index.astype(str)

    # Load text embeddings
    emb_sem_df = pd.read_pickle(emb_sem)
    emb_sent_df = pd.read_pickle(emb_sentence)
    emb_word_df = pd.read_pickle(emb_word)

    return dict(
        X_proteomics=X_prot,
        X_infoalign=X_cp,
        X_ecfp=X_ecfp,
        X_mic=X_mic,
        X_gc=X_gc,
        y_labels=y_lbl,
        text_embeddings=dict(semantic=emb_sem_df, sentence=emb_sent_df, word=emb_word_df),
    )


def make_feature_dicts(
    X_proteomics: pd.DataFrame,
    X_ecfp: pd.DataFrame,
    X_infoalign: pd.DataFrame,
    X_mic: pd.DataFrame,
    X_gc: pd.DataFrame,
) -> Dict[str, dict]:
    """Convert each modality to {SampleID -> {col -> value}} maps."""
    return {
        "Proteomics": X_proteomics.to_dict(orient="index"),
        "ECFP": X_ecfp.to_dict(orient="index"),
        "CellPaint": X_infoalign.to_dict(orient="index"),
        "MIC": X_mic.to_dict(orient="index"),
        "GC": X_gc.to_dict(orient="index"),
    }


def attach_modalities_to_backbone(
    backbone_df: pd.DataFrame, feature_dicts: Dict[str, dict]
) -> pd.DataFrame:
    """Add modality arrays as columns to the pairwise backbone rows."""
    out = backbone_df.copy()
    for mod in feature_dicts:
        out[mod] = out["SampleID"].map(lambda s: np.array(list(feature_dicts[mod][s].values())))
    return out


def modality_dims_from_frames(
    X_proteomics: pd.DataFrame,
    X_ecfp: pd.DataFrame,
    X_infoalign: pd.DataFrame,
    X_mic: pd.DataFrame,
    X_gc: pd.DataFrame,
    text_dim: int = 768,
) -> Dict[str, int]:
    return {
        "Text_Embedding": int(text_dim),
        "Proteomics": int(X_proteomics.shape[1]),
        "ECFP": int(X_ecfp.shape[1]),
        "CellPaint": int(X_infoalign.shape[1]),
        "MIC": int(X_mic.shape[1]),
        "GC": int(X_gc.shape[1]),
    }


def assemble_pairwise_training_table(
    text_df: pd.DataFrame,
    llm_model: str,
    y_labels: pd.Series,
    samples: pd.Index,
    feature_dicts: Dict[str, dict],
    selected_modalities: List[str],
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
