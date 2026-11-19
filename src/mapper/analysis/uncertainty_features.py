"""Original compound-level uncertainty features from primary member scores."""

import numpy as np
import pandas as pd
from itertools import combinations


def expert_metrics(experts):
    x = np.asarray(experts)
    assert x.ndim == 3 and x.shape[1:] == (32, 2)
    assert np.isfinite(x).all() and ((x >= 0) & (x <= 1)).all()
    p = np.clip(x, 1e-12, 1 - 1e-12)
    p = p / p.sum(axis=-1, keepdims=True)
    mean = p.mean(axis=1)
    entropy = lambda a: -(a * np.log(a)).sum(axis=-1)
    pred_entropy = entropy(mean)
    exp_entropy = entropy(p).mean(axis=1)
    modal = p.argmax(axis=-1)
    agree = np.maximum((modal == 0).mean(axis=1), (modal == 1).mean(axis=1))
    # Same draw stream and per-pair arithmetic as the historical 256-draw loop.
    rng = np.random.default_rng(12345)
    pairs = rng.integers(0, 32, size=(len(p), 256, 2))
    index = np.arange(len(p))[:, None]
    pi, pj = p[index, pairs[:, :, 0]], p[index, pairs[:, :, 1]]
    kl = (pi * (np.log(pi) - np.log(pj))).sum(axis=-1)
    valid = pairs[:, :, 0] != pairs[:, :, 1]
    kl = np.where(valid, kl, 0).sum(axis=1, dtype=np.float64) / np.maximum(valid.sum(axis=1), 1)
    return pd.DataFrame(
        dict(
            PredEntropy=pred_entropy,
            ExpEntropy=exp_entropy,
            EpistemicMI=pred_entropy - exp_entropy,
            MeanPairKL=kl,
            VarMax=p.max(axis=-1).var(axis=1),
            VariationRatio=1 - agree,
            KConsistency=agree,
        )
    )


def compound_features(rows, candidates):
    """One compound per call, all replicates/descriptions/seeds. No labels used."""
    d = rows[rows.CandidateMoA.isin(candidates)].copy()
    assert d.CompoundID.nunique() == 1 and d.SampleID.nunique() >= 1
    assert not d.duplicated(["seed", "RowID"]).any()
    assert set(d.CandidateMoA) == set(candidates) and len(candidates) >= 2
    scores = d.groupby("CandidateMoA").ProbaPos.mean().sort_index()
    order = np.argsort(-scores.to_numpy(), kind="stable")
    top, second = scores.index[order[:2]]
    selected = d[d.CandidateMoA == top]
    per_seed = selected.groupby("seed").ProbaPos.mean()
    aug = selected.groupby(["seed", "SampleID"]).ProbaPos.var(ddof=0).fillna(0.0)
    rep = d.groupby(["SampleID", "CandidateMoA"]).ProbaPos.mean().unstack().sort_index(axis=1)
    sorted_rep = np.sort(rep.to_numpy(), axis=1)
    margins = sorted_rep[:, -1] - sorted_rep[:, -2]
    result = dict(
        CompoundID=str(d.CompoundID.iloc[0]),
        PredictedMoA=str(top),
        SecondMoA=str(second),
        PredScore=float(scores[top]),
        SecondScore=float(scores[second]),
        Margin=float(scores[top] - scores[second]),
        SeedStd=float(per_seed.std(ddof=0)),
        AugVar=float(aug.mean()),
        BioVar=float(rep[top].var(ddof=0)),
        RepMarginStd=float(margins.std(ddof=0)),
        RepMarginIQR=float(np.quantile(margins, 0.75) - np.quantile(margins, 0.25)),
        RepAgree=float(rep.idxmax(axis=1).eq(top).mean()),
    )
    for name in [
        "PredEntropy",
        "ExpEntropy",
        "EpistemicMI",
        "MeanPairKL",
        "VarMax",
        "VariationRatio",
        "KConsistency",
    ]:
        result[name] = float(selected[name].mean())
    return result, scores.to_dict()
