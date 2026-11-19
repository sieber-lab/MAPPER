"""Development-fitted original entropy/variance baseline definitions."""

import numpy as np
import pandas as pd

GATE_SIGNALS = ("MeanPairKL", "PredEntropy", "EpistemicMI", "BioVar", "RepMarginStd")
SHRINK_K = 20.0


def safe_std(values: np.ndarray) -> float:
    value = float(np.nanstd(np.asarray(values, float), ddof=0))
    return value if value > 1e-12 else 1.0


def fit_gate(correct_id: pd.DataFrame) -> dict:
    """Match the author's original z-gate definitions, fitted within a fold."""
    if not correct_id.role.eq("id").all() or not correct_id.correct.eq(1).all():
        raise ValueError("Gate reference must contain correct known-class calls only")
    global_margin = float(np.nanpercentile(correct_id.Margin, 10))
    margin_std = safe_std(correct_id.Margin)
    global_stats = {
        field: (float(np.nanmean(correct_id[field])), safe_std(correct_id[field]))
        for field in GATE_SIGNALS
    }
    classes = {}
    for predicted_class, group in correct_id.groupby("PredictedMoA"):
        n = len(group)
        weight = n / (n + SHRINK_K)
        class_margin = float(np.nanpercentile(group.Margin, 10))
        tau = weight * class_margin + (1 - weight) * global_margin
        fields = {}
        for field in GATE_SIGNALS:
            mu_g, sd_g = global_stats[field]
            fields[field] = (
                weight * float(np.nanmean(group[field])) + (1 - weight) * mu_g,
                weight * safe_std(group[field]) + (1 - weight) * sd_g,
            )
        classes[predicted_class] = {"tau": tau, "fields": fields}
    return {
        "global_margin": global_margin,
        "margin_std": margin_std,
        "global_stats": global_stats,
        "classes": classes,
    }


def gate_risk(table: pd.DataFrame, params: dict, *, class_conditional: bool) -> np.ndarray:
    values = np.zeros(len(table), float)
    for index, row in enumerate(table.itertuples(index=False)):
        class_ref = params["classes"].get(row.PredictedMoA) if class_conditional else None
        tau = class_ref["tau"] if class_ref is not None else params["global_margin"]
        stats = class_ref["fields"] if class_ref is not None else params["global_stats"]
        z_values = [
            (float(getattr(row, field)) - stats[field][0]) / (stats[field][1] + 1e-9)
            for field in GATE_SIGNALS
        ]
        valid = np.asarray(z_values, float)
        positive_z = max(float(np.nanmax(valid)), 0.0) if np.isfinite(valid).any() else 0.0
        margin_penalty = max((tau - float(row.Margin)) / (params["margin_std"] + 1e-9), 0.0)
        values[index] = positive_z + margin_penalty
    return values


def cutoff(correct_id_scores: np.ndarray) -> float:
    """Use the strict higher-quantile rule of the nested reference analysis."""
    if len(correct_id_scores) < 5 or not np.isfinite(correct_id_scores).all():
        raise ValueError("Invalid correct-ID calibration scores")
    return float(np.nextafter(np.quantile(correct_id_scores, 0.9, method="higher"), np.inf))


def paired_bootstrap_rates(
    calls: pd.DataFrame, weights: np.ndarray, compound_order: list[str], method: str
) -> dict:
    part = calls.loc[calls.method.eq(method)]
    id_rows = part.loc[part.role.eq("id")].set_index("CompoundID").loc[compound_order]
    ood_rows = part.loc[part.role.eq("ood")].set_index("CompoundID").loc[compound_order]
    if len(id_rows) != 51 or len(ood_rows) != 51:
        raise ValueError("Incomplete paired compound roles")
    known_correct = id_rows.correct.to_numpy(int)
    id_flags = id_rows.flag.to_numpy(int)
    ood_flags = ood_rows.flag.to_numpy(int)
    ood_draw = (weights @ ood_flags) / weights.sum(axis=1)
    denominator = weights @ known_correct
    id_draw = np.divide(
        weights @ (id_flags * known_correct),
        denominator,
        out=np.full(len(denominator), np.nan),
        where=denominator > 0,
    )
    if not np.isfinite(id_draw).all():
        raise ValueError("Bootstrap draw without any correctly classified ID compounds")
    return {
        "ood_count": int(ood_flags.sum()),
        "ood_n": len(ood_rows),
        "ood_detection": float(ood_flags.mean()),
        "ood_ci_low": float(np.quantile(ood_draw, 0.025)),
        "ood_ci_high": float(np.quantile(ood_draw, 0.975)),
        "correct_id_count": int((id_flags * known_correct).sum()),
        "correct_id_n": int(known_correct.sum()),
        "correct_id_false_alarm": float((id_flags * known_correct).sum() / known_correct.sum()),
        "correct_id_ci_low": float(np.quantile(id_draw, 0.025)),
        "correct_id_ci_high": float(np.quantile(id_draw, 0.975)),
        "all_id_count": int(id_flags.sum()),
        "all_id_n": len(id_rows),
    }
