"""Small fixed treatment-time and MIC adjustments to control disagreement."""

import numpy as np
import pandas as pd


def time_mismatch(minutes: float, reference: float, fold_cap: float) -> float:
    """Capped absolute log2-fold departure from the reference exposure."""
    if minutes <= 0 or reference <= 0 or fold_cap <= 1:
        raise ValueError("Time and fold cap must be positive, with cap > 1")
    return min(1.0, abs(np.log2(minutes / reference)) / np.log2(fold_cap))


def interval_for_time(minutes: float, design: dict) -> tuple[float, float, float]:
    if pd.isna(minutes):
        return np.nan, 0.0, 1.0
    value = time_mismatch(float(minutes), design["reference_minutes"], design["time_fold_cap"])
    return value, value, value


def interval_for_mic(category: str, provenance: str, design: dict) -> tuple[float, float, float]:
    if category == "unknown":
        return np.nan, 0.0, 1.0
    if category not in design["mic_category_mismatch"]:
        raise ValueError(f"Unrecognized MIC category: {category}")
    value = float(design["mic_category_mismatch"][category])
    if provenance.startswith("provisional"):
        return value, 0.0, 1.0
    return value, value, value


def case_times(cases: pd.DataFrame, metadata: dict) -> pd.Series:
    by_dataset = {row["id"]: row["conditions"]["exposure_minutes"] for row in metadata["rows"]}
    if set(cases.dataset) != set(by_dataset):
        raise ValueError("Study metadata and case inventory differ")
    result = []
    for row in cases.itertuples():
        if row.dataset == "smx":
            if not row.case.endswith("h") or not row.case[:-1].isdigit():
                raise ValueError(f"Unexpected SMX time label: {row.case}")
            result.append(float(row.case[:-1]) * 60.0)
        else:
            result.append(by_dataset[row.dataset])
    return pd.Series(result, index=cases.index, dtype=float)
