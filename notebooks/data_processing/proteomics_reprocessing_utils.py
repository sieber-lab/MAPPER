from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


label_dict = {
    "JH1": "Cell wall biosynthesis",
    "JH2": "Cell wall biosynthesis",
    "JH3": "Cell wall biosynthesis",
    "JH4": "Cell wall biosynthesis",
    "JH5": "Cell wall biosynthesis",
    "JH6": "Cell wall biosynthesis",
    "JH7": "Cell wall biosynthesis",
    "JH8": "Cell wall biosynthesis",
    "JH9": "Cell wall biosynthesis",
    "JH10": "Cell wall biosynthesis",
    "JH0": "Cell wall biosynthesis",
    "JH11": "Protein synthesis",
    "JH12": "Protein synthesis",
    "JH13": "Protein synthesis",
    "JH14": "Protein synthesis",
    "JH15": "Protein synthesis",
    "JH16": "Protein synthesis",
    "JH17": "Protein synthesis",
    "JH18": "Protein synthesis",
    "JH19": "Protein synthesis",
    "JH55": "Protein synthesis",
    "JH56": "Protein synthesis",
    "JH20": "Folic acid metabolism",
    "JH21": "Folic acid metabolism",
    "JH22": "Folic acid metabolism",
    "JH23": "Folic acid metabolism",
    "JH24": "Folic acid metabolism",
    "JH25": "Folic acid metabolism",
    "JH57": "Folic acid metabolism",
    "JH63": "Folic acid metabolism",
    "JH26": "DNA replication",
    "JH27": "DNA replication",
    "JH28": "DNA replication",
    "JH29": "DNA replication",
    "JH30": "DNA replication",
    "JH32": "DNA replication",
    "JH33": "DNA replication",
    "JH34": "DNA replication",
    "JH35": "DNA replication",
    "JH36": "DNA replication",
    "JH68": "DNA replication",
    "JH70": "DNA replication",
    "JH72": "DNA replication",
    "JH31": "Nitrofurans",
    "JH58": "Nitrofurans",
    "JH64": "Nitrofurans",
    "JH65": "Nitrofurans",
    "JH37": "RNA Synthesis",
    "JH38": "RNA Synthesis",
    "JH39": "RNA Synthesis",
    "JH40": "Fatty acid synthesis",
    "JH41": "Fatty acid synthesis",
    "JH42": "Fatty acid synthesis",
    "JH61": "Fatty acid synthesis",
    "JH43": "Membrane disruption",
    "JH44": "Membrane disruption",
    "JH45": "Membrane disruption",
    "JH71": "Membrane disruption",
    "JH46": "Membrane disruption",
    "JH47": "Membrane disruption",
    "JH66": "Membrane disruption",
    "JH67": "Membrane disruption",
    "JH48": "lipopolysaccharide synthesis",
    "JH49": "lipopolysaccharide synthesis",
    "JH50": "lipopolysaccharide synthesis",
    "JH62": "lipopolysaccharide synthesis",
    "JH51": "Random",
    "JH53": "Random",
    "D": "DMSO",
    "H": "DMSO",
}


GROUP_TO_JHNUMS = {
    "DM": [68, 67, 66, 43],
    "DL": [65, 64, 53, 51],
    "DK": [62, 61, 58, 21, 20],
    "DI": [12, 3, 0],
    "10": [10],
    "DH": [18, 15, 14, 8, 6],
    "HH": [44],
    "DG": [47, 46, 45],
    "DF": [39, 37, 33, 32, 31],
    "HC": [29],
    "DE": [26, 9, 7],
    "DD": [16, 5, 2, 23],
    "DC": [38, 30, 24],
    "DB": [56, 50, 49, 48, 40, 42],
    "DA": [41, 28],
    "H": [27],
    "Punkt_D": [36, 35],
    "D": [13, 17, 55],
}


def parse_uniprot_fasta_to_df(fasta_file: str | Path) -> pd.DataFrame:
    records = []
    with open(fasta_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            description = line[1:].strip()
            gene_name = re.search(r"GN=([\w\d\-]+)", description)
            gene_name = gene_name.group(1) if gene_name else None
            records.append({"Gene_Name": gene_name})
    return pd.DataFrame(records)


def is_intensity_column(col_name: str) -> bool:
    return not col_name.startswith(("C: ", "N: ", "T: "))


def filter_intensity_and_metadata(
    df: pd.DataFrame, metadata_cols: list[str] | None = None
) -> pd.DataFrame:
    if metadata_cols is None:
        metadata_cols = ["T: Genes"]
    intensity_cols = [c for c in df.columns if is_intensity_column(c)]
    return df[intensity_cols + metadata_cols]


def expand_gene_rows(df: pd.DataFrame, gene_col: str = "T: Genes") -> pd.DataFrame:
    expanded_rows = []
    for _, row in df.iterrows():
        genes = str(row[gene_col]).split(";")
        for gene in genes:
            new_row = row.copy()
            new_row[gene_col] = gene.strip()
            expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)


def create_combined_fingerprint(
    filtered_dataframes: list[pd.DataFrame], fasta_df: pd.DataFrame
) -> pd.DataFrame:
    all_genes = sorted(fasta_df["Gene_Name"].dropna().unique())
    combined_df = pd.DataFrame(index=all_genes)

    for df in filtered_dataframes:
        df = df.rename(columns={"T: Genes": "Gene_Name"})
        df = expand_gene_rows(df, gene_col="Gene_Name")
        replicate_cols = [c for c in df.columns if c != "Gene_Name"]

        for col in replicate_cols:
            temp_df = df[["Gene_Name", col]].copy().set_index("Gene_Name")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                combined_df[col] = temp_df[col]

    return combined_df.fillna(0)


def preprocess_X(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(",", ".", regex=True).apply(pd.to_numeric, errors="coerce")


def simplify_sample_names(index_like: pd.Index | pd.Series) -> pd.Index:
    names = []
    for raw in map(str, index_like):
        parts = raw.split("_")
        if len(parts) < 2:
            names.append(raw)
            continue
        names.append(f"JH{parts[-2]}_{parts[-1]}")
    return pd.Index(names)


def map_labels(index_series: pd.Series, labels: dict, fallback: str = "Unknown") -> pd.Series:
    def lookup(sample_name: str) -> str:
        short = sample_name.split("_")[0]
        return labels.get(short, fallback)

    return index_series.map(lookup)


def build_proteomics_matrix(
    fasta_path: str | Path,
    export_path: str | Path,
    labels: dict | None = None,
    exclude_labels: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if labels is None:
        labels = label_dict
    if exclude_labels is None:
        exclude_labels = []

    fasta_df = parse_uniprot_fasta_to_df(fasta_path)
    df = pd.read_csv(export_path, sep="\t")
    filtered = filter_intensity_and_metadata(df, metadata_cols=["T: Genes"])
    combined = create_combined_fingerprint([filtered], fasta_df)

    X = combined.T.copy()
    X.index = simplify_sample_names(X.index)
    X["Label"] = map_labels(X.index.to_series(), labels)
    X = X[~X["Label"].isin(exclude_labels + ["Unknown"])]

    y = X["Label"].copy()
    X = preprocess_X(X.drop(columns=["Label"]))

    if X.shape[1] != 4401:
        warnings.warn(f"Expected 4401 proteomics features, found {X.shape[1]}.")
    return X, y


def _auto_gene_col(df: pd.DataFrame, default: str = "T: Genes") -> str:
    if default in df.columns:
        return default
    candidates = [
        c
        for c in df.columns
        if c.strip().lower() in {"t: genes", "genes", "gene", "gene names", "majority protein ids"}
    ]
    if not candidates:
        raise KeyError(f"Could not find gene column in reference table: {default}")
    return candidates[0]


def safe_simplify_reference_names(index_like) -> pd.Index:
    out = []
    for name in map(str, index_like):
        n = name.strip()
        parts = n.split("_")
        jh_parts = [p for p in parts if re.fullmatch(r"JH\d+", p)]
        if jh_parts and len(parts) >= 2:
            out.append(f"{jh_parts[-1]}_{parts[-1]}")
        elif len(parts) >= 2:
            out.append(f"{parts[-2]}_{parts[-1]}")
        else:
            out.append(n)
    return pd.Index(out)


def clean_reference_names(index_like: pd.Index) -> pd.Index:
    cleaned = []
    for raw in map(str, index_like):
        n = raw.strip()

        m = re.match(r"^(DA)(\d+)_G[A-Z]\d+$", n)
        if m:
            cleaned.append(f"{m.group(1)}_{m.group(2)}")
            continue

        m = re.match(r"^(DB)(\d+)_G[A-Z]\d+$", n)
        if m:
            cleaned.append(f"{m.group(1)}_{m.group(2)}")
            continue

        m = re.match(r"^Ctrl[_\s]?10[_\-]?(\d+)$", n)
        if m:
            cleaned.append(f"Ctrl_10_{m.group(1)}")
            continue

        cleaned.append(n)
    return pd.Index(cleaned)


def build_reference_matrix(fasta_path: str | Path, reference_export_path: str | Path) -> pd.DataFrame:
    fasta_df = parse_uniprot_fasta_to_df(fasta_path)
    df = pd.read_csv(reference_export_path, sep="\t", dtype=str)
    gene_col = _auto_gene_col(df)
    filtered = filter_intensity_and_metadata(df, metadata_cols=[gene_col]).rename(
        columns={gene_col: "T: Genes"}
    )
    expanded = expand_gene_rows(filtered, gene_col="T: Genes")
    meas_cols = [c for c in expanded.columns if c != "T: Genes"]
    expanded[meas_cols] = expanded[meas_cols].replace(",", ".", regex=True).apply(
        pd.to_numeric, errors="coerce"
    )

    combined = create_combined_fingerprint([expanded], fasta_df)
    X_ref = combined.T.copy()
    X_ref.index = safe_simplify_reference_names(X_ref.index)
    X_ref = preprocess_X(X_ref).fillna(0.0)
    X_ref.index = clean_reference_names(X_ref.index)
    if X_ref.index.duplicated().any():
        X_ref = X_ref.groupby(X_ref.index).mean(numeric_only=True).sort_index()
    return X_ref


def refs_for_group(group: str, ref_index: pd.Index) -> list[str]:
    g = re.escape(group)
    pat_underscore = re.compile(rf"^{g}_[0-9]+$")
    pat_nounders = re.compile(rf"^{g}[0-9]+$")
    return sorted(
        name for name in ref_index if pat_underscore.match(str(name)) or pat_nounders.match(str(name))
    )


def ttest_equalvar_ignore_zeros(comp_vals, ref_vals, alpha: float = 0.05) -> tuple[bool, float]:
    comp = np.asarray(comp_vals, dtype=float)
    ref = np.asarray(ref_vals, dtype=float)
    comp = comp[comp != 0]
    ref = ref[ref != 0]

    n_c = comp.size
    n_r = ref.size

    if n_r < 2:
        return False, 1.0

    if n_c >= 2:
        _t, p = stats.ttest_ind(comp, ref, equal_var=True, alternative="two-sided")
        if np.isnan(p):
            return False, 1.0
        return p <= alpha, float(p)

    if n_c == 1 and n_r >= 3:
        x = comp[0]
        m_r = np.mean(ref)
        s_r = np.std(ref, ddof=1)
        if s_r <= 0:
            return False, 1.0
        t = (x - m_r) / (s_r * np.sqrt(1.0 + 1.0 / n_r))
        df = n_r - 1
        p = 2.0 * stats.t.sf(np.abs(t), df)
        return p <= alpha, float(p)

    return False, 1.0


def parse_jh_number(sample_name: str) -> int | None:
    m = re.match(r"^JH(\d+)_", str(sample_name))
    return int(m.group(1)) if m else None


def build_significant_matrix(X_samples: pd.DataFrame, X_ref: pd.DataFrame) -> pd.DataFrame:
    group_to_refs = {g: refs_for_group(g, X_ref.index) for g in GROUP_TO_JHNUMS}
    jh_to_group = {n: g for g, nums in GROUP_TO_JHNUMS.items() for n in nums}

    compound_sets: dict[str, list[str]] = {}
    for sample in X_samples.index:
        jh = parse_jh_number(sample)
        if jh is None:
            continue
        compound_sets.setdefault(f"JH{jh}", []).append(sample)

    X_filtered_sig = X_samples.copy()
    for comp_key, comp_rows in sorted(compound_sets.items(), key=lambda kv: int(kv[0][2:])):
        jh = int(comp_key[2:])
        ctrl_group = jh_to_group.get(jh)
        if ctrl_group is None:
            continue
        ref_rows = group_to_refs.get(ctrl_group, [])
        if not ref_rows:
            continue

        Xc = X_samples.loc[comp_rows]
        Xr = X_ref.loc[ref_rows]
        genes = Xc.columns.intersection(Xr.columns)
        Xc = Xc[genes].copy()
        Xr = Xr[genes].copy()

        sig_by_gene = pd.Series(False, index=genes)
        for gene in genes:
            is_sig, _p = ttest_equalvar_ignore_zeros(Xc[gene].values, Xr[gene].values, alpha=0.05)
            sig_by_gene[gene] = is_sig

        sig_cols = sig_by_gene.index[sig_by_gene.values]
        nonsig_cols = sig_by_gene.index[~sig_by_gene.values]
        X_filtered_sig.loc[comp_rows, sig_cols] = X_samples.loc[comp_rows, sig_cols]
        X_filtered_sig.loc[comp_rows, nonsig_cols] = 0.0

    return X_filtered_sig


def split_like_reference(
    full_matrix: pd.DataFrame,
    base_reference: pd.DataFrame,
    special_reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = full_matrix.reindex(base_reference.index)
    special = full_matrix.reindex(special_reference.index)

    if base.isna().all(axis=1).any() or special.isna().all(axis=1).any():
        missing = list(base.index[base.isna().all(axis=1)]) + list(
            special.index[special.isna().all(axis=1)]
        )
        raise KeyError(f"Missing rows after split: {missing[:10]}")

    base = base.loc[:, base_reference.columns]
    special = special.loc[:, special_reference.columns]
    return base.fillna(0.0), special.fillna(0.0)


def compare_dataframes(name: str, observed: pd.DataFrame, expected: pd.DataFrame) -> dict:
    result = {
        "name": name,
        "same_shape": observed.shape == expected.shape,
        "same_index": observed.index.equals(expected.index),
        "same_columns": observed.columns.equals(expected.columns),
        "exact_equal": observed.equals(expected),
        "observed_shape": observed.shape,
        "expected_shape": expected.shape,
        "max_abs_diff": np.nan,
        "n_diff_cells": np.nan,
    }

    if (
        observed.shape == expected.shape
        and observed.index.equals(expected.index)
        and observed.columns.equals(expected.columns)
    ):
        try:
            diff = observed.astype(float) - expected.astype(float)
            result["max_abs_diff"] = float(np.nanmax(np.abs(diff.to_numpy())))
            result["n_diff_cells"] = int((diff != 0).sum().sum())
        except Exception:
            result["max_abs_diff"] = np.nan
            result["n_diff_cells"] = np.nan
    return result


def compare_tsv_files(name: str, observed_path: str | Path, expected_path: str | Path) -> dict:
    observed = pd.read_csv(observed_path, sep="\t", dtype=str)
    expected = pd.read_csv(expected_path, sep="\t", dtype=str)
    result = {
        "name": name,
        "same_shape": observed.shape == expected.shape,
        "same_columns": observed.columns.equals(expected.columns),
        "exact_equal": observed.equals(expected),
        "observed_shape": observed.shape,
        "expected_shape": expected.shape,
        "n_diff_cells": np.nan,
    }
    if observed.shape == expected.shape and observed.columns.equals(expected.columns):
        result["n_diff_cells"] = int((observed != expected).sum().sum())
    return result
