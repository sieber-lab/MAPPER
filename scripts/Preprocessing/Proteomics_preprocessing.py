# proteomics_preprocessing.py

import os
import re
import pandas as pd
import numpy as np
from typing import List
from Bio import SeqIO
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings

# --- Loading and Parsing ---

def parse_uniprot_fasta_to_df(fasta_file: str) -> pd.DataFrame:
    """
    Parse a UniProt FASTA file into a DataFrame with metadata and sequence information.

    Parameters:
        fasta_file (str): Path to the FASTA file.

    Returns:
        pd.DataFrame: Parsed UniProt metadata and sequences.
    """
    records = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        description = record.description
        uniprot_id = record.id.split('|')[1]
        protein_name = record.id.split('|')[2]
        full_description = " ".join(description.split()[1:])
        organism = re.search(r'OS=([^=]+) OX=', description)
        organism = organism.group(1) if organism else None
        tax_id = re.search(r'OX=(\d+)', description)
        tax_id = tax_id.group(1) if tax_id else None
        gene_name = re.search(r'GN=([\w\d\-]+)', description)
        gene_name = gene_name.group(1) if gene_name else None
        protein_existence = re.search(r'PE=(\d+)', description)
        protein_existence = protein_existence.group(1) if protein_existence else None
        sequence_version = re.search(r'SV=(\d+)', description)
        sequence_version = sequence_version.group(1) if sequence_version else None

        records.append({
            "Gene_Name": gene_name
        })

    return pd.DataFrame(records)

# --- Filtering and Cleaning ---

def is_intensity_column(col_name: str) -> bool:
    """Check if a column name corresponds to an intensity column (exclude certain metadata prefixes)."""
    excluded_prefixes = ("C: ", "N: ", "T: ")
    return not col_name.startswith(excluded_prefixes)

def filter_intensity_and_metadata(df: pd.DataFrame, metadata_cols: List[str] = ["T: Genes"]) -> pd.DataFrame:
    """
    Filter DataFrame to retain only intensity columns and specified metadata columns.

    Parameters:
        df (pd.DataFrame): Input data.
        metadata_cols (list): List of metadata columns to retain.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    intensity_cols = [c for c in df.columns if is_intensity_column(c)]
    selected_cols = intensity_cols + metadata_cols
    return df[selected_cols]

def expand_gene_rows(df: pd.DataFrame, gene_col: str = "T: Genes") -> pd.DataFrame:
    """
    Expand rows with multiple gene names into one row per gene.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        gene_col (str): Column containing semicolon-separated gene names.

    Returns:
        pd.DataFrame: Expanded DataFrame.
    """
    expanded_rows = []
    for _, row in df.iterrows():
        genes = row[gene_col].split(';')
        for gene in genes:
            new_row = row.copy()
            new_row[gene_col] = gene.strip()
            expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)

# --- Feature Engineering ---

def create_combined_fingerprint(filtered_dataframes: List[pd.DataFrame], fasta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a gene-by-sample fingerprint matrix from multiple filtered proteomics dataframes.

    Parameters:
        filtered_dataframes (list of pd.DataFrame): Proteomics data tables.
        fasta_df (pd.DataFrame): FASTA-derived gene name list.

    Returns:
        pd.DataFrame: Gene-wise fingerprint matrix.
    """
    all_genes = sorted(fasta_df["Gene_Name"].dropna().unique())
    combined_df = pd.DataFrame(index=all_genes)

    for df in filtered_dataframes:
        df = df.rename(columns={"T: Genes": "Gene_Name"})
        df = expand_gene_rows(df, gene_col="Gene_Name")
        replicate_cols = [c for c in df.columns if c not in ["Gene_Name"]]

        for col in replicate_cols:
            temp_df = df[["Gene_Name", col]].copy().set_index("Gene_Name")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                combined_df[col] = temp_df[col]

    nan_count = combined_df.isna().sum().sum()
    if nan_count > 0:
        print(f"[Warning] {nan_count} missing values filled with 0 in combined fingerprint matrix.")

    return combined_df.fillna(0)

def simplify_sample_names(index_series: pd.Series) -> List[str]:
    """
    Simplify sample names by extracting last two underscore-separated elements.

    Parameters:
        index_series (pd.Series): Sample index series.

    Returns:
        list: Simplified sample names.
    """
    split_index = index_series.str.split("_", expand=False)
    return ["JH" + str(x[-2]) + "_" + str(x[-1]) for x in split_index]

def map_labels(index_series: pd.Series, label_dict: dict, fallback="Unknown") -> pd.Series:
    """
    Map sample names to class labels based on the first part before underscore.

    Parameters:
        index_series (pd.Series): Sample index series.
        label_dict (dict): Mapping dictionary.
        fallback (str): Fallback label if mapping fails.

    Returns:
        pd.Series: Mapped labels.
    """
    def lookup(sample_name):
        short = sample_name.split('_')[0]  # Take first part before underscore
        return label_dict.get(short, fallback)

    return index_series.apply(lookup)

def preprocess_X(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess feature matrix by replacing commas with dots and converting to numeric.

    Parameters:
        df (pd.DataFrame): Input feature DataFrame.

    Returns:
        pd.DataFrame: Cleaned feature matrix.
    """
    return (
        df.replace(',', '.', regex=True)
          .apply(pd.to_numeric, errors='coerce')
    )

def binarize_labels(y: pd.Series) -> (pd.Series, list):
    """
    Binarize labels into numeric codes.

    Parameters:
        y (pd.Series): Labels to binarize.

    Returns:
        tuple: (Binarized labels, list of original label names)
    """
    if y.dtype == "object":
        y_cat = y.astype("category")
        y_codes = y_cat.cat.codes
        return y_codes, y_cat.cat.categories.tolist()
    return y, None

# --- Saving/Loading (optional for convenience) ---


def save_pickle(obj, filename: str):
    """Save an object to a pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filename: str):
    """Load an object from a pickle file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


def prepare_proteomics_data(
    fasta_file: str,
    data_folder: str,
    label_dict: dict,
    exclude_labels: list = ["Fatty acid synthesis", "Folic acid metabolism"]
):
    """
    Preprocess proteomics data and return feature matrix X and labels y.

    Parameters:
        fasta_file (str): Path to UniProt FASTA file.
        data_folder (str): Path to the folder containing data tables.
        label_dict (dict): Mapping of sample IDs to labels.
        exclude_labels (list): Labels to exclude from final dataset.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Encoded labels.
    """
    # Load FASTA
    fasta_df = parse_uniprot_fasta_to_df(fasta_file)

    # Load and filter all data tables
    dataframes = []
    for file in os.listdir(data_folder):
        if file.endswith(".fasta") or os.path.isdir(os.path.join(data_folder, file)):
            continue
        df = pd.read_csv(os.path.join(data_folder, file), sep="\t")
        filtered = filter_intensity_and_metadata(df)
        dataframes.append(filtered)

    # Create combined fingerprint
    combined_df = create_combined_fingerprint(dataframes, fasta_df)
    combined_df_T = combined_df.T.copy()
    combined_df_T.index = simplify_sample_names(combined_df_T.index)

    # Label and filter
    combined_df_T["Label"] = map_labels(combined_df_T.index.to_series(), label_dict)
    final_df = combined_df_T[~combined_df_T["Label"].isin(exclude_labels +  ["Unknown"])]

    # Prepare X and y
    X = preprocess_X(final_df.drop(columns=["Label"]))
    y, class_mapping = binarize_labels(final_df["Label"])
    y_classes = final_df["Label"]
    # Filter out samples with invalid labels
    valid_idx = y != -1
    X = X[valid_idx]
    y = y[valid_idx]
    y_classes = y_classes[valid_idx]
    assert X.shape[0] == y.shape[0], "Mismatch between X and y dimensions."
    if X.shape[1] != 4401:
        warnings.warn(
            f"prepare_proteomics_data: expected 4401 features in X, but got {X.shape[1]}"
        )
    return X, y, y_classes


# --- Run the Preprocessing ---
label_dict = {
    # JH1..JH10
    'JH1': 'Cell wall biosynthesis',
    'JH2': 'Cell wall biosynthesis',
    'JH3': 'Cell wall biosynthesis',
    'JH4': 'Cell wall biosynthesis',
    'JH5': 'Cell wall biosynthesis',
    'JH6': 'Cell wall biosynthesis',
    'JH7': 'Cell wall biosynthesis',
    'JH8': 'Cell wall biosynthesis',
    'JH9': 'Cell wall biosynthesis',
    'JH10': 'Cell wall biosynthesis',
    'JH0': 'Cell wall biosynthesis',

    # JH11..JH19, JH55..JH56
    'JH11': 'Protein synthesis',
    'JH12': 'Protein synthesis',
    'JH13': 'Protein synthesis',
    'JH14': 'Protein synthesis',
    'JH15': 'Protein synthesis',
    'JH16': 'Protein synthesis',
    'JH17': 'Protein synthesis',
    'JH18': 'Protein synthesis',
    'JH19': 'Protein synthesis',
    'JH55': 'Protein synthesis',
    'JH56': 'Protein synthesis',

    # JH20..JH25, JH57
    'JH20': 'Folic acid metabolism',
    'JH21': 'Folic acid metabolism',
    'JH22': 'Folic acid metabolism',
    'JH23': 'Folic acid metabolism',
    'JH24': 'Folic acid metabolism',
    'JH25': 'Folic acid metabolism',
    'JH57': 'Folic acid metabolism',
    'JH63': 'Folic acid metabolism',

    # JH26..JH36
    'JH26': 'DNA replication',
    'JH27': 'DNA replication',
    'JH28': 'DNA replication',
    'JH29': 'DNA replication',
    'JH30': 'DNA replication',
    'JH31': 'DNA replication',
    'JH32': 'DNA replication',
    'JH33': 'DNA replication',
    'JH34': 'DNA replication',
    'JH35': 'DNA replication',
    'JH36': 'DNA replication',
    'JH68': 'DNA replication',
    'JH70': 'DNA replication',
    'JH72': 'DNA replication',

    #nitrofurans
    'JH31': 'Nitrofurans',
    'JH58': 'Nitrofurans',
    'JH64': 'Nitrofurans',
    'JH65': 'Nitrofurans',

    # JH37..JH39
    'JH37': 'RNA Synthesis',
    'JH38': 'RNA Synthesis',
    'JH39': 'RNA Synthesis',

    # JH40..JH42
    'JH40': 'Fatty acid synthesis',
    'JH41': 'Fatty acid synthesis',
    'JH42': 'Fatty acid synthesis',
    'JH61': 'Fatty acid synthesis',

    # JH43..JH45
    'JH43': 'Membrane disruption',
    'JH44': 'Membrane disruption',
    'JH45': 'Membrane disruption',
    'JH71': 'Membrane disruption',

    # JH46..JH47
    'JH46': 'Membrane disruption',
    'JH47': 'Membrane disruption',
    'JH66': 'Membrane disruption',
    'JH67': 'Membrane disruption',

    # JH48..JH50
    'JH48': 'lipopolysaccharide synthesis',
    'JH49': 'lipopolysaccharide synthesis',
    'JH50': 'lipopolysaccharide synthesis',
    'JH62': 'lipopolysaccharide synthesis',

    # random

    'JH51': 'Random',
    'JH53': 'Random',

    # DMSO
    'D': 'DMSO',
    'H': 'DMSO',
} # Use your existing label_dict here

