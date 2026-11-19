import pandas as pd
import pickle
from typing import Dict
import os
from sklearn.preprocessing import MinMaxScaler
import useful_rdkit_utils as uru
from rdkit import Chem
import re
import numpy as np


def prepare_compound_data_from_excels(
    growth_curve_file: str, mic_file: str, smiles_file: str, output_pickle_path: str = None
) -> Dict[str, dict]:
    """
    Prepare and save compound_data dictionary by combining growth curves, MIC data, and SMILES information.

    Parameters:
        growth_curve_file (str): Path to Excel file containing growth curves and MoA info (e.g., "250324_compounds.xlsx").
        mic_file (str): Path to Excel file containing MIC data ("240814_inhouse_library.xlsx", "MIC Map" sheet).
        smiles_file (str): Path to Excel file containing SMILES info ("240814_inhouse_library.xlsx", "MCE data" sheet).
        output_pickle_path (str): Path to save the prepared pickle file.

    Returns:
        dict: compound_data dictionary.
    """
    # Define mapping of internal sample names to chemical names
    reverse_name_dict = {v: k for k, v in name_dict.items()}
    print(reverse_name_dict)
    # === Step 1: Parse Growth Curves and MoA ===
    df_growth = pd.read_excel(growth_curve_file, header=None, sheet_name="Growth Curves")
    df_growth = df_growth.iloc[:, [2] + list(range(15, 26))]
    df_growth = df_growth.dropna(subset=[16]).reset_index(drop=True)

    indices_with_zero = df_growth[df_growth[16] == 0].index.tolist()

    df_dict = {}

    for i, idx in enumerate(indices_with_zero):
        name = df_growth.iloc[idx + 1, 0]
        index = reverse_name_dict.get(name)
        if index is None:
            print(f"[Warning] {name} not found in reverse_name_dict. Skipping.")
            continue
        moa = df_growth.iloc[idx + 2, 0]
        next_idx = indices_with_zero[i + 1] if (i + 1) < len(indices_with_zero) else len(df_growth)
        sub_df = df_growth.iloc[idx:next_idx, 1:12].dropna()

        df_dict[index] = {"growth": sub_df, "mode_of_action": moa, "name": name}

    # === Step 2: Parse MIC Data ===
    df_mic = pd.read_excel(mic_file, sheet_name="MIC Map", header=None)
    df_mic = df_mic.drop(columns=[0])

    mic_blocks = [
        df_mic.iloc[i * 2 : (i * 2) + 2].reset_index(drop=True) for i in range(len(df_mic) // 2)
    ]

    for mic_df in mic_blocks:
        idx = mic_df.iloc[1, 0]
        if idx in df_dict:
            df_dict[idx]["mic"] = mic_df
        else:
            print(f"[Warning] MIC info for '{idx}' not found in df_dict. Skipping.")

    # === Step 3: Parse SMILES ===
    df_smiles = pd.read_excel(smiles_file, sheet_name="MCE data", header=None)
    df_smiles = df_smiles.iloc[6:]
    df_smiles.columns = df_smiles.iloc[0]
    df_smiles = df_smiles[1:]
    df_smiles = df_smiles[["ProductName", "Smiles"]]

    for idx in df_dict:
        match = df_smiles[df_smiles["ProductName"] == df_dict[idx]["name"]]
        if not match.empty:
            smiles = match.iloc[0]["Smiles"].split(".")[0]  # Keep first fragment
            df_dict[idx]["smiles"] = smiles
        else:
            print(f"[Warning] No SMILES found for {df_dict[idx]['name']}.")

    # === Step 4: Save
    if output_pickle_path is not None:
        if os.path.exists(output_pickle_path):
            print(f"[Warning] {output_pickle_path} already exists. Overwriting.")
        else:
            print(f"Saving compound_data to {output_pickle_path}...")
            with open(output_pickle_path, "wb") as f:
                pickle.dump(df_dict, f)
            print(f"✅ compound_data saved to {output_pickle_path}")

    return df_dict


class ChemicalRepresenter:
    """
    Wrapper for generating chemical structure-based representations.
    Supports ECFP and RDKit descriptors.
    """

    def __init__(self, representation="ecfp", radius=2, nBits=2048):
        """
        Initialize the representer.

        Parameters:
            representation (str): Either "ecfp" or "rdkit".
            radius (int): Radius for ECFP.
            nBits (int): Bit length for fingerprint.
        """
        self.representation = representation
        self.radius = radius
        self.nBits = nBits

    def calculate_single(self, smiles: str):
        """
        Generate fingerprint for a single SMILES.

        Parameters:
            smiles (str): SMILES string.

        Returns:
            np.ndarray: Fingerprint array.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        if self.representation == "ecfp":
            smi2fp = uru.Smi2Fp(radius=self.radius, fpSize=self.nBits)
            return smi2fp.get_np(smiles)

        elif self.representation == "rdkit":
            rdkit_desc = uru.RDKitDescriptors()
            return rdkit_desc.calc_smiles(smiles)

        raise ValueError(f"Unsupported representation: {self.representation}")

    def calculate_batch(self, smiles_list, index):
        """
        Batch generate chemical fingerprints.

        Parameters:
            smiles_list (list of str): SMILES strings.
            index (list-like): Sample identifiers.

        Returns:
            pd.DataFrame: Fingerprint matrix.
        """
        fingerprints = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            try:
                fp = self.calculate_single(smi)
            except Exception as e:
                print(f"[Warning] Skipping sample {i}: {e}")
                fp = np.full(self.nBits, np.nan)  # or zeros
            fingerprints.append(fp)
            valid_indices.append(index[i])
        final_df = pd.DataFrame(fingerprints, index=valid_indices)
        if self.representation == "ecfp":
            assert (
                final_df.shape[1] == self.nBits
            ), f"Expected {self.nBits} columns, got {final_df.shape[1]}"
        elif self.representation == "rdkit":
            assert final_df.shape[1] == 210, f"Expected 5 columns, got {final_df.shape[1]}"
        assert final_df.shape[0] == len(
            smiles_list
        ), f"Expected {len(smiles_list)} rows, got {final_df.shape[0]}"
        return final_df


def generate_growth_curve_matrix(
    loaded_dict, X_proteomics, drop_ctrl=True, final_timepoint_index=-1
):
    selected_growth_curves = {}
    mic_factors = []

    for compound, data in loaded_dict.items():
        if "growth" not in data:
            continue

        df = data["growth"].copy()
        # --- shape check: must be (n_rows, 11) → 1 label + 10 measurements
        if df.shape[1] != 11:
            raise ValueError(
                f"Growth DataFrame for {compound!r} must have 11 columns (1 label + 10 timepoints),"
                f" got {df.shape[1]}"
            )

        if drop_ctrl:
            df = df[~df.iloc[:, 0].str.lower().str.contains("dmso|control", na=False)]

        df = df.reset_index(drop=True)
        df.columns = range(df.shape[1])

        condition_names = df.iloc[:, 0]
        od_values = df.iloc[:, 1:].astype(float)
        ctrl_curve = (
            data["growth"][
                data["growth"].iloc[:, 0].str.lower().str.contains("dmso|control", na=False)
            ]
            .iloc[0, 1:]
            .astype(float)
            .values
        )

        selected_row_idx = None
        for idx, row in od_values.iterrows():
            if row.iloc[final_timepoint_index] < 0.5 * ctrl_curve[final_timepoint_index]:
                selected_row_idx = idx
                break

        if selected_row_idx is None:
            selected_row_idx = od_values.index[-1]

        selected_curve = od_values.loc[selected_row_idx].values
        condition_str = condition_names[selected_row_idx].lower().replace(",", ".")

        match = re.search(r"([\d\.]+)\s*x", condition_str)
        try:
            factor = float(match.group(1)) if match else 1.0
        except ValueError:
            factor = 1.0

        selected_growth_curves[compound] = selected_curve * factor
        mic_factors.append(factor)

    df_growth_features = pd.DataFrame.from_dict(selected_growth_curves, orient="index")

    filtered_growth_matrix = df_growth_features.loc[
        [idx.split("_")[0] for idx in X_proteomics.index]
    ]
    filtered_growth_matrix.index = X_proteomics.index

    filtered_growth_matrix.columns = [f"t{i}" for i in range(filtered_growth_matrix.shape[1])]
    assert (
        filtered_growth_matrix.shape[1] == 10
    ), f"Expected 10 timepoints, got {filtered_growth_matrix.shape[1]}"
    assert filtered_growth_matrix.shape[0] == len(
        X_proteomics
    ), f"Expected {len(X_proteomics)} samples, got {filtered_growth_matrix.shape[0]}"
    return filtered_growth_matrix


def generate_mic_feature_matrix(loaded_dict, od_threshold=0.1, X_proteomics=None):
    def generate_dilution_series(start=800.0, stop=0.001):
        dilutions = []
        current = start
        while current >= stop:
            dilutions.append(round(current, 6))
            current /= 2
        return dilutions

    dilution_buckets = generate_dilution_series()
    feature_matrix = {}
    for compound, data in loaded_dict.items():
        mic_df = data["mic"]

        # --- shape check: must be exactly 2 rows and 11 columns (1 idx + 10 measurements)
        if mic_df.shape != (2, 11):
            raise ValueError(
                f"MIC DataFrame for {compound!r} must have shape (2,11), got {mic_df.shape}"
            )

        concentrations = mic_df.iloc[0, 1:].astype(float).values
        ods = mic_df.iloc[1, 1:].astype(float).values
        conc_to_od = dict(zip(concentrations, ods))

        sorted_concs = sorted(conc_to_od.keys(), reverse=True)
        highest_measured = sorted_concs[0]
        lowest_measured = sorted_concs[-1]
        od_highest = conc_to_od[highest_measured]
        od_lowest = conc_to_od[lowest_measured]

        vector = []
        for bucket in dilution_buckets:
            if bucket in conc_to_od:
                vector.append(conc_to_od[bucket])
            elif bucket > highest_measured:
                vector.append(od_highest)
            elif bucket < lowest_measured:
                vector.append(od_lowest)
            else:
                # Find nearest lower concentration
                lower = max([c for c in conc_to_od.keys() if c < bucket], default=lowest_measured)
                vector.append(conc_to_od.get(lower, od_lowest))
        feature_matrix[compound] = vector

    df_mic_features = pd.DataFrame.from_dict(
        feature_matrix, orient="index", columns=dilution_buckets
    )

    scaler = MinMaxScaler()

    # Apply row-wise min-max scaling
    mic_feature_df_scaled = df_mic_features
    mic_feature_df_scaled.loc[:, :] = scaler.fit_transform(mic_feature_df_scaled.T).T
    # Filter MIC matrix to match proteomics sample order
    filtered_mic_matrix = mic_feature_df_scaled.loc[
        [idx.split("_")[0] for idx in X_proteomics.index]
    ]
    filtered_mic_matrix.index = X_proteomics.index  # to ensure perfect alignment
    assert (
        filtered_mic_matrix.shape[1] == 20
    ), f"Expected {20} columns, got {filtered_mic_matrix.shape[1]}"
    assert filtered_mic_matrix.shape[0] == len(
        X_proteomics
    ), f"Expected {len(X_proteomics)} rows, got {filtered_mic_matrix.shape[0]}"

    return filtered_mic_matrix


name_dict = {
    # JH1..JH10
    "REF": "Ampicillin (sodium)",
    "JH0": "Ampicillin (sodium)",
    "JH1": "Amoxicillin",
    "JH2": "Sulopenem",
    "JH3": "Cefodizime",
    "JH4": "Aztreonam",
    "JH5": "Doripenem",
    "JH6": "Imipenem (monohydrate)",
    "JH7": "Ertapenem (disodium)",
    "JH8": "Cefiderocol",
    "JH9": "Cefepime",
    "JH10": "Fosfomycin (calcium)",
    # JH11..JH19, JH55..JH56
    "JH11": "Retapamulin",
    "JH12": "Framycetin (sulfate)",
    "JH13": "Chloramphenicol",
    "JH14": "Tigecycline",
    "JH15": "Tetracycline",
    "JH16": "Azithromycin",
    "JH17": "Gentamicin sulfate",
    "JH18": "Eravacycline (dihydrochloride)",
    "JH19": "Apramycin (sulfate)",
    "JH55": "Kanamycin sulfate",
    "JH56": "Tobramycine",
    # JH20..JH25, JH57
    "JH20": "Sulfamonomethoxine",
    "JH21": "Sulfisoxazole",
    "JH22": "Sulfacetamide",
    "JH23": "Trimethoprim",
    "JH24": "Tetroxoprim",
    "JH25": "Sulfaguanidine",
    "JH57": "Sulphanilamide",
    "JH63": "Sulfadiazine",
    # JH26..JH36
    "JH26": "Gatifloxacin",
    "JH27": "Temafloxacin",
    "JH28": "Lascufloxacin",
    "JH29": "Ciprofloxacin",
    "JH30": "Levofloxacin",
    "JH31": "Nitrofurantoin",
    "JH32": "Gepotidacin",
    "JH33": "Nalidixic acid",
    "JH34": "Novobiocin",
    "JH35": "Flumequine",
    "JH36": "Enrofloxacin",
    # JH37..JH39
    "JH37": "Rifaximin",
    "JH38": "Rifampicin",
    "JH39": "Rifabutin",
    # JH40..JH42
    "JH40": "Triclosan",
    "JH41": "Cerulenin",
    "JH42": "MUT056399",
    "JH61": "Debio-1452-NH2",
    # JH43..JH45
    "JH43": "Colistin (sulfate)",
    "JH44": "Polymyxin B",
    "JH45": "Squalamine",
    # JH46..JH47
    "JH46": "Thanatin (TFA)",
    "JH47": "Lolamicin",
    # JH48..JH50
    "JH48": "PF-5081090",
    "JH49": "ACHN-975 (TFA)",
    "JH50": "CHIR-090",
    "JH62": "BB-78485",
    # JH51..JH72
    "JH51": "Xanthocillin",
    "JH52": "Clioquinol",
    "JH53": "Nitroxoline",
    "JH54": "Furosemide",
    "JH58": "Nitrofurazone",
    "JH59": "Luteolin",
    "JH60": "Actinonin",
    "JH64": "MAK-1-385",
    "JH65": "MAK-1-413",
    "JH66": "Darobactin D22",
    "JH67": "Myxovirescin",
    "JH68": "Cystobactamid CN-DM-861",
    "JH70": "Corramycin",
    "JH71": "eric_peptide",
    "JH72": "Closthioamide",
}
