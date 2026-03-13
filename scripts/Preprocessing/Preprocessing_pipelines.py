import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# --- Import your modules ---
from Codebase.Proteomics_preprocessing import prepare_proteomics_data, label_dict
from Codebase.Chembio_preprocessing import prepare_compound_data_from_excels, generate_mic_feature_matrix, generate_growth_curve_matrix, ChemicalRepresenter
from Codebase.Cellbio_preprocessing import CellPaintRepresenter
from Codebase.Text_processing import augment_moa_descriptions, compute_llm_perplexities, generate_text_embeddings


def main():
    """
    Main function to run the full data preprocessing pipeline.
    """
    # --- Config ---
    data_date = datetime.today().strftime('%y%m%d')  # e.g., '250428'
    compound_data_folder = "Data/Compound_data/"
    embedding_folder = "Embeddings/"

    # --- Step 1: Proteomics preprocessing ---
    print("🔹 Running proteomics preprocessing...")
    fasta_path = "Data/Fasta_files/Ecoli.fasta"
    data_folder = "Data/MS_files/"
    X_proteomics, _, y_labels = prepare_proteomics_data(fasta_path, data_folder, label_dict, exclude_labels=[])
    assert X_proteomics.shape[0] == len(y_labels), "Mismatch between X_proteomics and y_labels dimensions."
    assert X_proteomics.shape[1] == 4401, "Expected 300 features in X_proteomics."
    # Save proteomics data
    X_proteomics.to_pickle(os.path.join(compound_data_folder, f"{data_date}_preprocessed_X_proteomics.pkl"))
    y_labels.to_pickle(os.path.join(compound_data_folder, f"{data_date}_preprocessed_y_labels.pkl"))

    # --- Step 2: Compound data preprocessing ---
    print("🔹 Preparing compound dictionary...")
    compound_dict = prepare_compound_data_from_excels(
        growth_curve_file=os.path.join(compound_data_folder, "250808_compounds.xlsx"),
        mic_file=os.path.join(compound_data_folder, "250808_compounds.xlsx"),
        smiles_file=os.path.join(compound_data_folder, "250808_compounds.xlsx"),
        output_pickle_path=None
    )

    # Save compound dictionary
    with open(os.path.join(compound_data_folder, f"{data_date}_preprocessed_compound_dict.pkl"), 'wb') as f:
        pickle.dump(compound_dict, f)

    # --- Step 3: Feature generation ---
    print("🔹 Generating feature matrices...")
    samples = X_proteomics.index
    smiles_list = [compound_dict[idx.split('_')[0]]['smiles'] for idx in samples]

    rep_ecfp = ChemicalRepresenter(representation="ecfp")
    X_ecfp = rep_ecfp.calculate_batch(smiles_list, samples)
    assert X_ecfp.shape[1] == 2048, "Expected 2048 features in X_ecfp."
    assert X_ecfp.shape[0] == len(samples), "Mismatch between X_ecfp and samples dimensions."
    rep_rdkit = ChemicalRepresenter(representation="rdkit")
    X_rdkit = rep_rdkit.calculate_batch(smiles_list, samples)
    assert X_rdkit.shape[1] == 210, "Expected 2048 features in X_rdkit."
    assert X_rdkit.shape[0] == len(samples), "Mismatch between X_rdkit and samples dimensions."
    rep_cellpaint = CellPaintRepresenter()
    X_cellpaint = rep_cellpaint.calculate_batch(smiles_list, samples)
    assert X_cellpaint.shape[1] == 300, "Expected 300 features in X_cellpaint."
    assert X_cellpaint.shape[0] == len(samples), "Mismatch between X_cellpaint and samples dimensions."
    X_mic = generate_mic_feature_matrix(compound_dict, X_proteomics=X_proteomics)
    assert X_mic.shape[1] == 20, "Expected 20 features in X_mic."
    assert X_mic.shape[0] == len(samples), "Mismatch between X_mic and samples dimensions."
    X_gc = generate_growth_curve_matrix(compound_dict, X_proteomics=X_proteomics)
    assert X_gc.shape[1] == 10, "Expected 10 features in X_gc."
    assert X_gc.shape[0] == len(samples), "Mismatch between X_gc and samples dimensions."
    # Save feature matrices
    X_ecfp.to_pickle(os.path.join(compound_data_folder, f"{data_date}_preprocessed_X_ecfp.pkl"))
    X_rdkit.to_pickle(os.path.join(compound_data_folder, f"{data_date}_preprocessed_X_rdkit.pkl"))
    X_cellpaint.to_pickle(os.path.join(compound_data_folder, f"{data_date}_preprocessed_X_cellpaint.pkl"))
    X_mic.to_pickle(os.path.join(compound_data_folder, f"{data_date}_preprocessed_X_mic.pkl"))
    X_gc.to_pickle(os.path.join(compound_data_folder, f"{data_date}_preprocessed_X_gc.pkl"))

    # --- Step 4: Text augmentation ---
    print("🔹 Performing text augmentation...")
    augment_moa_descriptions(
        input_excel_path=os.path.join(compound_data_folder, "MoA_descriptions.xlsx"),
        sentence_sheet_name="SentenceAugmentation",
        word_sheet_name="WordAugmentation",
        n_augmented=9
    )

    # --- Step 5: Text embeddings ---
    print("🔹 Generating text embeddings...")
    model_checkpoints = {
        "PubChemDeBERTa": "mschuh/PubChemDeBERTa-augmented",
        "BERT": "google-bert/bert-base-uncased",
        "BioBERT": "dmis-lab/biobert-v1.1",
    }

    # Semantic augmentation embeddings
    generate_text_embeddings(
        input_excel_path=os.path.join(compound_data_folder, "MoA_descriptions.xlsx"),
        sheet_name="SemanticAugmentation",
        model_checkpoints=model_checkpoints,
        output_pickle_path=os.path.join(embedding_folder, f"{data_date}_semanticAugmentation_embeddings.pkl")
    )

    # Sentence augmentation embeddings
    generate_text_embeddings(
        input_excel_path=os.path.join(compound_data_folder, "MoA_descriptions.xlsx"),
        sheet_name="SentenceAugmentation",
        model_checkpoints=model_checkpoints,
        output_pickle_path=os.path.join(embedding_folder, f"{data_date}_sentenceAugmentation_embeddings.pkl")
    )

    # Word augmentation embeddings
    generate_text_embeddings(
        input_excel_path=os.path.join(compound_data_folder, "MoA_descriptions.xlsx"),
        sheet_name="WordAugmentation",
        model_checkpoints=model_checkpoints,
        output_pickle_path=os.path.join(embedding_folder, f"{data_date}_wordAugmentation_embeddings.pkl")
    )

    print("\n✅ Full data preprocessing pipeline completed!")



if __name__ == "__main__":
    main()