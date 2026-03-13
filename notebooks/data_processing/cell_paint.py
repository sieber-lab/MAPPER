from infoalign.representer import InfoAlignRepresenter
import pandas as pd
import numpy as np

class CellPaintRepresenter:
    """
    Wrapper for generating cell painting representations using a pretrained InfoAlign model.
    """
    def __init__(self, model_path='infoalign_model/pretrain.pt'):
        """
        Initialize the InfoAlign model.

        Parameters:
            model_path (str): Path to the pretrained InfoAlign model.
        """
        self.model = InfoAlignRepresenter(model_path=model_path)
        self.embedding_dim = 300

    def calculate_single(self, smiles: str):
        """
        Generate the cell painting representation for a single SMILES.

        Parameters:
            smiles (str): SMILES string of the compound.

        Returns:
            np.ndarray: Cell painting embedding.
        """
        return self.model.predict(smiles)

    def calculate_batch(self, smiles_list, index):
        """
        Batch generate cell painting embeddings.

        Parameters:
            smiles_list (list of str): List of SMILES strings.
            index (list-like): Sample identifiers matching SMILES.

        Returns:
            pd.DataFrame: Cell painting fingerprint matrix.
        """
        fingerprints = []
        for i, smi in enumerate(smiles_list):
            try:
                fp = self.calculate_single(smi)
                fp_flat = np.array(fp).flatten()
            except Exception as e:
                print(f"[Warning] Skipping sample {index[i]}: {e}")
                fp_flat = np.full(self.embedding_dim, np.nan)
            fingerprints.append(fp_flat)

        final_df = pd.DataFrame(fingerprints, index=index)
        assert final_df.shape[1] == 300, f"Expected {300} features, got {final_df.shape[1]}"

        return final_df