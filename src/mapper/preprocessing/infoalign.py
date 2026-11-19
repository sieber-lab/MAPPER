"""Generate 300-dimensional pretrained InfoAlign molecular fingerprints."""

from __future__ import annotations

import numpy as np
import pandas as pd


class InfoAlignFingerprint:
    def __init__(self, model_path):
        from infoalign.representer import InfoAlignRepresenter

        self.model = InfoAlignRepresenter(model_path=model_path)
        self.embedding_dim = 300

    def calculate_single(self, smiles):
        return self.model.predict(smiles)

    def calculate_batch(self, smiles_list, index):
        fingerprints = []
        for i, smiles in enumerate(smiles_list):
            try:
                fingerprint = np.asarray(self.calculate_single(smiles)).flatten()
            except Exception as error:
                print(f"[Warning] Skipping sample {index[i]}: {error}")
                fingerprint = np.full(self.embedding_dim, np.nan)
            fingerprints.append(fingerprint)
        result = pd.DataFrame(fingerprints, index=index)
        if result.shape[1] != self.embedding_dim:
            raise ValueError("Expected a 300-dimensional InfoAlign representation")
        return result
