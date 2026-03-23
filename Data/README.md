# Data

Curated input data used by the notebooks and analysis scripts.

## Main subfolders

- `Base/` contains the primary modality matrices used for the main manuscript analyses.
- `Special/` contains derived or subsetted matrices used for specific modeling variants.
- `external_data/` contains processed inputs for the external transfer and cross-laboratory analyses.
- `eclipse_data/` contains processed inputs for the cross-instrument proteomics experiments.
- `Fasta_files/` and `MS_reference_files/` contain supporting proteomics reference inputs.

Most notebooks assume these tables exist locally and will read from this folder directly.
