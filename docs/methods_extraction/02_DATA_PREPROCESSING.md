# 02_DATA_PREPROCESSING

## Scope note
- Baseline training/eval scripts load precomputed pickles (`Data/Base/*.pkl`) and do not run raw proteomics preprocessing inline. [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: main :: `with open(f"{BASE_DIR}/X_proteomics.pkl"...)`]
- Raw-to-pickle preprocessing logic is implemented in `scripts/Preprocessing/*` and mirrored in `notebooks/data_processing/*`. [ref: code_repository/scripts/Preprocessing/Preprocessing_pipelines.py :: main :: `Running proteomics preprocessing...`]

## Raw-input transformations

### 1) Proteomics preprocessing
- FASTA parsing extracts `Gene_Name` records from UniProt headers. [ref: code_repository/scripts/Preprocessing/Proteomics_preprocessing.py :: parse_uniprot_fasta_to_df :: `records.append({"Gene_Name": gene_name})`]
- Input table filtering keeps columns not starting with prefixes `"C: "`, `"N: "`, `"T: "`, then re-adds metadata column `"T: Genes"`. [ref: code_repository/scripts/Preprocessing/Proteomics_preprocessing.py :: is_intensity_column/filter_intensity_and_metadata :: `excluded_prefixes = ("C: ", "N: ", "T: ")`]
- Rows with semicolon-separated genes are expanded one gene per row. [ref: code_repository/scripts/Preprocessing/Proteomics_preprocessing.py :: expand_gene_rows :: `genes = row[gene_col].split(';')`]
- Combined gene matrix missing values are filled with `0`. [ref: code_repository/scripts/Preprocessing/Proteomics_preprocessing.py :: create_combined_fingerprint :: `return combined_df.fillna(0)`]
- Sample names are normalized to format `JH<id>_<rep>` using last two underscore fragments. [ref: code_repository/scripts/Preprocessing/Proteomics_preprocessing.py :: simplify_sample_names :: `"JH" + str(x[-2]) + "_" + str(x[-1])`]
- Labels are mapped from prefix before underscore via `label_dict`; fallback is `"Unknown"`. [ref: code_repository/scripts/Preprocessing/Proteomics_preprocessing.py :: map_labels :: `return label_dict.get(short, fallback)`]
- Filtering removes `Unknown` and any labels in `exclude_labels`. [ref: code_repository/scripts/Preprocessing/Proteomics_preprocessing.py :: prepare_proteomics_data :: `~...isin(exclude_labels + ["Unknown"])`]
- In orchestration, `exclude_labels=[]` is passed, so no MoA classes are removed there. [ref: code_repository/scripts/Preprocessing/Preprocessing_pipelines.py :: main :: `prepare_proteomics_data(..., exclude_labels=[])`]
- Feature values are normalized to numeric by replacing commas with dots then `pd.to_numeric(errors='coerce')` (non-numeric become NaN). [ref: code_repository/scripts/Preprocessing/Proteomics_preprocessing.py :: preprocess_X :: `replace(',', '.', regex=True)`]

### 2) Chemistry / MIC / growth-curve features
- Compound dictionary is assembled from Excel sheets (`Growth Curves`, `MIC Map`, `MCE data`). [ref: code_repository/scripts/Preprocessing/Chembio_preprocessing.py :: prepare_compound_data_from_excels :: `sheet_name="Growth Curves"`, `"MIC Map"`, `"MCE data"`]
- SMILES preprocessing keeps first fragment only (`split(".")[0]`). [ref: code_repository/scripts/Preprocessing/Chembio_preprocessing.py :: prepare_compound_data_from_excels :: `smiles = ...split(".")[0]`]
- ECFP parameters: radius `2`, bits `2048`. [ref: code_repository/scripts/Preprocessing/Chembio_preprocessing.py :: ChemicalRepresenter.__init__ :: `radius=2, nBits=2048`]
- Invalid SMILES handling in batch feature generation: fill with NaN vector (`np.full(self.nBits, np.nan)`). [ref: code_repository/scripts/Preprocessing/Chembio_preprocessing.py :: calculate_batch :: `except ... fp = np.full(self.nBits, np.nan)`]
- Growth-curve feature selection picks first condition where final OD `< 0.5 * control`; otherwise last condition is used. [ref: code_repository/scripts/Preprocessing/Chembio_preprocessing.py :: generate_growth_curve_matrix :: `row.iloc[final_timepoint_index] < 0.5 * ctrl_curve[...]`]
- Growth-curve values are multiplied by parsed concentration factor from condition text regex `([\d\.]+)\s*x`. [ref: code_repository/scripts/Preprocessing/Chembio_preprocessing.py :: generate_growth_curve_matrix :: `factor = float(match.group(1)) if match else 1.0`]
- MIC vectorization uses halving dilution buckets from `800.0` to `0.001` and fills missing concentrations by nearest rule; final matrix is row-wise min-max scaled (`MinMaxScaler` on transposed matrix). [ref: code_repository/scripts/Preprocessing/Chembio_preprocessing.py :: generate_mic_feature_matrix :: `generate_dilution_series(start=800.0, stop=0.001)`]
- `generate_mic_feature_matrix` has parameter `od_threshold=0.1` but no direct branch on this value is present. [NEEDS CONFIRMATION]. [ref: code_repository/scripts/Preprocessing/Chembio_preprocessing.py :: generate_mic_feature_matrix :: `od_threshold=0.1`]

### 3) Cell painting features
- Cell painting embeddings come from `InfoAlignRepresenter(model_path='infoalign_model/pretrain.pt')`. [ref: code_repository/scripts/Preprocessing/Cellbio_preprocessing.py :: CellPaintRepresenter.__init__ :: `model_path='infoalign_model/pretrain.pt'`]
- Embedding dimensionality is asserted to `300`; failures are filled with NaN vector length `300`. [ref: code_repository/scripts/Preprocessing/Cellbio_preprocessing.py :: calculate_batch :: `np.full(self.embedding_dim, np.nan)`]

### 4) Text augmentation and text embeddings
- Sentence-level and word-level augmentation each generate `n_augmented + 1` columns (`Aug_0`..`Aug_9` when default `n_augmented=9`). [ref: code_repository/scripts/Preprocessing/Text_processing.py :: augment_moa_descriptions :: `n_augmented: int = 9`]
- Augmentation randomness is seeded (`random.seed(seed)`, `np.random.seed(seed)`) with default `seed=42`. [ref: code_repository/scripts/Preprocessing/Text_processing.py :: augment_moa_descriptions :: `seed: int = 42`]
- If insufficient unique shuffles exist, the original text is reused as padding. [ref: code_repository/scripts/Preprocessing/Text_processing.py :: augment_moa_descriptions :: `row_augments += [base_text] * missing`]
- Embeddings are generated for each `(MoA, Augmentation, Model)` record and expected shape is `(768,)`; failures store `Embedding=None`. [ref: code_repository/scripts/Preprocessing/Text_processing.py :: generate_text_embeddings :: `if emb.shape[0] != 768`]
- Default checkpoint map in pipeline: `PubChemDeBERTa`, `BERT`, `BioBERT`. [ref: code_repository/scripts/Preprocessing/Preprocessing_pipelines.py :: main :: `model_checkpoints = {...}`]

### 5) Deterministic external/eclipsed dataset building notebooks
- External dataset notebook writes deterministic outputs for `test_full`, `test_significant`, and reduced train sets under `Data/external_data/*`. [ref: code_repository/notebooks/data_processing/external_proteomics_processing.ipynb :: intro/config cells :: `creates deterministic external datasets`]
- Eclipse dataset notebook writes analogous deterministic outputs under `Data/eclipse_data/*`. [ref: code_repository/notebooks/data_processing/eclipse_data_processing.ipynb :: intro/config cells :: `creates deterministic datasets for eclipse evaluation`]

## Label construction used by baseline modeling
- Pairwise label used for binary training/evaluation is generated in backbone builder as exact candidate-vs-true MoA match (`int(moa == true_moa)`). [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: generate_backbone_df :: `"Label": int(moa == true_moa)`]
- No-text multiclass path label-encodes class strings via `LabelEncoder`. [ref: code_repository/scripts/Data_generation/Codebase/eval_utils.py :: evaluate_model_multiclass_samples :: `le = LabelEncoder().fit(y_labels.values)`]

## Preprocessing artifacts produced
- Main preprocessing pipeline persists timestamped preprocessed modality pickles in `Data/Compound_data/` and embedding pickles in `Embeddings/`. [ref: code_repository/scripts/Preprocessing/Preprocessing_pipelines.py :: save blocks :: `to_pickle`]
- Modeling/evaluation scripts then consume non-timestamped canonical files in `Data/Base/*.pkl` and `Embeddings/250808_*_embeddings.pkl`. [ref: code_repository/scripts/Data_generation/Feature_eval_script.py :: data constants :: `EMB_SEM  = "Embeddings/250808_semanticAugmentation_embeddings.pkl"`]
