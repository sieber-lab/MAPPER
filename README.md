# MAPPER

MAPPER ranks nine antibiotic mode-of-action classes and flags uncertain assignments.
It accompanies [*Decoding antibiotic modes of action from multimodal cellular responses*](https://doi.org/10.64898/2026.03.31.715570).

For the manuscript release, run `git clone --branch v0.1.0 https://github.com/sieber-lab/MAPPER.git`,
then `cd MAPPER`. Run the commands below from that directory.

| I want to… | Start here | Download |
|---|---|---|
| Find the training data or a particular test set | [Where are the data?](#where-are-the-data) | GitHub for processed measurements; source repositories for raw MS |
| Predict a new sample, or try the paper's corramycin example | [Run a prediction](#run-a-prediction) | GitHub + [fitted models on Zenodo](https://doi.org/10.5281/zenodo.24369937) |
| Reproduce a figure or recalculate the statistics | [Reproduce the paper](#reproduce-the-paper) | GitHub only; no GPU or retraining |
| Train models from the processed measurements | [Training and recorded study fits](docs/reproduction.md) | GitHub; external and nested workflows also use the recorded inputs listed in the guide |

## Where are the data?

“In-house” describes **where a sample was measured**, not whether it was used for training. Eclipse and corramycin were measured in-house, but both belong to the Figure 6 test set.

| Samples | Processed files in this repository | Raw measurements or published source |
|---|---|---|
| **Training library:** 51 antibiotics, 200 profiles | [Training matrix and individual modalities](data/inhouse/); [compound identities, classes and sample IDs](data/metadata/compound_metadata.csv) | [PRIDE PXD076144](https://www.ebi.ac.uk/pride/archive/projects/PXD076144) |
| **Five in-house holdout antibiotics:** 20 profiles (Fig. 5) | [Holdout matrix and individual modalities](data/holdout/) | [PRIDE PXD076144](https://www.ebi.ac.uk/pride/archive/projects/PXD076144) |
| **In-house Eclipse:** three library antibiotics remeasured on another instrument (Fig. 6) | [Aligned profiles](data/protein_mapping/prepared/eclipse_full.npz); [Perseus exports and intensity fingerprints](data/processed_sources/eclipse_data/) | [PRIDE PXD076144](https://www.ebi.ac.uk/pride/archive/projects/PXD076144) |
| **In-house corramycin:** four treated profiles (Fig. 6) | [Aligned profiles and matched control means](data/protein_mapping/prepared/inhouse_corramycin_full.npz); [example input CSVs](docs/examples/corramycin/) | Separate PRIDE submission; accession pending |
| **Published external studies**, including Yu et al., Subanovic et al., HSI#6 and TAT-RasGAP (Fig. 6) | [One aligned matrix per dataset](data/protein_mapping/prepared/); [Yu adaptation inputs](data/study_inputs/adaptation/) | [Dataset index: publications, accessions and supplementary tables](data/figure_inputs/external_final/datasets.csv) |

The reference models use **4,401-protein intensity profiles**. For Figure 6, `*_full.npz` means the external measurements have been aligned to those 4,401 protein identifiers; it does **not** mean this was the final prediction input. The external predictor uses **873 proteins** and log2 fold changes relative to matched controls, without significance selection. The [panel definition](data/protein_mapping/prepared/panel.npz) and [exact 873-protein matrices for the 43 known-class tests](data/study_inputs/external/matrices.npz) are included. The two additional unseen-mechanism tests have separate model inputs in `artifacts/external_novel/prepared/` after installing the Zenodo archive.

For upstream in-house quantification, see the [processed Perseus tables](data/processed_sources/MS_reference_files/). For the numbers behind a particular panel, use the [figure and table index](data/paper_index.json). Raw instrument files are not needed to reproduce the figures or run predictions.

## Run a prediction

From the repository root, install the [matching Zenodo model ZIP](https://doi.org/10.5281/zenodo.24369937) with the supplied installer, **without unzipping it manually**:

```bash
python3 scripts/install_artifacts.py --archive /path/to/MAPPER_fitted_artifacts.zip
python3 scripts/verify_data.py --artifacts
```

Use a separate Python environment for the fitted models (the saved preprocessing objects need the pinned versions). For CPU inference:

```bash
python3.10 -m venv .venv-models
.venv-models/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv-models/bin/python -m pip install -r requirements/inference.txt
```

**Try a real sample.** These commands take the supplied corramycin treated and control CSVs all the way to a prediction:

```bash
.venv-models/bin/python scripts/prepare_external.py \
  --treated docs/examples/corramycin/treated.csv --controls docs/examples/corramycin/controls.csv \
  --compound-id JH70 --output results/example/corramycin.npz
.venv-models/bin/python scripts/predict_external.py \
  --profiles results/example/corramycin.npz --arm proteomics --context full51 \
  --output results/example/corramycin_predictions.csv
```

Expected: **DNA replication**, maximum matching score **0.330**, cutoff **0.145**, **not flagged**. The output CSV also gives all nine class scores. A matching score is not a calibrated probability; an uncertainty flag is not a novel-mechanism call.

**For your own compound,** replace the two CSVs and the compound ID. Each row is a mapped protein and each column after the identifier is a biological profile. Supply **log2 abundances**, not raw intensities or precomputed ratios. The preparation step subtracts the mean of the matched untreated or solvent controls. The example uses the stored mean of four DMSO controls, not four invented replicates. See the [example and input guide](docs/examples/corramycin/README.md) for protein IDs, missing values, choosing an antibiotic-excluded model, and optional chemical fingerprints.

To reproduce the **five Figure 5 holdouts** instead, use the original intensity-based models:

```bash
.venv-models/bin/python scripts/predict.py --arm multimodal --output results/holdout_full
.venv-models/bin/python scripts/predict.py --arm proteomics --output results/holdout_proteomics
```

## Reproduce the paper

From the repository root, install the plotting environment separately. Linux needs Cairo and librsvg (`libcairo2`, `librsvg2-2`); use Arial or Arimo (`fonts-croscore`) for the paper typography.

```bash
python3.10 -m venv .venv-figures
.venv-figures/bin/python -m pip install -r requirements/figures.txt
make PYTHON=.venv-figures/bin/python figure-inputs
.venv-figures/bin/python scripts/build_figures.py main06  # Figure 6 only
make PYTHON=.venv-figures/bin/python figures             # all main, Extended Data and SI figures
make PYTHON=.venv-figures/bin/python reports             # numerical results from saved predictions
```

Alternatively, run the cells of any of the [23 numerical figure notebooks](notebooks/) with the plotting environment as the kernel. Figure 1 is supplied artwork. Each figure is exported to `figures/<figure_id>/` as PDF, SVG and PNG with a provenance record. Neither the notebooks nor `make reports` refit the models.

For fitting from scratch, the [reproduction guide](docs/reproduction.md) describes environments, experiment entry points, recorded folds and seeds, and a worked training command. Original MAPPER code is [MIT licensed](LICENSE); see [third-party notices](THIRD_PARTY_NOTICES.md) for other materials.
