# Reproduction levels

The workflows separate three operations with different requirements:

1. Figure generation reads the saved analysis tables and supplied artwork.
   It uses a CPU and does not fit models or choose thresholds.
2. Numerical analysis recomputes statistics and summaries from the saved
   compound/profile predictions, using the fixed partitions and aggregation.
3. Model fitting uses the processed measurement matrices and cached text
   embeddings. Frozen model bundles permit prediction without refitting.

The figure environment and trusted-model environment are separate because
the fitted scikit-learn objects were serialized with older library versions.
Install the matching requirements rather than updating a fitted bundle's
preprocessing library in place. Pickle, joblib and PyTorch files are executable
serialization formats: load only trusted study artifacts.

The reference predictor uses the original 4,401-protein intensity input and
all five numerical modalities. The external application instead uses an
873-protein panel and log2 treated/control fold changes, without significance
selection, with optional ECFP. A different input representation requires a
matching fitted model and cutoff; changing only the test matrix is not valid.

The final reference checkpoints are the matched refits used in Figure 5,
not the older deployment pickle. Their numeric block order is proteomics,
ECFP, InfoAlign, MIC and growth curves; the text block comes first. Prediction
applies both saved preprocessing stages: the outer StandardScaler and the
TabM QuantileTransformer. The reference matrix has 7,547 columns with text,
or 5,169 for proteomics plus text. No scaler is fitted on prediction inputs.

`data/inhouse/paper_training.npz` and `data/holdout/paper_profiles.npz` preserve
the exact processed arrays and profile order. Frozen checkpoints, calibration
records, external predictions and the antibiotic-exclusion plans are under
`artifacts/`. Set `MAPPER_ARTIFACTS` when storing that directory elsewhere.

External log2FC inputs require matched untreated or solvent-control profiles.
The preparation CLI accepts already mapped log2 protein abundances, subtracts
the observed control mean for each protein, selects the fixed panel and fills
unavailable fold changes with zero. It does not divide log-transformed values.
Published linear treatment/control ratios must instead be log2-transformed.
Protein identifier matching and ambiguous-group handling are implemented in
`analysis/identifiers.py` and `analysis/representations.py`; the exact study
mapping tables, reference annotations and selected panel are in
`data/protein_mapping/` and the supplementary mapping workbook.

The Zenodo companion contains the additional fitted-model/runtime artifacts,
not another copy of the GitHub source tree. Run `scripts/install_artifacts.py`
against its ZIP in the matching GitHub checkout before using saved predictors.
The installer restores byte-identical metadata/result files from the checkout
where possible and checks the full artifact inventory. It requires only the
Python standard library. Code/curated-input version compatibility is recorded
in the deposit manifest; a local uncommitted checkout is not a published
GitHub release.

The fixed training procedure stops after three epochs without a new minimum
mean training loss, up to 100 epochs, and uses the final-epoch weights. This
matches the fitted models used in the paper. Cleanup does not replace this
procedure with best-epoch checkpoint restoration.

InfoAlign is the public modality name. Historical serialized keys can still
contain `CellPaint`; the compatibility mapping never sorts or changes blocks.
Feature order comes from each model bundle's stored metadata.

Figure 1 is supplied artwork. The other 24 figure notebooks assemble every
panel, annotation and artwork component in code. Generated SVG/PDF/PNG files
require no subsequent manual editing. Numerical table and artwork hashes are
checked before plotting. Frozen provenance labels may retain their original
source filenames; these are not required server paths.

## Validation and aggregation

Feature comparisons retain 25 held-out folds: five five-fold partitions,
with one model seed per fold. Compound metrics are calculated for each of the
five complete repetitions and then averaged. They are not calculated after
averaging predictions across repetitions. The two MLP implementations are
separate model families.

The uncertainty comparison uses five outer compound folds and nine outer
class-omission tests. All methods receive the same primary predictions.
The retained meta-model comparator uses all 15 original features, quadratic
expansion and balanced logistic regression with C=1. Its architecture is fixed;
the nested procedure fits and calibrates it without seeing an outer test
compound. Low maximum matching score is the default deployed uncertainty rule.

External antibiotic identities present in the reference library are excluded
from all training, preprocessing and cutoff-calibration stages for their test.
The 43 known-class treatments and the two additional unseen-MoA examples are
reported separately. The Yu adaptation excludes all measurements of the test
antibiotic across both sources. Its saved internal calibration is not a new
guarantee of external false-alarm rates.

The exact numerical methods are importable under `src/mapper/analysis/`.
`make reports` checks the saved scientific endpoints against recomputed values.
The conditional bootstrap summaries hold the fitted models and partitions
fixed. Full GPU refitting is separate from figure and numerical-report checks;
CPU/GPU or library changes can give small floating-point differences.

Bounded training pilots run one recorded fold/seed per comparison, retaining
the full epoch limits, stopping rules, feature order and training-only scaling.
They check fresh prediction, aggregation and checkpoint reloads. Where a saved
fold prediction is available, the pilot also tests numerical agreement against
it. External calibration pilots fit all three internal calibration folds for
one context and one seed; their new cutoffs are audit outputs, not replacements
for the five-seed paper cutoffs. A bounded pilot is not a rerun of all CV folds,
the full nested primary-model graph, pretrained encoder training or raw mass
spectrometry processing. Each job retains its exact executed source snapshot,
input hashes, log and result under `results/reproducibility_pilots/`.

The meta-model replay check allows small numerical differences between CPU
optimizer/linear-algebra builds (raw margins: relative 1e-5, absolute 1e-7;
calibrated probabilities: relative 1e-5, absolute 1e-6). Uncertainty flags,
score ordering and benchmark AUROC/average precision must still match exactly.
This portability check does not change the fitted method or any paper result.

## Preparing a recorded study fit

Run commands from the repository root. Start with the separate model environment
described in the [README](../README.md#run-a-prediction); for training install
`requirements/training.txt` into that environment and use the appropriate PyTorch
build for your CPU or GPU. In the examples below, `python` refers to that
environment's interpreter. Give each new fit its own output directory.

| To rerun | Start with |
|---|---|
| Model/feature screen (192 configurations) or encoder screen (nine) | `python scripts/feature_screen.py --list` or `python scripts/encoder_screen.py --list`; use `--help` to select a configuration and output. |
| Reference model or leave-one-antibiotic-out fit | `scripts/train_reference.py` (one declared seed) or `scripts/leave_one_out.py`. |
| Class-text controls or nested uncertainty comparator | `scripts/run_text_control.py`, or `scripts/refit_uncertainty.py` to refit the original meta-model on the saved nested primary outputs. |
| Operator, batch, preprocessing or external/adaptation fits | Study plan and `scripts/run_primary_job.py`, as described below. |

The screen commands accept `--pilot --iteration 0 --fold 0` for one recorded fold;
`leave_one_out.py --pilot --compound JH0` selects one antibiotic and one seed.
These are training checks, not full CV results. Generating new InfoAlign
descriptors is optional; the study inputs already contain those features.
For new descriptors use the [authors' package](https://github.com/liugangcode/infoalign-package)
in its own environment.

The processed reference arrays and complete split manifests are sufficient
to reconstruct operator, measurement-batch and preprocessing jobs without
access to the original working directories. For example:

```bash
python scripts/prepare_study_job.py --study operators --design whole --list
python scripts/prepare_study_job.py --study operators --design whole --job DS_to_LE --output results/operator_inputs
python scripts/run_primary_job.py --inputs results/operator_inputs/matrices.npz --plan results/operator_inputs/plan.json --fields results/operator_inputs/fields.json --job study --arm proteomics --seed 42 --output results/operator_fit
```

Other operator designs are `count_matched`, `structure_matched`,
`class_excluded`, `class_mixed` and `class_same_operator`; `--list` prints their
recorded split names. Use `--study batches --job C1` for a measurement-batch
holdout. For preprocessing use `--study preprocessing --preprocessing log2fc_873
--iteration 0 --fold 0`; iterations and folds range from 0 to 4. Available
choices are printed by `--study preprocessing --list`. `--inspect` checks
inputs without writing or fitting. Original operator fits use seeds 42–46;
preprocessing and batch comparisons use seed 42. Repeat only the recorded
jobs/seeds needed for the intended analysis; full studies require multiple fits.

The complete nested uncertainty job graph is in
`data/configurations/uncertainty_plan.json`, with the reference matrices and
`data/configurations/reference_fields.json`. External and adaptation job plans
are in their respective `artifacts/*/prepared/` directories; the corresponding
field maps are `data/configurations/external_fields.json` and
`data/study_inputs/adaptation/fields.json`. These use the same
`run_primary_job.py` interface. No test antibiotic may enter its predictor fit
or any of its cutoff-calibration fits.

Historical fit manifests are retained byte-for-byte. Their source filenames
document provenance, not required locations on a reader's computer. The
separate relocation manifests use the present paths and hashes; old manifests
are never re-signed to make relocated code appear to be the original source.
