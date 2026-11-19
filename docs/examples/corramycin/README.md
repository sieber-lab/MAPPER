# Predict an antibiotic from treated and control proteomes

The [two prediction commands in the main README](../../../README.md#run-a-prediction) use these real Figure 6 measurements. Corramycin (JH70) was profiled in-house, separately from the 51 training antibiotics. No model fitting is needed for this example.

### What do the input tables contain?

- [treated.csv](treated.csv): four corramycin-treated profiles, one per column.
- [controls.csv](controls.csv): one column containing the per-protein mean log2 abundance of the four DMSO controls, using the available measurements for each protein. This is the control summary stored in the study data, not a single biological replicate.
- Both tables use the same 873 named protein features. They were exported from `treated` and `control_pools` in the [aligned study matrix](../../../data/protein_mapping/prepared/inhouse_corramycin_full.npz), in the order given by `selected_genes` in the [panel file](../../../data/protein_mapping/prepared/panel.npz).

### Substitute your own measurements

1. Prepare two CSVs for **one antibiotic and one experimental condition**. Put the reference protein identifier in the first column and one biological profile in each remaining column. The identifiers used by MAPPER are *E. coli* K-12 gene symbols for the measured proteins; map your identifiers using the [reference annotation](../../../data/protein_mapping/audit/reference_gene_identifiers.csv) before preparing the CSVs. Do not supply unresolved or duplicate protein rows.
2. Enter **log2 protein abundances**. Leave missing values blank; do not substitute zeros for missing intensities. You can supply all mapped proteins: the script selects the panel by name. In the controls CSV, provide either the individual matched control profiles or their per-protein mean on the log2 scale. The script subtracts the control mean from each treated profile, then fills unavailable fold changes with zero. It does not take ratios of log-transformed values.
3. Check overlap **before interpreting the prediction**. The generated NPZ includes a boolean `observed` array; `observed.sum(axis=1)` gives the number of measured treated/control pairs per profile out of 873. In particular, a result with no matching proteins is not interpretable.
4. Choose the training context. Use `full51` for an antibiotic absent from the reference library. For a paper antibiotic, look up its JH identity in the [compound metadata](../../../data/metadata/compound_metadata.csv) and the available identity-excluded contexts in the [study plan](../../../data/study_inputs/external/plan.json). For example, use `--compound-id JH29 --context omit_JH29` for ciprofloxacin (pass the ID to preparation and the context to prediction). The software does not resolve chemical synonyms. If no omitted context is available, do not report a `full51` result as a prediction on an unseen antibiotic.

### What should I see?

For the supplied example, the proteomics-only model ranks **DNA replication** first (score approximately 0.330, cutoff 0.145, `uncertain=False`). Compare with the `proteomics` / `inhouse_corramycin` row of the [paper prediction table](../../../data/figure_inputs/external_final/current_case_predictions.csv). The CSV output contains one row for the antibiotic after averaging replicate predictions, not four independent calls.

### What if I also have the structure?

Use `--arm ecfp_proteomics` for the combined model. The two intensity CSVs alone do not supply chemical features: the prepared NPZ must also contain an `ecfp` array (one row per profile, 2,048 bits per row, repeating the same fingerprint for replicates of the same antibiotic). The study used a radius-2 Morgan bit fingerprint, without chirality, on the first dot-separated fragment of the recorded SMILES; there was no additional neutralization or tautomer standardization. The [study matrices](../../../data/study_inputs/external/matrices.npz) contain the fingerprints used for our external tests. Use the cutoff returned for the selected arm; it is not the proteomics-only cutoff.
