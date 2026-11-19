"""Prepare an original operator, batch or preprocessing fit from recorded splits.

Example: prepare_study_job.py --study operators --design whole --job DS_to_LE
         --output results/prepared/operator_DS_to_LE
Then use run_primary_job.py with the emitted matrices, fields and plan files.
No split is generated, and no model is fitted by this preparation command.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from mapper.primary_jobs import prepare_job
from mapper.preprocessing import control_alignment, median_alignment, missing_values, protein_groups

SPLITS = ROOT / "data/study_inputs/splits"
DESIGNS = dict(
    whole="operator_whole",
    count_matched="operator_count_matched",
    structure_matched="operator_structure_matched",
    class_excluded="operator_class_excluded",
    class_mixed="operator_class_mixed",
    class_same_operator="operator_class_same_operator",
)
PREPROCESSING = [
    *missing_values.ARMS,
    "median_alignment",
    "single_protein_groups",
    "control_relative",
    "log2fc_4401",
    "log2fc_873",
    "nominal",
    "bh005",
]


def read_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_manifest(study, design):
    name = DESIGNS[design] if study == "operators" else "measurement_batch"
    path = SPLITS / f"{name}.json"
    return path, json.loads(path.read_text())


def prepare(study, *, design="whole", job=None, preprocessing="zero", iteration=0, fold=0):
    base_path = ROOT / "data/inhouse/paper_training.npz"
    data = read_arrays(base_path)
    fields = json.loads((ROOT / "data/configurations/reference_fields.json").read_text())
    sources = [base_path]
    if study in ["operators", "batches"] or job is not None:
        split_path, manifest = load_manifest(study, design)
        spec = manifest["splits"][job]
        classes = manifest["classes"]
        sources.append(split_path)
    else:
        split_path = ROOT / "data/study_inputs/preprocessing/folds.json"
        split = json.loads(split_path.read_text())["folds"][iteration][fold]
        spec = dict(
            train_compounds=split["train"],
            test_compounds=split["test"],
            train_samples=data["sample_ids"][np.isin(data["compounds"], split["train"])].tolist(),
            test_samples=data["sample_ids"][np.isin(data["compounds"], split["test"])].tolist(),
        )
        classes = sorted(set(data["labels"]))
        sources.append(split_path)
    if study == "preprocessing":
        folder = ROOT / "data/study_inputs/preprocessing"
        raw_path = folder / "raw_inputs.npz"
        raw = read_arrays(raw_path)
        sources.append(raw_path)
        training = np.isin(raw["sample_ids"], spec["train_samples"])
        if not training.any():
            raise ValueError("No training profiles")
        if preprocessing in missing_values.ARMS:
            params = missing_values.fit_parameters(raw["log2_groups"][training])
            groups = missing_values.transform(
                raw["log2_groups"],
                params,
                preprocessing,
                missing_values.SEED + 100 * iteration + fold,
            )
            values = missing_values.expand_genes(groups, raw["gene_to_group"])
        elif preprocessing == "median_alignment":
            ref = median_alignment.fit_reference(raw["log2_groups"][training])
            groups = median_alignment.transform(raw["log2_groups"], ref)[0]
            values = missing_values.expand_genes(groups, raw["gene_to_group"])
        elif preprocessing == "single_protein_groups":
            groups = np.nan_to_num(raw["log2_groups"], nan=0.0)
            values = protein_groups.exclude_multigene_groups(
                missing_values.expand_genes(groups, raw["gene_to_group"]), raw["gene_to_group"]
            )
        elif preprocessing == "control_relative":
            controls_path = folder / "control_inputs.npz"
            controls = read_arrays(controls_path)
            sources.append(controls_path)
            np.testing.assert_array_equal(controls["sample_ids"], raw["sample_ids"])
            indices = controls["sample_group_indices"]
            means = controls["control_means"]
            ref = control_alignment.fit_reference(means, indices[training])
            groups = control_alignment.transform(raw["log2_groups"], means[indices], ref)[0]
            values = missing_values.expand_genes(groups, raw["gene_to_group"])
        else:
            file = folder / (
                "fold_changes.npz" if preprocessing.startswith("log2fc_") else "significance.npz"
            )
            cached = read_arrays(file)
            sources.append(file)
            order = pd.Index(cached["sample_ids"]).get_indexer(raw["sample_ids"])
            if (order < 0).any():
                raise ValueError("Unmatched preprocessing profile")
            values = cached[preprocessing][order]
        order = pd.Index(raw["sample_ids"]).get_indexer(data["sample_ids"])
        values = values[order]
        data["proteomics"] = values
        data["multimodal"] = np.column_stack([values, data["multimodal"][:, 4401:]])
        if preprocessing == "log2fc_873":
            panel = read_arrays(folder / "fold_changes.npz")
            data["genes"] = panel["genes"][panel["panel_mask"]]
    selected = set(spec["train_samples"]) | set(spec["test_samples"])
    # The original operator scripts used lexical profile order. Other studies
    # retained the natural measurement-matrix order. Keep that distinction.
    ids = (
        sorted(selected)
        if study == "operators"
        else [s for s in data["sample_ids"] if s in selected]
    )
    indices = pd.Index(data["sample_ids"]).get_indexer(ids)
    if (indices < 0).any() or len(indices) != len(selected):
        raise ValueError("Split refers to unavailable profiles")
    for key in ["sample_ids", "compounds", "labels", "multimodal", "proteomics"]:
        data[key] = data[key][indices]
    for role in ["train", "test"]:
        actual = set(data["sample_ids"][np.isin(data["compounds"], spec[f"{role}_compounds"])])
        if actual != set(spec[f"{role}_samples"]):
            raise ValueError("Compound selection does not preserve the original profile split")
    plan = dict(
        classes=classes,
        seeds=list(range(42, 47)) if study == "operators" else [42],
        jobs=dict(
            study=dict(
                train=spec["train_compounds"], predict=spec["test_compounds"], candidates=classes
            )
        ),
    )
    for arm in fields["feature_blocks"]:
        prepare_job(data, plan, "study", fields, arm)
    provenance = dict(
        study=study,
        design=design,
        job=job,
        preprocessing=preprocessing,
        iteration=iteration,
        fold=fold,
        source_files=[
            dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sources
        ],
    )
    return data, plan, fields, provenance


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--study", choices=["operators", "batches", "preprocessing"], required=True)
    p.add_argument("--design", choices=DESIGNS, default="whole")
    p.add_argument(
        "--job", help="Recorded operator split or measurement batch, e.g. DS_to_LE or C1"
    )
    p.add_argument("--preprocessing", choices=PREPROCESSING, default="zero")
    p.add_argument("--iteration", type=int, choices=range(5), default=0)
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--list", action="store_true")
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.list:
        if args.study == "preprocessing":
            print(json.dumps(dict(arms=PREPROCESSING, iterations=5, folds=5), indent=2))
        else:
            _, manifest = load_manifest(args.study, args.design)
            print(json.dumps(list(manifest["splits"]), indent=2))
        return
    if args.study in ["operators", "batches"] and args.job is None:
        p.error("Choose a recorded --job; --list shows the choices")
    if not args.inspect and (args.output is None or args.output.exists()):
        p.error("Supply a new --output directory, or use --inspect")
    data, plan, fields, provenance = prepare(
        args.study,
        design=args.design,
        job=args.job,
        preprocessing=args.preprocessing,
        iteration=args.iteration,
        fold=args.fold,
    )
    summary = dict(
        **provenance,
        profiles=len(data["sample_ids"]),
        train_compounds=len(plan["jobs"]["study"]["train"]),
        test_compounds=len(plan["jobs"]["study"]["predict"]),
        numeric_widths={
            a: data[fields["feature_blocks"][a][0]].shape[1] for a in fields["feature_blocks"]
        },
    )
    if not args.inspect:
        args.output.mkdir(parents=True)
        np.savez_compressed(args.output / "matrices.npz", **data)
        for filename, value in [
            ("plan.json", plan),
            ("fields.json", fields),
            ("provenance.json", summary),
        ]:
            (args.output / filename).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
