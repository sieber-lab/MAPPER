"""Run one fresh fitting pilot with original settings and isolated outputs."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mapper.pilot_checks import checkpoint_roundtrip, report_pairs


def command(script, arguments):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)], check=True
    )


def compare_scores(current, reference, output):
    reference = ROOT / reference
    if not reference.exists():
        raise FileNotFoundError(reference)
    old = pd.read_csv(reference, float_precision="round_trip")
    if "view" in old:
        old = old.loc[old["view"].eq("standard")]
    keys = [
        key
        for key in ["SampleID", "CompoundID", "CandidateMoA", "Aug_Index"]
        if key in current and key in old
    ]
    if not keys:
        raise ValueError("No shared prediction identities")
    a = current.groupby(keys).Score.mean().sort_index()
    b = old.groupby(keys).Score.mean().sort_index()
    pd.testing.assert_index_equal(a.index, b.index)
    difference = float(np.max(np.abs(a - b)))
    record = dict(
        rows=len(a),
        maximum_absolute_difference=difference,
        reference=str(reference.relative_to(ROOT)),
        within_numerical_tolerance=bool(np.allclose(a, b, atol=2e-6, rtol=1e-5)),
    )
    (output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    if not record["within_numerical_tolerance"]:
        raise AssertionError(
            f"Fresh predictions differ from original by {difference}; inspect comparison.json"
        )
    return record


def screen(job, out, pilot_root):
    from mapper import evaluation

    factory = evaluation.get_model

    class CheckedModel:
        def __init__(self, model, kind):
            self.model = model
            self.kind = kind

        def fit(self, x, y):
            self.model.fit(x, y)
            checkpoint_roundtrip(self.model, x, out / "checkpoint", self.kind)
            return self

        def __getattr__(self, name):
            return getattr(self.model, name)

    def checked(name, *args, **kwargs):
        return CheckedModel(factory(name, *args, **kwargs), name)

    evaluation.get_model = checked
    from mapper.screens import configurations, fit_configuration

    config = next(c for c in configurations(job["study"]) if c["id"] == job["configuration"])
    prediction = out / "predictions"
    result = fit_configuration(
        config,
        prediction,
        leave_one_out=job.get("loo", False),
        pilot=(0, 0, job.get("compound", "JH0")),
    )
    outputs = result["per_fold_outputs"]
    if len(outputs) != 1:
        raise ValueError("Pilot must execute exactly one fit")
    data = outputs[0].reset_index().rename(columns={"ProbaPos": "Score"})
    if "Text_MoA" in data:
        data = data.rename(columns={"Text_MoA": "CandidateMoA"})
    labels = pd.read_pickle(ROOT / "data/inhouse/y_labels.pkl")
    data["True_MoA"] = data.SampleID.map(labels)
    report_pairs(data, out)
    if job.get("reference"):
        compare_scores(data, job["reference"], out)


def text_control(job, out, pilot_root):
    args = [
        "--model",
        job["model"],
        "--arm",
        job["arm"],
        "--iteration",
        0,
        "--fold",
        0,
        "--output",
        out / "fit",
    ]
    if job.get("fixed_duration"):
        args += ["--fixed-duration"]
    command("run_text_control.py", args)
    frame = pd.read_csv(out / "fit/predictions.csv.gz", float_precision="round_trip")
    report_pairs(frame, out)
    with (out / "fit/model.pkl").open("rb") as stream:
        bundle = pickle.load(stream)
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_text_control import load_inputs
    from mapper.analysis.text_controls import assemble

    x, _, _ = assemble(load_inputs(), job["arm"], 0, 0, training=False)
    checkpoint_roundtrip(
        bundle["model"], bundle["standard_scaler"].transform(x[:128]), out / "reload", job["model"]
    )
    if job.get("reference"):
        compare_scores(frame, job["reference"], out)


def make_pairs(scores, profile_indices, text_indices, data, fields):
    profiles = data[fields["profile_ids"]][profile_indices]
    compounds = data[fields["compound_ids"]][profile_indices]
    labels = data[fields["labels"]][profile_indices]
    candidates = data[fields["text_classes"]][text_indices]
    return pd.DataFrame(
        dict(
            SampleID=np.repeat(profiles, len(candidates)),
            CompoundID=np.repeat(compounds, len(candidates)),
            True_MoA=np.repeat(labels, len(candidates)),
            CandidateMoA=np.tile(candidates, len(profiles)),
            Score=scores[:, 1],
        )
    )


def primary(job, out, pilot_root):
    base = Path(job["base"])
    if not base.is_absolute():
        base = ROOT / base
    command(
        "run_primary_job.py",
        [
            "--inputs",
            base / "matrices.npz",
            "--plan",
            base / "plan.json",
            "--fields",
            base / "fields.json",
            "--job",
            job["job"],
            "--arm",
            job["arm"],
            "--seed",
            42,
            "--output",
            out / "fit",
        ],
    )
    with np.load(base / "matrices.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    fields = json.loads((base / "fields.json").read_text())
    with np.load(out / "fit/predictions.npz", allow_pickle=False) as z:
        frame = make_pairs(
            z["probabilities"], z["profile_indices"], z["text_indices"], data, fields
        )
    if len(frame):
        frame.to_csv(out / "pairwise.csv.gz", index=False)
        report_pairs(frame, out)
        if job.get("reference"):
            compare_scores(frame, job["reference"], out)
    if job.get("application"):
        from mapper.external import predict_checkpoint

        application = job["application"]
        with np.load(ROOT / application["file"], allow_pickle=False) as z:
            values = (
                np.column_stack([z[k] for k in application["feature_blocks"]])
                if application.get("feature_blocks")
                else z[application["features"]]
            )
            profiles = z[application["profiles"]]
            compounds = z[application["compounds"]]
        if application.get("case_keys"):
            mask = np.isin(compounds, application["case_keys"])
            values, profiles, compounds = values[mask], profiles[mask], compounds[mask]
        scores = predict_checkpoint(out / "fit", values, data[fields["text"]])
        candidates = data[fields["text_classes"]]
        frame = pd.DataFrame(
            dict(
                SampleID=np.repeat(profiles, len(candidates)),
                CompoundID=np.repeat(compounds, len(candidates)),
                CandidateMoA=np.tile(candidates, len(profiles)),
                Score=scores.ravel(),
            )
        )
        frame.to_csv(out / "application.csv.gz", index=False)
        if application.get("reference"):
            compare_scores(frame, application["reference"], out)


def baseline(job, out, pilot_root):
    from mapper.analysis.baselines import pearson_similarity, tanimoto, class_max, grouped_means
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    with np.load(ROOT / "data/analysis/text_controls/matrices.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    split = json.loads((ROOT / "data/analysis/text_controls/folds.json").read_text())["folds"][0][0]
    labels = dict(zip(data["compounds"], data["labels"]))
    classes = pd.Index(data["classes"])
    train = np.isin(data["compounds"], split["train"])
    test = np.isin(data["compounds"], split["test"])
    yt = classes.get_indexer(data["labels"][train])
    train_ids = split["train"]
    test_ids = split["test"]
    comp_y = classes.get_indexer([labels[c] for c in train_ids])
    x = data["numeric"][:, :4401]
    ecfp = data["numeric"][:, 4401:6449]
    arm = job["arm"]
    model = None
    if arm == "proteomic_matching":
        a = grouped_means(x[train], data["compounds"][train], train_ids)
        b = grouped_means(x[test], data["compounds"][test], test_ids)
        scaler = StandardScaler().fit(a)
        scores = (
            class_max(pearson_similarity(scaler.transform(b), scaler.transform(a)), comp_y) + 1
        ) / 2
    elif arm == "chemical_similarity":
        a = grouped_means(ecfp[train], data["compounds"][train], train_ids)
        b = grouped_means(ecfp[test], data["compounds"][test], test_ids)
        scores = class_max(tanimoto(b, a), comp_y)
    elif arm == "class_prior":
        scores = np.tile(
            np.bincount(comp_y, minlength=len(classes)) / len(comp_y), (len(test_ids), 1)
        )
    else:
        model = (
            RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=1,
                random_state=42,
                n_jobs=4,
                class_weight=None,
            )
            if arm == "proteomics_rf"
            else make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    multi_class="multinomial",
                    max_iter=10000,
                    random_state=42,
                    class_weight=None,
                ),
            )
        )
        model.fit(x[train], yt)
        checkpoint_roundtrip(model, x[test], out / "checkpoint", arm)
        scores = grouped_means(model.predict_proba(x[test]), data["compounds"][test], test_ids)
    frame = pd.DataFrame(
        dict(
            CompoundID=np.repeat(test_ids, len(classes)),
            True_MoA=np.repeat([labels[c] for c in test_ids], len(classes)),
            SampleID=np.repeat(test_ids, len(classes)),
            CandidateMoA=np.tile(classes, len(test_ids)),
            Score=scores.ravel(),
        )
    )
    report_pairs(frame, out)
    if job.get("reference"):
        compare_scores(frame, job["reference"], out)


def context_fit(job, out, pilot_root, *, adaptation=False):
    """A full three-fold cutoff-calibration context, using one training seed."""
    from mapper.external import predict_checkpoint
    from mapper.uncertainty import calibrate_score_cutoff, flag_uncertain

    base = ROOT / job["base"]
    plan = json.loads((base / "plan.json").read_text())
    fields = json.loads((base / "fields.json").read_text())
    context = next(c for c in plan["contexts"] if c["context"] == job["context"])
    with np.load(base / "matrices.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    tables = []
    comparisons = []
    for fold in context["calibration"]:
        sub = out / f"calibration_{fold['fold']}"
        sub.mkdir()
        primary(dict(base=job["base"], job=fold["job"], arm=job["arm"]), sub, pilot_root)
        table = pd.read_csv(sub / "compound_scores.csv", float_precision="round_trip")
        if adaptation:
            # The paper's adaptation cutoff uses held-out Yu antibiotics.
            domains = dict(zip(data[fields["compound_ids"]], data["domains"]))
            yu_groups = set(
                data[fields["compound_ids"]][np.char.lower(data["domains"].astype(str)) == "yu"]
            )
            table = table.loc[table.CompoundID.isin(yu_groups)]
        tables.append(table)
        if not adaptation:
            original = (
                ROOT
                / "artifacts"
                / ("external_ecfp" if job["arm"] == "ecfp_only" else "external")
                / "fits"
                / job["arm"]
                / fold["job"]
                / "seed_42/predictions.npz"
            )
            with np.load(original, allow_pickle=False) as z:
                expected = z["pair_scores"]
                indices = z["profile_indices"]
            with np.load(sub / "fit/predictions.npz", allow_pickle=False) as z:
                observed = z["probabilities"][:, 1].reshape(expected.shape)
                np.testing.assert_array_equal(indices, z["profile_indices"])
            np.testing.assert_allclose(observed, expected, atol=2e-6, rtol=1e-5)
            comparisons.append(
                dict(
                    fold=fold["fold"],
                    maximum_absolute_difference=float(np.max(abs(observed - expected))),
                )
            )
    calibration = pd.concat(tables, ignore_index=True)
    if calibration.duplicated(["CompoundID", "CandidateMoA"]).any():
        raise ValueError("Repeated calibration identity")
    wide = calibration.pivot(
        index=["CompoundID", "True_MoA"], columns="CandidateMoA", values="Score"
    )
    correct = wide.idxmax(axis=1).to_numpy() == wide.index.get_level_values("True_MoA").to_numpy()
    cutoff = calibrate_score_cutoff(wide.max(axis=1).to_numpy(), correct)
    calibration.to_csv(out / "calibration_scores.csv", index=False)
    main = out / "main"
    main.mkdir()
    primary(dict(base=job["base"], job=context["main_job"], arm=job["arm"]), main, pilot_root)
    if adaptation:
        mask = (data["groups"] == context["omitted"]) & (
            np.char.lower(data["domains"].astype(str)) == "yu"
        )
        assert mask.any() and context["omitted"] not in context["develop"]
        values = data["proteomics"][mask]
        case_keys = data["case_keys"][mask]
    else:
        mask = np.isin(data["external_case_keys"], context["external_cases"])
        blocks = dict(
            proteomics=["external_proteomics"],
            ecfp_only=["external_ecfp"],
            ecfp_proteomics=["external_proteomics", "external_ecfp"],
        )
        values = np.column_stack([data[key][mask] for key in blocks[job["arm"]]])
        case_keys = data["external_case_keys"][mask]
    scores = predict_checkpoint(main / "fit", values, data[fields["text"]])
    classes = sorted(set(data[fields["text_classes"]]))
    resolved = np.column_stack(
        [scores[:, data[fields["text_classes"]] == c].mean(axis=1) for c in classes]
    )
    table = pd.DataFrame(resolved, columns=classes)
    table["case"] = case_keys
    averaged = table.groupby("case")[classes].mean()
    averaged["prediction"] = averaged.idxmax(axis=1)
    averaged["maximum_score"] = averaged[classes].max(axis=1)
    averaged["uncertain"] = flag_uncertain(averaged.maximum_score.to_numpy(), cutoff)
    averaged.to_csv(out / "application.csv")
    record = dict(
        calibration_folds=3,
        seed=42,
        correct_calibration_cases=int(correct.sum()),
        calibration_cases=len(wide),
        score_cutoff=cutoff,
        pilot_only_cutoff=True,
        calibration_source="Yu" if adaptation else "inhouse",
        application_cases=len(averaged),
        original_prediction_comparisons=comparisons,
        test_identity_excluded_from_all_training=bool(
            not set(context.get("omitted", []) if not adaptation else [context["omitted"]])
            & set(context["develop"])
        ),
    )
    (out / "calibration.json").write_text(json.dumps(record, indent=2) + "\n")


def input_preparation(job, out, pilot_root):
    """Roundtrip an actual mapped treatment/control example through the public CLI."""
    source = ROOT / "data/protein_mapping/prepared/inhouse_corramycin_full.npz"
    with np.load(source, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    np.testing.assert_allclose(
        data["control_mean"],
        np.broadcast_to(data["control_mean"][0], data["control_mean"].shape),
        equal_nan=True,
    )
    pd.DataFrame(data["treated"].T, index=data["genes"], columns=data["sample_ids"]).to_csv(
        out / "treated.csv"
    )
    pd.DataFrame(
        data["control_mean"][0], index=data["genes"], columns=["matched_control_mean"]
    ).to_csv(out / "controls.csv")
    command(
        "prepare_external.py",
        [
            "--treated",
            out / "treated.csv",
            "--controls",
            out / "controls.csv",
            "--compound-id",
            "corramycin",
            "--output",
            out / "profiles.npz",
        ],
    )
    with np.load(ROOT / "data/protein_mapping/prepared/panel.npz", allow_pickle=False) as z:
        proteins = z["selected_genes"]
    columns = pd.Index(data["genes"]).get_indexer(proteins)
    assert (columns >= 0).all()
    with np.load(out / "profiles.npz", allow_pickle=False) as z:
        expected = np.where(data["valid"][:, columns], data["fold_changes"][:, columns], 0.0)
        np.testing.assert_allclose(z["features"], expected, atol=2e-12, rtol=0)
        np.testing.assert_array_equal(z["observed"], data["valid"][:, columns])
    (out / "preparation.json").write_text(
        json.dumps(
            dict(
                actual_treatment_control_example=True,
                proteins=len(proteins),
                profiles=len(data["sample_ids"]),
                fold_changes_match=True,
            ),
            indent=2,
        )
        + "\n"
    )


def figure3(job, out, pilot_root):
    from sklearn.preprocessing import StandardScaler
    from mapper.models.factory import get_model
    from mapper.screens import configurations, prepare_configuration
    from mapper.external import pair_inputs

    if job["arm"] == "text":
        config = next(
            c
            for c in configurations("features")
            if c["model"] == job["model"]
            and c["stored_feature_order"] == ["Text_Embedding", "Proteomics"]
        )
        table, x, y = prepare_configuration(config)
        split = json.loads((ROOT / "fold_manifests/figure3d_text_compound_folds.json").read_text())[
            "folds"
        ][0][0]
        train = table.CompoundID.isin(split["train"]).to_numpy()
        test = table.CompoundID.isin(split["test"]).to_numpy()
        train_x, train_y, test_x = x[train], y[train], x[test]
        test_rows = table.loc[test].rename(columns={"Text_MoA": "CandidateMoA"})
        n_classes = 2
    else:
        x = pd.read_pickle(ROOT / "data/inhouse/X_proteomics.pkl")
        labels = pd.read_pickle(ROOT / "data/inhouse/y_labels.pkl")
        compounds = x.index.str.split("_").str[0]
        split = json.loads(
            (ROOT / "fold_manifests/figure3d_no_text_compound_folds.json").read_text()
        )["folds"][0][0]
        train = np.isin(compounds, split["train"])
        test = np.isin(compounds, split["test"])
        classes = np.unique(labels)
        repeat = 90 if job["arm"] == "repeated" else 1
        train_x = np.repeat(x.to_numpy()[train], repeat, axis=0)
        train_y = np.repeat(pd.Index(classes).get_indexer(labels.to_numpy()[train]), repeat)
        test_x = x.to_numpy()[test]
        n_classes = 9
        test_rows = pd.DataFrame(
            dict(
                SampleID=np.repeat(x.index[test], 9),
                CompoundID=np.repeat(compounds[test], 9),
                True_MoA=np.repeat(labels.to_numpy()[test], 9),
                CandidateMoA=np.tile(classes, test.sum()),
            )
        )
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    model = get_model(job["model"], input_dim=train_x.shape[1], n_classes=n_classes, seed=42)
    if job["model"] == "lightgbm":
        model.set_params(n_jobs=4)
    model.fit(train_x, train_y)
    checkpoint_roundtrip(model, scaler.transform(test_x[:128]), out / "checkpoint", job["model"])
    scores = model.predict_proba(scaler.transform(test_x))
    frame = test_rows[["SampleID", "CompoundID", "True_MoA", "CandidateMoA"]].copy()
    frame["Score"] = scores[:, 1] if n_classes == 2 else scores.ravel()
    report_pairs(frame, out)
    if job.get("reference"):
        compare_scores(frame, job["reference"], out)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pilot-root", type=Path, required=True)
    args = p.parse_args()
    job = json.loads(args.job.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    import torch

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU pilot environment lost CUDA; no silent CPU fallback")
    start = time.monotonic()
    print(
        json.dumps(dict(job=job, torch=torch.__version__, gpu=torch.cuda.get_device_name(0))),
        flush=True,
    )
    handlers = dict(
        screen=screen,
        text_control=text_control,
        primary=primary,
        baseline=baseline,
        figure3=figure3,
        external_context=context_fit,
        adaptation_context=lambda j, o, p: context_fit(j, o, p, adaptation=True),
        input_preparation=input_preparation,
    )
    if job["kind"] == "command":
        command(job["script"], job.get("arguments", []))
    else:
        handlers[job["kind"]](job, args.output, args.pilot_root)
    record = dict(
        id=job["id"],
        status="passed",
        elapsed_seconds=time.monotonic() - start,
        unchanged_model_hyperparameters=True,
        seed=42,
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ["src", "scripts"]
            for path in (ROOT / folder).rglob("*.py")
        },
    )
    (args.output / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print("PILOT PASSED", job["id"], flush=True)


if __name__ == "__main__":
    main()
