"""Fit one explicitly declared primary job; --inspect never starts fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from sklearn.preprocessing import StandardScaler
from mapper.primary_jobs import prepare_job
from mapper.external import restore_checkpoint


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--arm", default="proteomics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    fields = json.loads(args.fields.read_text())
    if args.seed not in plan["seeds"]:
        parser.error("Seed is not part of the fixed plan")
    with np.load(args.inputs, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    job = prepare_job(arrays, plan, args.job, fields, args.arm)
    summary = {key: value for key, value in job.items() if key not in {"train", "test", "target"}}
    summary.update(
        training_pairs=len(job["target"]), test_pairs=len(job["test"]), seed=args.seed, arm=args.arm
    )
    if args.inspect:
        print(json.dumps(summary, default=lambda value: value.tolist(), indent=2))
        return
    if args.output is None or args.output.exists():
        parser.error("Supply a new --output directory; existing results are never overwritten")
    import torch
    from mapper.models.factory import get_model

    torch.set_num_threads(4)
    started = time.monotonic()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(job["train"])
    model = get_model("tabm", input_dim=job["feature_width"], n_classes=2, seed=args.seed)
    model.fit(scaled, job["target"])
    checkpoint = dict(
        input_dim=job["feature_width"],
        seed=args.seed,
        state_dict={
            key: value.detach().cpu().clone() for key, value in model.model.state_dict().items()
        },
    )
    restored = restore_checkpoint(checkpoint, model.scaler)
    check = scaled[: min(90, len(scaled))]
    np.testing.assert_array_equal(model.predict_proba(check), restored.predict_proba(check))
    test_values = scaler.transform(job["test"]) if len(job["test"]) else job["test"]
    scores = restored.predict_proba(test_values) if len(test_values) else np.empty((0, 2))
    args.output.mkdir(parents=True)
    torch.save(checkpoint, args.output / "checkpoint.pt")
    with (args.output / "preprocessors.pkl").open("wb") as stream:
        pickle.dump(dict(standard_scaler=scaler, quantile_transformer=model.scaler), stream)
    np.savez_compressed(
        args.output / "predictions.npz",
        probabilities=scores,
        profile_indices=job["test_profile_indices"],
        text_indices=job["text_indices"],
    )
    record = dict(
        job=args.job,
        arm=args.arm,
        seed=args.seed,
        train_compounds=job["train_compounds"],
        test_compounds=job["test_compounds"],
        training_pairs=len(job["target"]),
        inputs={str(path): sha(path) for path in [args.inputs, args.plan, args.fields]},
        source_sha256={
            str(path.relative_to(ROOT)): sha(path)
            for path in sorted((ROOT / "src/mapper/models").glob("*.py"))
        },
        output_sha256={path.name: sha(path) for path in args.output.iterdir()},
        checkpoint_roundtrip_exact=True,
        final_epoch_weights=True,
        elapsed_seconds=time.monotonic() - started,
    )
    (args.output / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
