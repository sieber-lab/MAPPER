"""Fit one seed of the final reference model using the fixed input plan."""

import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--arm", choices=["multimodal", "proteomics"], default="multimodal")
    parser.add_argument("--seed", type=int, choices=range(42, 47), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if not args.inspect and args.output is None:
        parser.error("A new --output directory is required for model fitting")
    sys.argv = [
        "run_primary_job.py",
        "--inputs",
        str(ROOT / "data/inhouse/paper_training.npz"),
        "--plan",
        str(ROOT / "data/configurations/reference.json"),
        "--fields",
        str(ROOT / "data/configurations/reference_fields.json"),
        "--job",
        "full51",
        "--arm",
        args.arm,
        "--seed",
        str(args.seed),
    ]
    sys.argv += ["--inspect"] if args.inspect else ["--output", str(args.output)]
    runpy.run_path(str(ROOT / "scripts/run_primary_job.py"), run_name="__main__")


if __name__ == "__main__":
    main()
