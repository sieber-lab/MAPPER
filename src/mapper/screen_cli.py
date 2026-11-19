"""Small explicit command interface; listing configurations never starts a fit."""

import argparse
from pathlib import Path
from .screens import configurations, fit_configuration


def main(study, leave_one_out=False):
    parser = argparse.ArgumentParser(description=f"Fixed {study} configurations")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--configuration")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--pilot", action="store_true", help="Execute one original fold/seed, not the full study"
    )
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--compound", default="JH0", help="Held-out identity for a LOO pilot")
    args = parser.parse_args()
    choices = {item["id"]: item for item in configurations(study)}
    if args.list or not args.configuration:
        print("\n".join(choices))
        return
    if args.configuration not in choices:
        parser.error("Unknown configuration; see --list")
    if args.output is None:
        parser.error("A new --output directory is required to fit models")
    fit_configuration(
        choices[args.configuration],
        args.output,
        leave_one_out=leave_one_out,
        pilot=(args.iteration, args.fold, args.compound) if args.pilot else None,
    )
