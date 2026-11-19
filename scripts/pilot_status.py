"""Show progress without attaching to or interrupting the pilot queue."""

import argparse, json
from pathlib import Path

parser = argparse.ArgumentParser(__doc__)
parser.add_argument(
    "--root",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "results/reproducibility_pilots",
)
args = parser.parse_args()
plan = args.root / "plan.json"
status = args.root / "status.json"
if not status.exists():
    runtime = args.root / "runtime_status.txt"
    print(runtime.read_text().strip() if runtime.exists() else "Not started")
else:
    data = json.loads(status.read_text())
    total = len(json.loads(plan.read_text())["jobs"])
    print(
        f"{sum(r['state']=='passed' for r in data['jobs'])}/{total} passed; "
        f"{sum(r['state']=='failed' for r in data['jobs'])} failed; queue {data['state']}"
    )
    for row in data["jobs"]:
        print(
            f"{row['state']:8s} {row['id']}"
            + (f"  {row['elapsed_seconds']:.0f}s" if "elapsed_seconds" in row else "")
        )
