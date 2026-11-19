"""Detached fixed-plan pilot queue. Never overwrites paper results or old attempts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "plan.json").read_text())
    state = dict(started_utc=datetime.now(timezone.utc).isoformat(), state="running", jobs=[])
    protected = {row["path"]: row["sha256"] for row in plan.get("protected", [])}
    for job in plan["jobs"]:
        destination = root / "jobs" / job["id"]
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / "passed.json").exists():
            state["jobs"].append(dict(id=job["id"], state="passed", reused=True))
            continue
        recorded_source = destination / "source"
        if recorded_source.exists():
            raise FileExistsError(f"Existing incomplete attempt: {destination}")
        recorded_source.mkdir()
        # SMB shares may reject symlinks. Keep permanent source copies on
        # the share, and execute an identical copy on the local filesystem.
        snapshot = Path(tempfile.mkdtemp(prefix="mapper-pilot-"))
        for name in ["src", "scripts", "fold_manifests", "requirements", "tests"]:
            shutil.copytree(
                repo / name, recorded_source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
            shutil.copytree(recorded_source / name,snapshot / name)
        for name in ["data", "artifacts"]:
            (snapshot / name).symlink_to(repo / name, target_is_directory=True)
        entry = dict(
            id=job["id"], state="running", started_utc=datetime.now(timezone.utc).isoformat()
        )
        state["jobs"].append(entry)
        write_json(root / "status.json", state)
        (destination / "job.json").write_text(json.dumps(job, indent=2) + "\n")
        print(f"START {len(state['jobs'])}/{len(plan['jobs'])} {job['id']}", flush=True)
        start = time.monotonic()
        env = dict(
            os.environ,
            PYTHONPATH=str(snapshot / "src"),
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
        )
        with (destination / "log.txt").open("w") as log:
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(snapshot / "scripts/run_pilot_job.py"),
                        "--job",
                        str(destination / "job.json"),
                        "--output",
                        str(destination / "outputs"),
                        "--pilot-root",
                        str(root),
                    ],
                    cwd=snapshot,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                entry["returncode"] = process.wait(timeout=job.get("timeout_seconds", 1800))
                entry["state"] = "passed" if entry["returncode"] == 0 else "failed"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                entry.update(state="failed", error="Pilot timeout")
        entry["elapsed_seconds"] = time.monotonic() - start
        entry["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if entry["state"] == "passed":
            write_json(destination / "passed.json", entry)
        write_json(root / "status.json", state)
        print(f"{entry['state'].upper()} {job['id']} {entry['elapsed_seconds']:.1f}s", flush=True)
        if entry["state"] == "failed":
            print((destination / "log.txt").read_text()[-5000:], flush=True)
    changed = []
    for name, digest in protected.items():
        path = repo / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(name)
    state.update(
        state="finished",
        finished_utc=datetime.now(timezone.utc).isoformat(),
        passed=sum(r["state"] == "passed" for r in state["jobs"]),
        failed=sum(r["state"] == "failed" for r in state["jobs"]),
        protected_changes=changed,
    )
    write_json(root / "status.json", state)
    print(
        json.dumps({k: state[k] for k in ["state", "passed", "failed", "protected_changes"]}),
        flush=True,
    )
    raise SystemExit(1 if state["failed"] or changed else 0)


if __name__ == "__main__":
    main()
