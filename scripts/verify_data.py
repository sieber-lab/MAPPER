"""Verify canonical input hashes; optionally check the large local artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--artifacts", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    records = manifest["files"]
    if args.artifacts:
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from mapper.paths import artifact_root

        records += [
            row
            for row in json.loads((ROOT / "data/artifacts.json").read_text())["files"]
            if row["path"].startswith("artifacts/")
        ]
    for row in records:
        path = ROOT / row["path"]
        if row["path"].startswith("artifacts/"):
            path = artifact_root() / Path(row["path"]).relative_to("artifacts")
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or digest(path) != row["sha256"]
        ):
            raise ValueError("Missing or changed input: " + row["path"])
    print(f"Verified {len(records)} canonical input files.")


if __name__ == "__main__":
    main()
