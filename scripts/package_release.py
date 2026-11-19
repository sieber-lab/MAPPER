"""Build local, checksum-verified code/data and fitted-artifact ZIP packages.

Nothing is uploaded or committed. Only the documented source directories,
manifest-listed inputs and optional manifest-listed fitted artifacts enter
the packages; private archives, Git metadata and local results are excluded.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mapper.paths import artifact_root

FILES = [
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "Makefile",
    "pyproject.toml",
    ".gitignore",
]
DIRECTORIES = [
    "src",
    "scripts",
    "notebooks",
    "requirements",
    "tests",
    "docs",
    "fold_manifests",
    "licenses",
]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(include_artifacts=False):
    records = {}
    for name in FILES:
        records[name] = dict(path=ROOT / name)
    for name in DIRECTORIES:
        for path in sorted((ROOT / name).rglob("*")):
            if path.is_file() and not any(
                x in path.parts for x in ["__pycache__", ".ipynb_checkpoints"]
            ):
                if path.suffix not in [".pyc", ".pyo"]:
                    if path.is_symlink():
                        raise ValueError("Resolve source symlinks before packaging: " + str(path))
                    records[str(path.relative_to(ROOT))] = dict(path=path)
    for entry in json.loads((ROOT / "data/manifest.json").read_text())["files"]:
        records[entry["path"]] = dict(path=ROOT / entry["path"], expected=entry)
    for name in ["data/manifest.json", "data/artifacts.json", "data/paper_index.json"]:
        records.setdefault(name, dict(path=ROOT / name))
    artifacts = {}
    if include_artifacts:
        for entry in json.loads((ROOT / "data/artifacts.json").read_text())["files"]:
            if entry["path"].startswith("artifacts/"):
                source = artifact_root() / Path(entry["path"]).relative_to("artifacts")
                artifacts[entry["path"]] = dict(path=source, expected=entry)
    return records, artifacts


def build_zip(path, records, manifest_name, extra_manifest=None):
    rows = []
    with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=4, allowZip64=True) as archive:
        for number, (name, item) in enumerate(sorted(records.items()), 1):
            source = item["path"]
            if not source.is_file():
                raise FileNotFoundError(source)
            stat = source.stat()
            expected = item.get("expected")
            if expected and stat.st_size != expected["bytes"]:
                raise ValueError("Unexpected size: " + name)
            info = ZipInfo("MAPPER/" + name, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = (0o100755 if source.suffix == ".sh" else 0o100644) << 16
            digest = hashlib.sha256()
            with source.open("rb") as stream, archive.open(info, "w", force_zip64=True) as target:
                for block in iter(lambda: stream.read(4 * 1024**2), b""):
                    digest.update(block)
                    target.write(block)
            actual = digest.hexdigest()
            if expected and actual != expected["sha256"]:
                raise ValueError("Changed canonical input: " + name)
            if (
                source.stat().st_size != stat.st_size
                or source.stat().st_mtime_ns != stat.st_mtime_ns
            ):
                raise RuntimeError("Input changed during packaging: " + name)
            rows.append(dict(path=name, bytes=stat.st_size, sha256=actual))
            if number == 1 or number % 100 == 0 or number == len(records):
                print(f"{path.name}: {number}/{len(records)} files", flush=True)
        info = ZipInfo("MAPPER/" + manifest_name, date_time=(2000, 1, 1, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(
            info, json.dumps(dict(files=rows, **(extra_manifest or {})), indent=2) + "\n"
        )
    return dict(
        filename=path.name,
        bytes=path.stat().st_size,
        sha256=sha(path),
        members=len(rows) + 1,
        input_bytes=sum(r["bytes"] for r in rows),
    )


def zenodo_release(output, source, artifacts):
    """Keep project code/data on GitHub and archive only missing runtime files."""
    github_by_hash = {}
    for name, entry in sorted(source.items()):
        if name.startswith("data/"):
            checksum = entry.get("expected", {}).get("sha256") or sha(entry["path"])
            github_by_hash.setdefault(checksum, name)
    payload, reused = {}, []
    for name, entry in sorted(artifacts.items()):
        identity = entry["expected"]
        github_path = github_by_hash.get(identity["sha256"])
        if github_path:
            github_source = source[github_path]["path"]
            if (
                github_source.stat().st_size != identity["bytes"]
                or sha(github_source) != identity["sha256"]
            ):
                raise ValueError("Changed GitHub reuse input: " + github_path)
            reused.append(dict(identity, github_path=github_path))
        else:
            payload[name] = entry
    forbidden = {".py", ".pyc", ".ipynb", ".sh", ".toml", ".cff"}
    if any(Path(name).suffix in forbidden for name in payload):
        raise ValueError("Project code must not enter the Zenodo-only payload")
    commit, clean = None, False
    if (ROOT / ".git").exists():
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        clean = not subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
        ).strip()
    output.mkdir(parents=True)
    archive = build_zip(
        output / "MAPPER_fitted_artifacts.zip",
        payload,
        "ARTIFACTS_MANIFEST.json",
        extra_manifest=dict(reuse_from_github=reused, code_included=False),
    )
    record = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope="Zenodo companion files only; project code and curated inputs remain in GitHub",
        repository_url="https://github.com/sieber-lab/MAPPER",
        github_commit=commit if clean else None,
        github_base_commit=commit,
        github_checkout_clean=clean,
        matching_github_release_required=True,
        github_compatibility=[
            dict(path=name, sha256=sha(ROOT / name))
            for name in [
                "data/manifest.json",
                "data/artifacts.json",
                "scripts/install_artifacts.py",
            ]
        ],
        packages=[archive],
        payload_files=len(payload),
        reused_github_files=reused,
        code_included=False,
        publication_uploads=0,
    )
    (output / "release_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    readme = """MAPPER: companion model/runtime files
====================================

Project code, figure notebooks, processed inputs and documentation:
https://github.com/sieber-lab/MAPPER

This deposit supplies the fitted model weights, fitted preprocessing objects,
calibration records, prediction outputs and model-specific runtime inputs
needed to reproduce and use the MAPPER models. It does not contain a second
copy of the project source code, notebooks or general analysis/figure inputs.
Four byte-identical metadata/result files are reused from the GitHub checkout
instead of being duplicated here; the installer restores their required paths.

Installation
------------
Use the matching MAPPER GitHub release and run from its repository root:

  python3 scripts/install_artifacts.py --archive /path/to/MAPPER_fitted_artifacts.zip
  python3 scripts/verify_data.py --artifacts

The installer uses only Python's standard library, verifies every checksum
and never overwrites an existing file with different contents. Then follow
the repository README for figure generation, model prediction and fitting.
The separately supplied environments distinguish plotting from inference.

Compatibility and contents
--------------------------
release_manifest.json records archive hashes and the required GitHub-file
hashes. The ZIP's ARTIFACTS_MANIFEST.json records all payload identities and
the files restored from GitHub. The matching GitHub release/commit should
also be linked in the Zenodo record.

  artifacts/reference/       Full/proteomics reference predictor bundles
  artifacts/external/        873-protein log2FC and Proteomics + ECFP models
  artifacts/external_ecfp/   ECFP-only comparison models
  artifacts/external_novel/ Additional unseen-mechanism application records
  artifacts/adaptation/     Held-antibiotic-out Yu adaptation models/results

These are study model artifacts, not the original DIA-NN/Perseus programs or
third-party pretrained encoder weights. Raw instrument data are not duplicated
here. The GitHub README lists the PRIDE or publication source separately for
the reference library, Figure 5 holdouts, Eclipse, corramycin and external studies.
Software licensing and third-party notices are supplied in the GitHub project;
the software MIT licence does not relicense external data or artwork.
"""
    (output / "README.txt").write_text(readme)
    names = ["MAPPER_fitted_artifacts.zip", "release_manifest.json", "README.txt"]
    checksums = {archive["filename"]: archive["sha256"]}
    (output / "SHA256SUMS.txt").write_text(
        "".join(
            (checksums[name] if name in checksums else sha(output / name)) + "  " + name + "\n"
            for name in names
        )
    )
    print(json.dumps(record, indent=2), flush=True)
    return record


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True, help="New local release directory")
    p.add_argument("--with-artifacts", action="store_true")
    p.add_argument(
        "--zenodo-only",
        action="store_true",
        help="Archive runtime artifacts only, without project code or GitHub inputs",
    )
    p.add_argument(
        "--extract-to", type=Path, help="New directory for independent extraction checks"
    )
    args = p.parse_args()
    if args.output.exists() or (args.extract_to is not None and args.extract_to.exists()):
        p.error("Output and extraction directories must not already exist")
    if args.zenodo_only and args.extract_to is not None:
        p.error(
            "Install a Zenodo-only archive with scripts/install_artifacts.py in the matching checkout"
        )
    source, artifacts = collect(args.with_artifacts or args.zenodo_only)
    if args.zenodo_only:
        zenodo_release(args.output, source, artifacts)
        return
    args.output.mkdir(parents=True)
    packages = [
        build_zip(
            args.output / "MAPPER_code_and_processed_data.zip", source, "PACKAGE_MANIFEST.json"
        )
    ]
    if artifacts:
        packages.append(
            build_zip(
                args.output / "MAPPER_fitted_artifacts.zip", artifacts, "ARTIFACTS_MANIFEST.json"
            )
        )
    record = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        packages=packages,
        private_files_excluded=True,
        publication_uploads=0,
    )
    if args.extract_to is not None:
        args.extract_to.mkdir(parents=True)
        for entry in packages:
            with ZipFile(args.output / entry["filename"]) as archive:
                archive.extractall(args.extract_to)
        record["extracted_copy"] = str(args.extract_to / "MAPPER")
    (args.output / "release_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
