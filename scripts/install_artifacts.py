"""Install trusted Zenodo model/runtime files into a matching GitHub checkout.

Uses only the Python standard library. Byte-identical files already supplied
in GitHub are restored from that checkout rather than duplicated in Zenodo.
Existing files with different contents are never overwritten.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "MAPPER/ARTIFACTS_MANIFEST.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(value, prefix=None):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in ["..", "."] for p in path.parts):
        raise ValueError("Unsafe manifest path")
    if "\\" in value or (prefix is not None and path.parts[0] != prefix):
        raise ValueError("Unexpected manifest path")
    return path


def matches(path, record):
    return (
        path.is_file()
        and path.stat().st_size == record["bytes"]
        and digest(path) == record["sha256"]
    )


def write_checked(stream, target, record):
    if target.exists():
        if not matches(target, record):
            raise FileExistsError("Existing file differs; not overwritten: " + str(target))
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".mapper-artifact-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            shutil.copyfileobj(stream, output, length=4 * 1024**2)
        if not matches(temporary, record):
            raise ValueError("Checksum mismatch: " + str(target))
        if target.exists():
            if not matches(target, record):
                raise FileExistsError("Destination changed during installation: " + str(target))
            return False
        os.chmod(temporary, 0o644)
        temporary.replace(target)
        return True
    finally:
        if temporary.exists():
            temporary.unlink()


def install(archive_path, repository=ROOT, destination=None, inspect=False):
    repository = Path(repository).resolve()
    destination = Path(
        destination or os.environ.get("MAPPER_ARTIFACTS", repository / "artifacts")
    ).resolve()
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP entries")
        manifest = json.loads(archive.read(MANIFEST))
        payload = manifest["files"]
        reused = manifest.get("reuse_from_github", [])
        rows = payload + reused
        if len({r["path"] for r in rows}) != len(rows):
            raise ValueError("Repeated manifest path")
        expected_names = {MANIFEST} | {"MAPPER/" + r["path"] for r in payload}
        if set(names) != expected_names:
            raise ValueError("ZIP contains missing or unlisted files")
        for row in payload:
            if archive.getinfo("MAPPER/" + row["path"]).file_size != row["bytes"]:
                raise ValueError("ZIP member size differs from manifest: " + row["path"])
        # The repository's independently supplied manifest is authoritative.
        expected = {
            r["path"]: r
            for r in json.loads((repository / "data/artifacts.json").read_text())["files"]
            if r["path"].startswith("artifacts/")
        }
        if set(expected) != {r["path"] for r in rows}:
            raise ValueError("Archive does not match this checkout artifact inventory")
        for row in rows:
            relative = safe_path(row["path"], "artifacts").relative_to("artifacts")
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError("Destination escapes artifact directory")
            original = expected[row["path"]]
            if any(row[key] != original[key] for key in ["bytes", "sha256"]):
                raise ValueError(
                    "Artifact identity differs from repository manifest: " + row["path"]
                )
            if target.exists() and not matches(target, row):
                raise FileExistsError("Existing file differs; not overwritten: " + str(target))
        for row in reused:
            relative = safe_path(row["github_path"])
            if relative.parts[0] == "artifacts":
                raise ValueError("A reused source must belong to the GitHub checkout")
            source = repository.joinpath(*relative.parts)
            if not source.resolve().is_relative_to(repository) or not matches(source, row):
                raise ValueError("Missing/different GitHub source: " + row["github_path"])
        result = dict(
            archive_files=len(payload),
            reused_github_files=len(reused),
            total_artifacts=len(rows),
            installed=0,
            unchanged=0,
            inspected_only=inspect,
        )
        if inspect:
            return result
        for row in payload:
            target = destination.joinpath(*safe_path(row["path"], "artifacts").parts[1:])
            with archive.open("MAPPER/" + row["path"]) as stream:
                created = write_checked(stream, target, row)
            result["installed" if created else "unchanged"] += 1
        for row in reused:
            target = destination.joinpath(*safe_path(row["path"], "artifacts").parts[1:])
            with (repository / row["github_path"]).open("rb") as stream:
                created = write_checked(stream, target, row)
            result["installed" if created else "unchanged"] += 1
        return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument(
        "--destination", type=Path, help="Defaults to MAPPER_ARTIFACTS or checkout/artifacts"
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Check inventory and GitHub source files without extracting",
    )
    args = p.parse_args()
    print(
        json.dumps(
            install(args.archive, destination=args.destination, inspect=args.inspect), indent=2
        )
    )


if __name__ == "__main__":
    main()
