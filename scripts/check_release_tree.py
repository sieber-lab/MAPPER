"""Check the Git index or a commit against the expected public release paths.

Run from a maintained checkout after staging changes, or from a clean clone.
This checks packaging, not numerical results. No files are modified.
"""
import argparse
from pathlib import Path
import subprocess

from package_release import collect

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true", help="Check the Git index")
    group.add_argument("--ref", default="HEAD", help="Check this commit (default: HEAD)")
    args = parser.parse_args()
    cmd = ["ls-files", "-z"] if args.staged else ["ls-tree", "-rz", "--name-only", args.ref]
    paths = subprocess.check_output(["git", "-C", str(ROOT), *cmd]).decode().split("\0")
    actual = set(paths) - {""}
    expected = set(collect()[0])
    # Also check a few essential paths explicitly: collect() discovers Python
    # files in the worktree, so an incomplete *clean clone* could hide an omission.
    expected.update({
        "src/mapper/figures/registry.py", "src/mapper/figures/style.py",
        "tests/figures/test_external_coverage.py", "scripts/install_artifacts.py",
    })
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise SystemExit(f"Missing expected files: {missing}\nUnexpected public files: {extra}")
    print(f"Release inventory matches: {len(actual)} Git paths.")


if __name__ == "__main__":
    main()
