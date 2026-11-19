"""Configurable roots for the separately stored large research artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def repository_root():
    return Path(__file__).resolve().parents[2]


def artifact_root():
    return Path(os.environ.get("MAPPER_ARTIFACTS", repository_root() / "artifacts")).resolve()
