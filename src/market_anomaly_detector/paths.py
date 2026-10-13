"""Repo-relative default paths, each overridable via environment variable."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

DATA_PATH = Path(
    os.environ.get("MAD_DATA_PATH", REPO_ROOT / "data" / "raw" / "ews_weekly_2000_2021.csv")
)

ARTIFACT_DIR = Path(os.environ.get("MAD_ARTIFACT_DIR", REPO_ROOT / "artifacts"))
