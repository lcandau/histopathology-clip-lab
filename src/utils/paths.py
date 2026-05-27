"""Path helpers — Colab vs local execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_colab() -> bool:
    return "google.colab" in sys.modules


def repo_root() -> Path:
    if is_colab():
        return Path("/content/histopathology-clip-lab")
    return Path(__file__).resolve().parents[2]


def drive_root() -> Path:
    if is_colab():
        return Path("/content/drive/MyDrive/clip_histopathology")
    return repo_root() / "local_drive" / "clip_histopathology"


def run_dir(experiment: str, version: str) -> Path:
    out = drive_root() / experiment / version
    out.mkdir(parents=True, exist_ok=True)
    return out


def results_dir(kind: str) -> Path:
    """kind: 'metrics', 'plots', 'confusion_matrices', or 'splits'."""
    if kind == "splits":
        out = repo_root() / "data" / "splits"
    else:
        out = repo_root() / "results" / kind
    out.mkdir(parents=True, exist_ok=True)
    return out
