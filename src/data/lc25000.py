"""LC25000 class constants, file discovery and stratified splits."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ClassInfo:
    index: int
    id: str
    name: str
    subdir: str


CLASS_INFO: tuple[ClassInfo, ...] = (
    ClassInfo(0, "lung_n", "benign lung tissue", "lung_image_sets/lung_n"),
    ClassInfo(1, "lung_aca", "lung adenocarcinoma", "lung_image_sets/lung_aca"),
    ClassInfo(2, "lung_scc", "lung squamous cell carcinoma", "lung_image_sets/lung_scc"),
    ClassInfo(3, "colon_n", "benign colon tissue", "colon_image_sets/colon_n"),
    ClassInfo(4, "colon_aca", "colon adenocarcinoma", "colon_image_sets/colon_aca"),
)

NUM_CLASSES = len(CLASS_INFO)
CLASS_IDS: tuple[str, ...] = tuple(c.id for c in CLASS_INFO)
CLASS_NAMES: tuple[str, ...] = tuple(c.name for c in CLASS_INFO)
INDEX_TO_ID = {c.index: c.id for c in CLASS_INFO}
INDEX_TO_NAME = {c.index: c.name for c in CLASS_INFO}
ID_TO_INDEX = {c.id: c.index for c in CLASS_INFO}


_DEDUPE_EXCLUSIONS_PATH = Path(__file__).with_name("lc25000_dedupe_exclusions.json")


def _load_dedupe_exclusions() -> frozenset[str]:
    """Set of basenames to skip because they are byte-duplicates of another file
    in the dataset. LC25000 contains ~1,280 such duplicates which would otherwise
    cause train/test leakage under a random split."""
    if not _DEDUPE_EXCLUSIONS_PATH.is_file():
        return frozenset()
    payload = json.loads(_DEDUPE_EXCLUSIONS_PATH.read_text())
    return frozenset(payload.get("excluded_basenames", []))


def discover_records(
    dataset_root: str | os.PathLike,
    *,
    apply_dedupe: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (paths, class_indices) for every JPEG under dataset_root.

    By default (apply_dedupe=True) skips ~1,280 byte-duplicate files listed in
    lc25000_dedupe_exclusions.json — these would otherwise leak from train to
    test under a random split because they are byte-identical copies of files
    kept in the dataset.

    Pass apply_dedupe=False to reproduce the legacy 25,000-file behaviour (used
    by pre-deduplication runs).
    """
    base = Path(dataset_root) / "lung_colon_image_set"
    excluded = _load_dedupe_exclusions() if apply_dedupe else frozenset()
    paths: list[str] = []
    indices: list[int] = []
    for info in CLASS_INFO:
        class_dir = base / info.subdir
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg")):
                continue
            if fname in excluded:
                continue
            paths.append(str(class_dir / fname))
            indices.append(info.index)
    return np.array(paths), np.array(indices, dtype=np.int32)


def stratified_split(
    paths: np.ndarray,
    indices: np.ndarray,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Stratified 3-way split. Returns a dict of train/val/test paths and indices."""
    assert 0 < val_fraction < 1 and 0 < test_fraction < 1
    assert val_fraction + test_fraction < 1

    train_paths, temp_paths, train_idx, temp_idx = train_test_split(
        paths, indices,
        test_size=val_fraction + test_fraction,
        random_state=seed, stratify=indices,
    )
    relative_test_fraction = test_fraction / (val_fraction + test_fraction)
    val_paths, test_paths, val_idx, test_idx = train_test_split(
        temp_paths, temp_idx,
        test_size=relative_test_fraction,
        random_state=seed, stratify=temp_idx,
    )
    return {
        "train_paths": train_paths, "train_indices": train_idx,
        "val_paths": val_paths,     "val_indices": val_idx,
        "test_paths": test_paths,   "test_indices": test_idx,
    }


def save_split(split: dict[str, np.ndarray], path: str | os.PathLike) -> None:
    payload = {
        "train_paths": [str(p) for p in split["train_paths"]],
        "train_indices": [int(i) for i in split["train_indices"]],
        "val_paths": [str(p) for p in split["val_paths"]],
        "val_indices": [int(i) for i in split["val_indices"]],
        "test_paths": [str(p) for p in split["test_paths"]],
        "test_indices": [int(i) for i in split["test_indices"]],
        "class_info": [{"index": c.index, "id": c.id, "name": c.name} for c in CLASS_INFO],
    }
    Path(path).write_text(json.dumps(payload))


def load_split(path: str | os.PathLike) -> dict[str, np.ndarray]:
    payload = json.loads(Path(path).read_text())
    return {
        "train_paths": np.array(payload["train_paths"]),
        "train_indices": np.array(payload["train_indices"], dtype=np.int32),
        "val_paths": np.array(payload["val_paths"]),
        "val_indices": np.array(payload["val_indices"], dtype=np.int32),
        "test_paths": np.array(payload["test_paths"]),
        "test_indices": np.array(payload["test_indices"], dtype=np.int32),
    }
