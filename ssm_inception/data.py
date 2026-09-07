"""DSADS loading, filtering, subject split, and normalization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import median_filter


DEVELOPMENT_SUBJECTS = (1, 2, 3, 4, 5, 6, 8)
TEST_SUBJECTS = (7,)
EXPECTED_SHAPE = (45, 125)


def _metadata(path: Path) -> Tuple[int, int]:
    activity_match = re.fullmatch(r"a(\d+)", path.parent.parent.name)
    subject_match = re.fullmatch(r"p(\d+)", path.parent.name)
    if activity_match is None or subject_match is None:
        raise ValueError(f"unexpected DSADS path: {path}")
    return int(activity_match.group(1)), int(subject_match.group(1))


def _ordered_files(root: Path):
    manifest = root / "feature1" / "files.txt"
    if manifest.exists():
        paths = [root / line.strip().lstrip("/") for line in manifest.read_text().splitlines()]
    else:
        paths = sorted(root.glob("a[0-9][0-9]/p[1-8]/s[0-9][0-9].txt"))
    if not paths:
        raise FileNotFoundError(
            f"no DSADS files found below {root}; expected a01/p1/s01.txt layout"
        )
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"manifest references missing file: {missing[0]}")
    return paths


def load_dsads(
    root: Union[str, Path],
    median_kernel_size: int = 3,
    median_boundary_mode: str = "mirror",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return windows [N,45,125], zero-based labels, and subject IDs."""
    root = Path(root).expanduser().resolve()
    windows, labels, subjects = [], [], []
    paths = _ordered_files(root)
    label_manifest = root / "feature1" / "labels.txt"
    manifest_labels = (
        np.loadtxt(label_manifest, dtype=np.int64) if label_manifest.exists() else None
    )
    if manifest_labels is not None and len(manifest_labels) != len(paths):
        raise ValueError("files.txt and labels.txt have different lengths")

    for index, path in enumerate(paths):
        activity, subject = _metadata(path)
        if manifest_labels is not None and int(manifest_labels[index]) != activity:
            raise ValueError(f"label/path disagreement at {path}")
        raw = np.loadtxt(path, delimiter=",", dtype=np.float32).T
        if raw.shape != EXPECTED_SHAPE:
            raise ValueError(f"{path}: expected {EXPECTED_SHAPE}, found {raw.shape}")
        filtered = median_filter(
            raw, size=(1, median_kernel_size), mode=median_boundary_mode
        )
        windows.append(filtered.astype(np.float32, copy=False))
        labels.append(activity - 1)
        subjects.append(subject)

    x = np.stack(windows)
    y = np.asarray(labels, dtype=np.int64)
    subject_ids = np.asarray(subjects, dtype=np.int64)
    return x, y, subject_ids


def prepare_subject_independent_data(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    epsilon: float = 1.0e-8,
    development_subjects: Sequence[int] = DEVELOPMENT_SUBJECTS,
    test_subjects: Sequence[int] = TEST_SUBJECTS,
) -> Dict[str, Tuple]:
    """Create the reported development/test split and fit z-score on dev only.

    The mean and standard deviation are element-wise arrays of shape
    [1, channels, time].  Test subject 7 never contributes to these values.
    """
    development_subjects = tuple(int(subject) for subject in development_subjects)
    test_subjects = tuple(int(subject) for subject in test_subjects)
    dev_mask = np.isin(subjects, development_subjects)
    test_mask = np.isin(subjects, test_subjects)
    if np.any(dev_mask & test_mask):
        raise RuntimeError("subject overlap detected")
    if set(np.unique(subjects[dev_mask])) != set(development_subjects):
        raise RuntimeError("one or more development subjects are missing")
    if set(np.unique(subjects[test_mask])) != set(test_subjects):
        raise RuntimeError("one or more test subjects are missing")

    dev_x = x[dev_mask]
    mean = dev_x.mean(axis=0, keepdims=True)
    std = dev_x.std(axis=0, keepdims=True)
    return {
        "development": (
            ((dev_x - mean) / (std + epsilon)).astype(np.float32),
            y[dev_mask],
            subjects[dev_mask],
        ),
        "test": (
            ((x[test_mask] - mean) / (std + epsilon)).astype(np.float32),
            y[test_mask],
            subjects[test_mask],
        ),
        "normalization": (mean.astype(np.float32), std.astype(np.float32)),
    }
