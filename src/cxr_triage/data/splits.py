"""Patient-level train, validation, and test splits.

ChestX-ray14 contains follow-up studies. An image-level shuffle puts those
studies on both sides of the split and leaks patient identity into the test
score. The list files shipped with the dataset were not built to prevent that.
Every helper here assigns the patient, then copies that assignment to each of
the patient's images.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from cxr_triage.labels import CHEST14_LABELS, encode_labels

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


def allocate_split_counts(n_patients: int, ratios: Sequence[float]) -> list[int]:
    """Turn ratios into integer counts that sum to ``n_patients``.

    Uses the largest-remainder method. When there are at least as many patients
    as splits and every ratio is positive, an empty split borrows one patient
    from the largest split so a tiny cohort still yields train, val, and test.
    """
    if n_patients < 0:
        raise ValueError("n_patients cannot be negative.")
    if len(ratios) != 3:
        raise ValueError("Expected three ratios: train, val, test.")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Ratios cannot be negative.")
    if n_patients == 0:
        return [0, 0, 0]

    # Integer quotas avoid binary-float surprises such as 0.7 * 8 landing a hair
    # under 5.6 and changing who receives a leftover patient.
    scale = 1_000_000
    weights = [int(round(ratio * scale)) for ratio in ratios]
    weight_sum = sum(weights)
    if weight_sum == 0:
        raise ValueError("Split ratios must not all be zero.")
    if weight_sum != scale:
        weights[0] += scale - weight_sum
    quotas = [n_patients * weight for weight in weights]
    counts = [quota // scale for quota in quotas]
    remainders = [quota % scale for quota in quotas]
    leftover = n_patients - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (remainders[index], weights[index], -index),
        reverse=True,
    )
    for index in range(leftover):
        counts[order[index]] += 1

    if n_patients >= len(ratios) and all(ratio > 0 for ratio in ratios):
        for index, count in enumerate(list(counts)):
            if count == 0:
                donor = max(range(len(counts)), key=lambda item: (counts[item], ratios[item]))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] = 1
    if sum(counts) != n_patients:
        raise RuntimeError(f"Split counts {counts} do not sum to {n_patients}.")
    return counts


def patient_level_split(
    patient_ids: Iterable[str],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
    strata: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Map each patient id to exactly one of ``train``, ``val``, or ``test``.

    Ids are sorted before shuffling so the result depends on ``seed`` and not on
    row order. When ``strata`` is provided, each stratum is split with the same
    ratios and the same rule, which is how the no-finding balance is kept.
    """
    ratios = (train_ratio, val_ratio, test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1. Got {sum(ratios):.6f}.")

    unique_ids = _sort_ids(set(map(str, patient_ids)))
    if strata is None:
        return _split_group(unique_ids, ratios, seed)

    grouped: dict[str, list[str]] = defaultdict(list)
    for patient_id in unique_ids:
        if patient_id not in strata:
            raise KeyError(f"No stratum for patient {patient_id}.")
        grouped[str(strata[patient_id])].append(patient_id)

    assignment: dict[str, str] = {}
    for offset, key in enumerate(sorted(grouped)):
        assignment.update(_split_group(grouped[key], ratios, seed + offset))
    return assignment


def patient_finding_strata(
    frame: pd.DataFrame,
    labels: tuple[str, ...] = CHEST14_LABELS,
) -> dict[str, str]:
    """Label each patient ``positive`` if any study has a finding, else ``negative``."""
    strata: dict[str, str] = {}
    for patient_id, group in frame.groupby("patient_id", sort=False):
        encoded = np.stack([encode_labels(value, labels) for value in group["finding_labels"]])
        strata[str(patient_id)] = "positive" if float(encoded.sum()) > 0 else "negative"
    return strata


def assign_patient_splits(
    frame: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
    strategy: str = "patient",
    labels: tuple[str, ...] = CHEST14_LABELS,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with a ``split`` column and no patient overlap."""
    if strategy not in {"patient", "patient_stratified"}:
        raise ValueError(f"Unknown split strategy {strategy!r}.")
    strata = patient_finding_strata(frame, labels) if strategy == "patient_stratified" else None
    mapping = patient_level_split(
        frame["patient_id"].tolist(),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        strata=strata,
    )
    assigned = frame.copy()
    assigned["split"] = assigned["patient_id"].map(mapping)
    if assigned["split"].isna().any():
        raise RuntimeError("Some patients were not assigned a split.")
    assert_no_patient_leakage(assigned)
    return assigned


def assert_no_patient_leakage(
    frame: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    split_col: str = "split",
) -> None:
    """Raise if any patient id is present in more than one split."""
    seen: dict[str, str] = {}
    for patient_id, split in zip(frame[patient_col].map(str), frame[split_col].map(str), strict=True):
        previous = seen.get(patient_id)
        if previous is None:
            seen[patient_id] = split
        elif previous != split:
            raise RuntimeError(f"Patient {patient_id} appears in both {previous} and {split}.")


def _split_group(patient_ids: Sequence[str], ratios: Sequence[float], seed: int) -> dict[str, str]:
    ordered = list(patient_ids)
    random.Random(seed).shuffle(ordered)
    counts = allocate_split_counts(len(ordered), ratios)
    assignment: dict[str, str] = {}
    cursor = 0
    for name, count in zip(SPLIT_NAMES, counts, strict=True):
        for patient_id in ordered[cursor : cursor + count]:
            assignment[patient_id] = name
        cursor += count
    return assignment


def _sort_ids(patient_ids: Iterable[str]) -> list[str]:
    values = [str(patient_id) for patient_id in patient_ids]
    if values and all(value.isdigit() for value in values):
        return sorted(values, key=int)
    return sorted(values)
