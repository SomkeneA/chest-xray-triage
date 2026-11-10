"""Patient-level splits must not leak, and they must be deterministic."""

from __future__ import annotations

import pandas as pd
import pytest

from cxr_triage.data.splits import (
    allocate_split_counts,
    assert_no_patient_leakage,
    assign_patient_splits,
    patient_level_split,
)


def _frame(n_patients: int = 40, images: int = 2) -> pd.DataFrame:
    rows = []
    for patient in range(1, n_patients + 1):
        label = "No Finding" if patient % 2 == 0 else "Effusion"
        for image in range(images):
            rows.append(
                {
                    "image_index": f"{patient:08d}_{image:03d}.png",
                    "finding_labels": label,
                    "patient_id": str(patient),
                }
            )
    return pd.DataFrame(rows)


def test_allocate_counts_sum_and_cover_small_cohorts():
    assert allocate_split_counts(100, (0.7, 0.1, 0.2)) == [70, 10, 20]
    assert allocate_split_counts(10, (0.7, 0.1, 0.2)) == [7, 1, 2]
    assert allocate_split_counts(8, (0.7, 0.1, 0.2)) == [6, 1, 1]
    assert allocate_split_counts(3, (0.7, 0.1, 0.2)) == [1, 1, 1]
    assert sum(allocate_split_counts(2, (0.7, 0.1, 0.2))) == 2


@pytest.mark.parametrize("strategy", ["patient", "patient_stratified"])
def test_no_patient_appears_in_two_splits(strategy: str):
    frame = _frame()
    assigned = assign_patient_splits(frame, seed=7, strategy=strategy)
    groups = {
        name: set(assigned.loc[assigned["split"] == name, "patient_id"]) for name in ("train", "val", "test")
    }
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    assert set.union(*groups.values()) == set(frame["patient_id"])
    assert (assigned.groupby("patient_id")["split"].nunique() == 1).all()
    assert set(assigned["split"]) == {"train", "val", "test"}


def test_split_depends_on_seed_not_row_order():
    frame = _frame()
    shuffled = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    first = assign_patient_splits(frame, seed=3, strategy="patient").set_index("image_index")
    second = assign_patient_splits(shuffled, seed=3, strategy="patient").set_index("image_index")
    assert first["split"].sort_index().equals(second["split"].sort_index())
    other = assign_patient_splits(frame, seed=4, strategy="patient").set_index("image_index")
    assert not first["split"].sort_index().equals(other["split"].sort_index())


def test_stratified_split_keeps_positive_and_negative_patients_in_each_split():
    frame = _frame(n_patients=50, images=1)
    assigned = assign_patient_splits(frame, seed=1, strategy="patient_stratified")
    for split in ("train", "val", "test"):
        ids = assigned.loc[assigned["split"] == split, "patient_id"].astype(int)
        assert (ids % 2 == 0).any()
        assert (ids % 2 == 1).any()


def test_leakage_guard_names_the_patient():
    frame = pd.DataFrame({"patient_id": ["1", "1"], "split": ["train", "test"]})
    with pytest.raises(RuntimeError, match="Patient 1"):
        assert_no_patient_leakage(frame)


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        patient_level_split(["1", "2", "3"], train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
