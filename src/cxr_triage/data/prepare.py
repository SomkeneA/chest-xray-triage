"""Turn a config file into the frame training and evaluation both use."""

from __future__ import annotations

import pandas as pd

from cxr_triage.config import ExperimentConfig
from cxr_triage.data.metadata import (
    filter_existing_images,
    filter_view,
    limit_per_split,
    load_metadata,
)
from cxr_triage.data.splits import assert_no_patient_leakage, assign_patient_splits
from cxr_triage.labels import CHEST14_LABELS


def prepare_experiment_frame(cfg: ExperimentConfig) -> pd.DataFrame:
    """Load, filter, split, and confirm that no patient crosses a split."""
    frame = load_metadata(cfg.data.csv_path)
    frame = filter_view(frame, cfg.data.view_position)
    frame = assign_patient_splits(
        frame,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.seed,
        strategy=cfg.data.split_strategy,
        labels=CHEST14_LABELS,
    )
    if cfg.data.drop_missing:
        frame = filter_existing_images(frame, cfg.data.image_dir)
    frame = limit_per_split(frame, cfg.data.max_per_split)
    assert_no_patient_leakage(frame)
    missing = [name for name in ("train", "val", "test") if name not in set(frame["split"])]
    if missing:
        counts = frame["split"].value_counts().to_dict()
        raise ValueError(
            f"Splits {missing} have no images after filtering. Counts: {counts}. "
            "Check the CSV, image directory, view filter, and ratios."
        )
    return frame
