"""Load the NIH ChestX-ray14 ``Data_Entry`` table.

The official CSV header contains unquoted commas inside ``OriginalImage[Width,Height]``.
Pandas splits those names, but the columns this pipeline needs stay intact:
``Image Index``, ``Finding Labels``, and ``Patient ID``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_IMAGE_ALIASES = ("Image Index", "image_index", "ImageIndex")
_LABEL_ALIASES = ("Finding Labels", "Finding Label", "finding_labels")
_PATIENT_ALIASES = ("Patient ID", "patient_id", "PatientId")
_VIEW_ALIASES = ("View Position", "view_position", "ViewPosition")


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Return a frame with ``image_index``, ``finding_labels``, and ``patient_id``."""
    # dtype=str keeps zero-padded Patient IDs such as 00000001. Letting pandas
    # infer numbers would turn that id into 1 and would also trip over ages
    # stored as 058Y. The official header has one more comma than each data row
    # because of an unquoted trailing comma; the columns we use still line up.
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    frame.columns = [str(column).strip() for column in frame.columns]
    renamed = frame.rename(
        columns={
            _first_present(frame, _IMAGE_ALIASES, "image index"): "image_index",
            _first_present(frame, _LABEL_ALIASES, "finding labels"): "finding_labels",
            _first_present(frame, _PATIENT_ALIASES, "patient id"): "patient_id",
        }
    )
    view_column = _first_present(frame, _VIEW_ALIASES, "", required=False)
    if view_column:
        renamed = renamed.rename(columns={view_column: "view_position"})
    elif "view_position" not in renamed.columns:
        renamed["view_position"] = pd.NA

    missing = [name for name in ("image_index", "finding_labels", "patient_id") if name not in renamed]
    if missing:
        raise ValueError(f"{path} is missing {missing}. Columns found: {list(frame.columns)}")

    renamed = renamed.copy()
    renamed["image_index"] = renamed["image_index"].map(_as_clean_str)
    renamed["patient_id"] = renamed["patient_id"].map(_as_patient_id)
    renamed["finding_labels"] = renamed["finding_labels"].fillna("No Finding").map(_as_clean_str)
    renamed.loc[renamed["finding_labels"] == "", "finding_labels"] = "No Finding"

    duplicate_count = int(renamed["image_index"].duplicated().sum())
    if duplicate_count:
        logger.warning("Dropping %s duplicate image index rows", duplicate_count)
        renamed = renamed.drop_duplicates(subset=["image_index"], keep="first")

    renamed = renamed.reset_index(drop=True)
    logger.info(
        "Loaded %s images from %s patients in %s",
        len(renamed),
        renamed["patient_id"].nunique(),
        path,
    )
    return renamed


def filter_view(frame: pd.DataFrame, view_position: str | None) -> pd.DataFrame:
    """Keep one view, such as PA, when the column is present."""
    if not view_position:
        return frame
    if "view_position" not in frame.columns or frame["view_position"].isna().all():
        raise ValueError(
            f"view_position is set to {view_position!r}, but the CSV has no View Position column."
        )
    selected = frame[frame["view_position"].astype(str).str.upper() == view_position.upper()]
    if selected.empty:
        raise ValueError(f"No rows left after filtering to view {view_position!r}.")
    return selected.reset_index(drop=True)


def filter_existing_images(frame: pd.DataFrame, image_dir: str | Path) -> pd.DataFrame:
    """Drop rows whose PNG is not on disk.

    A partial download can still be used for a smoke run. If every file is
    missing, training stops with the directory it looked in.
    """
    root = Path(image_dir)
    keep = [(root / name).is_file() for name in frame["image_index"]]
    missing = len(keep) - sum(keep)
    if missing:
        logger.warning("Dropping %s rows with no file under %s", missing, root)
    if missing == len(frame):
        raise FileNotFoundError(
            f"None of the {len(frame)} images exist under {root}. "
            "Download the NIH archives or point image_dir at the PNG folder."
        )
    mask = pd.Series(keep, index=frame.index)
    return frame.loc[mask].reset_index(drop=True)


def limit_per_split(frame: pd.DataFrame, max_per_split: int | None) -> pd.DataFrame:
    """Keep a deterministic prefix of each split, for debugging only."""
    if max_per_split is None:
        return frame
    if "split" not in frame.columns:
        raise ValueError("limit_per_split requires a split column.")
    parts = []
    for _, group in frame.groupby("split", sort=True):
        parts.append(group.sort_values("image_index").head(max_per_split))
    return pd.concat(parts, ignore_index=True)


def _first_present(
    frame: pd.DataFrame,
    aliases: tuple[str, ...],
    label: str,
    *,
    required: bool = True,
) -> str:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    if not required:
        return ""
    raise ValueError(f"Missing a {label} column. Columns found: {list(frame.columns)}")


def _as_clean_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _as_patient_id(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = _as_clean_str(value)
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text
