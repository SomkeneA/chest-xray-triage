"""Deterministic fake radiographs for tests and CPU smoke runs.

These images are drawn shapes, not clinical data, and they are not sampled
from NIH ChestX-ray14.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from cxr_triage.labels import CHEST14_LABELS

# image_index, finding_labels, follow_up, patient_id, age, sex, view
FIXTURE_ROWS: tuple[tuple[str, str, int, str, int, str, str], ...] = (
    ("00000001_000.png", "Cardiomegaly", 0, "1", 58, "M", "PA"),
    ("00000001_001.png", "Cardiomegaly|Effusion", 1, "1", 58, "M", "PA"),
    ("00000002_000.png", "No Finding", 0, "2", 34, "F", "PA"),
    ("00000003_000.png", "Pneumonia", 0, "3", 67, "M", "AP"),
    ("00000003_001.png", "Pneumonia|Infiltration", 1, "3", 67, "M", "AP"),
    ("00000004_000.png", "Atelectasis|Mass", 0, "4", 51, "F", "PA"),
    ("00000005_000.png", "No Finding", 0, "5", 29, "F", "PA"),
    ("00000006_000.png", "Nodule", 0, "6", 72, "M", "PA"),
    ("00000007_000.png", "Pneumothorax", 0, "7", 44, "F", "AP"),
    ("00000007_001.png", "Edema|Consolidation", 1, "7", 44, "F", "AP"),
    ("00000008_000.png", "Hernia", 0, "8", 63, "M", "PA"),
    ("00000008_001.png", "Emphysema|Fibrosis|Pleural_Thickening", 1, "8", 63, "M", "PA"),
)

CSV_COLUMNS = (
    "Image Index",
    "Finding Labels",
    "Follow-up #",
    "Patient ID",
    "Patient Age",
    "Patient Gender",
    "View Position",
)


def render_image(seed: int, size: int = 64) -> Image.Image:
    """Draw a grayscale chest-like oval and one bright blob."""
    rng = np.random.default_rng(seed)
    canvas = np.full((size, size), int(rng.integers(18, 36)), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    center = size // 2
    body = ((yy - center) / (size * 0.46)) ** 2 + ((xx - center) / (size * 0.34)) ** 2 <= 1.0
    canvas[body] = int(rng.integers(70, 120))
    blob_y = int(rng.integers(size // 5, 4 * size // 5))
    blob_x = int(rng.integers(size // 5, 4 * size // 5))
    radius = max(2, size // 12)
    blob = (yy - blob_y) ** 2 + (xx - blob_x) ** 2 <= radius**2
    canvas[blob] = 230
    return Image.fromarray(canvas, mode="L")


def write_fixture(root: str | Path, *, image_size: int = 64) -> Path:
    """Write the synthetic CSV and PNGs. Returns the directory."""
    destination = Path(root)
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "Data_Entry_2017.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in FIXTURE_ROWS:
            writer.writerow(row)
            seed = sum(ord(character) for character in row[0])
            render_image(seed, image_size).save(image_dir / row[0])
    return destination


def fixture_covers_all_labels() -> bool:
    """True when every canonical finding appears in the synthetic rows."""
    present: set[str] = set()
    for row in FIXTURE_ROWS:
        present.update(part.strip() for part in row[1].split("|") if part.strip() != "No Finding")
    return set(CHEST14_LABELS).issubset(present)


if __name__ == "__main__":
    written = write_fixture(Path("tests/fixtures/synthetic"))
    print(f"Wrote synthetic fixture to {written}")
