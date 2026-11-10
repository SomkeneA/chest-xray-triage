"""Metadata parsing, label encoding, and the synthetic radiograph fixture."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from cxr_triage.data.dataset import ChestXrayDataset
from cxr_triage.data.metadata import filter_existing_images, filter_view, load_metadata
from cxr_triage.data.synthetic import fixture_covers_all_labels, write_fixture
from cxr_triage.data.transforms import build_transforms
from cxr_triage.labels import CHEST14_LABELS, encode_labels

FIXTURE = Path("tests/fixtures/synthetic")


def test_encode_labels_and_unknown_token():
    cardiomegaly = encode_labels("Cardiomegaly")
    assert cardiomegaly[CHEST14_LABELS.index("Cardiomegaly")] == 1
    assert int(cardiomegaly.sum()) == 1
    assert int(encode_labels("No Finding").sum()) == 0
    assert int(encode_labels("").sum()) == 0
    pleural = encode_labels("Pleural Thickening|Hernia")
    assert pleural[CHEST14_LABELS.index("Pleural_Thickening")] == 1
    assert pleural[CHEST14_LABELS.index("Hernia")] == 1
    unknown = encode_labels("NotARealDisease|Effusion")
    assert unknown[CHEST14_LABELS.index("Effusion")] == 1
    assert int(unknown.sum()) == 1


def test_official_nih_header_keeps_zero_padded_patient_ids(tmp_path: Path):
    path = tmp_path / "Data_Entry_2017.csv"
    path.write_text(
        "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,"
        "View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y],\n"
        "00000001_000.png,Cardiomegaly,000,00000001,058Y,M,PA,2682,2749,0.143,0.143\n"
        "00000001_001.png,Cardiomegaly|Emphysema,001,00000001,058Y,M,PA,2894,2729,0.143,0.143\n",
        encoding="utf-8",
    )
    frame = load_metadata(path)
    assert list(frame["image_index"]) == ["00000001_000.png", "00000001_001.png"]
    assert frame.loc[0, "finding_labels"] == "Cardiomegaly"
    assert frame.loc[1, "finding_labels"] == "Cardiomegaly|Emphysema"
    assert set(frame["patient_id"]) == {"00000001"}
    assert frame.loc[0, "view_position"] == "PA"


def test_view_filter_and_duplicate_images(tmp_path: Path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "Image Index,Finding Labels,Patient ID,View Position\n"
        "a.png,No Finding,1,PA\n"
        "a.png,Effusion,1,PA\n"
        "b.png,Edema,2,AP\n",
        encoding="utf-8",
    )
    frame = load_metadata(path)
    assert list(frame["image_index"]) == ["a.png", "b.png"]
    frontal = filter_view(frame, "pa")
    assert list(frontal["image_index"]) == ["a.png"]


def test_synthetic_dataset_shapes_and_targets(tmp_path: Path):
    root = write_fixture(tmp_path / "synthetic", image_size=32)
    frame = load_metadata(root / "Data_Entry_2017.csv")
    assert len(frame) == 12
    assert frame["patient_id"].nunique() == 8
    assert fixture_covers_all_labels()
    dataset = ChestXrayDataset(frame, root / "images", build_transforms(32, train=False))
    image, target = dataset[0]
    assert image.shape == (3, 32, 32)
    assert target.shape == (14,)
    assert target[CHEST14_LABELS.index("Cardiomegaly")].item() == 1
    assert int(target.sum()) == 1
    both = dataset[1][1]
    assert int(both.sum()) == 2
    no_finding = int(frame.index[frame["finding_labels"] == "No Finding"][0])
    assert float(dataset[no_finding][1].sum()) == 0.0


def test_missing_images_are_dropped(tmp_path: Path):
    root = write_fixture(tmp_path / "synthetic", image_size=16)
    (root / "images" / "00000001_000.png").unlink()
    frame = load_metadata(root / "Data_Entry_2017.csv")
    filtered = filter_existing_images(frame, root / "images")
    assert len(filtered) == 11
    assert "00000001_000.png" not in set(filtered["image_index"])


def test_committed_fixture_opens():
    frame = load_metadata(FIXTURE / "Data_Entry_2017.csv")
    assert len(frame) == 12
    sample = Image.open(FIXTURE / "images" / frame.loc[0, "image_index"])
    assert sample.size == (64, 64)
    assert sample.mode == "L"


def test_transforms_resize_to_square():
    image = Image.new("L", (80, 90), color=30)
    assert build_transforms(64, train=False)(image).shape == (3, 64, 64)
    assert build_transforms(64, train=True)(image).shape == (3, 64, 64)
