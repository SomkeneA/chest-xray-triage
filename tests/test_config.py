"""Experiment YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from cxr_triage.config import load_config


def test_default_and_smoke_configs_load():
    default = load_config("configs/default.yaml")
    smoke = load_config("configs/smoke.yaml")
    assert default.model.name == "densenet121"
    assert default.model.pretrained is True
    assert default.train.pos_weight == "auto"
    assert smoke.model.pretrained is False
    assert smoke.train.device == "cpu"
    assert smoke.train.epochs == 1
    assert Path(smoke.data.csv_path).is_file()
    assert Path(smoke.data.image_dir, "00000001_000.png").is_file()


def test_unknown_key_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "seed: 1\nnot_a_setting: true\ndata:\n  csv_path: labels.csv\n  image_dir: images\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not_a_setting"):
        load_config(path)


def test_ratios_must_sum_to_one(tmp_path: Path):
    path = tmp_path / "ratios.yaml"
    path.write_text(
        "data:\n"
        "  csv_path: labels.csv\n"
        "  image_dir: images\n"
        "  train_ratio: 0.8\n"
        "  val_ratio: 0.8\n"
        "  test_ratio: 0.8\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sum to 1"):
        load_config(path)
