"""YAML experiment configuration.

Unknown keys are rejected so a typo cannot silently change a training run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_SPLIT_STRATEGIES = {"patient", "patient_stratified"}
_OPTIMIZERS = {"adam", "adamw"}
_SCHEDULERS = {"none", "cosine"}


@dataclass
class DataConfig:
    csv_path: str
    image_dir: str
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    split_strategy: str = "patient_stratified"
    view_position: str | None = None
    max_per_split: int | None = None
    drop_missing: bool = True


@dataclass
class ModelConfig:
    name: str = "densenet121"
    pretrained: bool = True
    dropout: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    pos_weight: str | list[float] = "auto"
    pos_weight_max: float | None = 20.0
    early_stopping_patience: int | None = 3
    grad_clip: float = 1.0
    amp: bool = True
    device: str = "auto"


@dataclass
class MLflowConfig:
    enabled: bool = True
    experiment_name: str = "chest-xray-triage"
    tracking_uri: str = "./mlruns"


@dataclass
class EvalConfig:
    threshold: float | str = 0.5
    ece_bins: int = 15
    output_dir: str = "outputs/eval"


@dataclass
class ExperimentConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    mlflow: MLflowConfig
    eval: EvalConfig
    checkpoint_dir: str = "checkpoints"


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment file and reject unknown or inconsistent settings."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")

    data = _require_mapping(_pop(raw, "data", required=True), "data")
    model = _require_mapping(_pop(raw, "model", {}), "model")
    train = _require_mapping(_pop(raw, "train", {}), "train")
    mlflow = _require_mapping(_pop(raw, "mlflow", {}), "mlflow")
    eval_cfg = _require_mapping(_pop(raw, "eval", {}), "eval")

    cfg = ExperimentConfig(
        seed=int(_pop(raw, "seed", 42)),
        checkpoint_dir=str(_pop(raw, "checkpoint_dir", "checkpoints")),
        data=DataConfig(
            csv_path=str(_pop(data, "csv_path", required=True)),
            image_dir=str(_pop(data, "image_dir", required=True)),
            train_ratio=float(_pop(data, "train_ratio", 0.70)),
            val_ratio=float(_pop(data, "val_ratio", 0.10)),
            test_ratio=float(_pop(data, "test_ratio", 0.20)),
            image_size=int(_pop(data, "image_size", 224)),
            batch_size=int(_pop(data, "batch_size", 32)),
            num_workers=int(_pop(data, "num_workers", 4)),
            split_strategy=str(_pop(data, "split_strategy", "patient_stratified")),
            view_position=_optional_str(_pop(data, "view_position", None)),
            max_per_split=_optional_int(_pop(data, "max_per_split", None)),
            drop_missing=bool(_pop(data, "drop_missing", True)),
        ),
        model=ModelConfig(
            name=str(_pop(model, "name", "densenet121")),
            pretrained=bool(_pop(model, "pretrained", True)),
            dropout=float(_pop(model, "dropout", 0.1)),
        ),
        train=TrainConfig(
            epochs=int(_pop(train, "epochs", 10)),
            lr=float(_pop(train, "lr", 1e-4)),
            weight_decay=float(_pop(train, "weight_decay", 1e-4)),
            optimizer=str(_pop(train, "optimizer", "adamw")),
            scheduler=str(_pop(train, "scheduler", "cosine")),
            pos_weight=_parse_pos_weight(_pop(train, "pos_weight", "auto")),
            pos_weight_max=_optional_float(_pop(train, "pos_weight_max", 20.0)),
            early_stopping_patience=_optional_int(_pop(train, "early_stopping_patience", 3)),
            grad_clip=float(_pop(train, "grad_clip", 1.0)),
            amp=bool(_pop(train, "amp", True)),
            device=str(_pop(train, "device", "auto")),
        ),
        mlflow=MLflowConfig(
            enabled=bool(_pop(mlflow, "enabled", True)),
            experiment_name=str(_pop(mlflow, "experiment_name", "chest-xray-triage")),
            tracking_uri=str(_pop(mlflow, "tracking_uri", "./mlruns")),
        ),
        eval=EvalConfig(
            threshold=_parse_threshold(_pop(eval_cfg, "threshold", 0.5)),
            ece_bins=int(_pop(eval_cfg, "ece_bins", 15)),
            output_dir=str(_pop(eval_cfg, "output_dir", "outputs/eval")),
        ),
    )
    _reject_unknown(raw, "root")
    _reject_unknown(data, "data")
    _reject_unknown(model, "model")
    _reject_unknown(train, "train")
    _reject_unknown(mlflow, "mlflow")
    _reject_unknown(eval_cfg, "eval")
    _validate(cfg)
    return cfg


def _parse_pos_weight(value: Any) -> str | list[float]:
    if value is None or value == "none":
        return "none"
    if value == "auto":
        return "auto"
    if isinstance(value, list):
        return [float(item) for item in value]
    raise ValueError("train.pos_weight must be auto, none, or a list of 14 weights.")


def _parse_threshold(value: Any) -> float | str:
    if value == "best_f1":
        return "best_f1"
    if isinstance(value, (int, float)):
        threshold = float(value)
        if not 0.0 < threshold < 1.0:
            raise ValueError("eval.threshold must be between 0 and 1, or best_f1.")
        return threshold
    raise ValueError("eval.threshold must be a probability or best_f1.")


def _validate(cfg: ExperimentConfig) -> None:
    ratios = (cfg.data.train_ratio, cfg.data.val_ratio, cfg.data.test_ratio)
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("Split ratios must all be positive.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1. Got {sum(ratios):.6f}.")
    if cfg.data.split_strategy not in _SPLIT_STRATEGIES:
        raise ValueError(
            f"Unknown split_strategy {cfg.data.split_strategy!r}. "
            f"Expected one of {sorted(_SPLIT_STRATEGIES)}."
        )
    if cfg.data.image_size < 32:
        raise ValueError("image_size must be at least 32 for DenseNet121.")
    if cfg.data.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if cfg.data.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if cfg.model.name != "densenet121":
        raise ValueError("Only model.name densenet121 is implemented.")
    if not 0.0 <= cfg.model.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if cfg.train.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if cfg.train.lr <= 0:
        raise ValueError("lr must be positive.")
    if cfg.train.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")
    if cfg.train.optimizer not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {cfg.train.optimizer!r}.")
    if cfg.train.scheduler not in _SCHEDULERS:
        raise ValueError(f"Unknown scheduler {cfg.train.scheduler!r}.")
    if isinstance(cfg.train.pos_weight, list) and len(cfg.train.pos_weight) != 14:
        raise ValueError("A pos_weight list must contain 14 values, one per finding.")
    if cfg.train.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("train.device must be auto, cpu, or cuda.")
    if cfg.eval.ece_bins < 2:
        raise ValueError("ece_bins must be at least 2.")
    if cfg.train.early_stopping_patience is not None and cfg.train.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative.")


def _pop(data: dict[str, Any], key: str, default: Any = None, *, required: bool = False) -> Any:
    if key not in data:
        if required:
            raise ValueError(f"Missing config key: {key}")
        return default
    return data.pop(key)


def _reject_unknown(data: dict[str, Any], prefix: str) -> None:
    if data:
        keys = ", ".join(sorted(map(str, data)))
        raise ValueError(f"Unknown config keys under {prefix}: {keys}")


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name} must be a mapping.")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
