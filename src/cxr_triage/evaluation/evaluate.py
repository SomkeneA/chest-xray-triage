"""Evaluate a checkpoint on a patient-level split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from cxr_triage.config import ExperimentConfig, load_config
from cxr_triage.data.dataset import ChestXrayDataset
from cxr_triage.data.prepare import prepare_experiment_frame
from cxr_triage.data.transforms import build_transforms
from cxr_triage.evaluation.calibration import save_reliability_diagrams
from cxr_triage.evaluation.metrics import evaluate_predictions, select_f1_threshold
from cxr_triage.evaluation.predict import collect_predictions
from cxr_triage.labels import CLINICAL_DISCLAIMER
from cxr_triage.models.checkpoint import load_checkpoint
from cxr_triage.seed import resolve_device, set_seed

logger = logging.getLogger(__name__)


def run_evaluation(
    cfg: ExperimentConfig,
    checkpoint: str | Path,
    *,
    split: str = "test",
    device_name: str | None = None,
) -> dict:
    """Score one split and write metrics, a CSV, and reliability diagrams."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test.")
    logger.info(CLINICAL_DISCLAIMER)
    set_seed(cfg.seed)
    device = resolve_device(device_name or cfg.train.device)
    loaded = load_checkpoint(checkpoint, device)
    _warn_if_config_drift(cfg, loaded.config)

    frame = prepare_experiment_frame(cfg)
    image_size = loaded.image_size
    if image_size != cfg.data.image_size:
        logger.warning(
            "Using checkpoint image_size %s instead of config image_size %s.",
            image_size,
            cfg.data.image_size,
        )

    thresholds = _thresholds(cfg, frame, loaded, image_size, device, split)
    truth, probabilities = _predict_split(frame, split, cfg, loaded, image_size, device)
    report = evaluate_predictions(
        truth,
        probabilities,
        labels=loaded.labels,
        thresholds=thresholds,
        ece_bins=cfg.eval.ece_bins,
    )
    report["split"] = split
    report["checkpoint"] = str(checkpoint)
    report["disclaimer"] = CLINICAL_DISCLAIMER

    output_dir = Path(cfg.eval.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_table(report, output_dir / "per_label.csv")
    save_reliability_diagrams(
        truth,
        probabilities,
        loaded.labels,
        output_dir / "reliability.png",
        n_bins=cfg.eval.ece_bins,
    )
    logger.info("Wrote %s", metrics_path)
    _log_summary(report)
    return report


def _thresholds(cfg, frame, loaded, image_size, device, split) -> list[float] | float:
    if cfg.eval.threshold != "best_f1":
        return cfg.eval.threshold
    if split == "val":
        logger.warning(
            "best_f1 thresholds are being selected on val and reported on val. "
            "That score is optimistic. Report test after selecting on val."
        )
    truth, probabilities = _predict_split(frame, "val", cfg, loaded, image_size, device)
    chosen = [
        select_f1_threshold(truth[:, index], probabilities[:, index]) for index in range(truth.shape[1])
    ]
    logger.info("Validation F1 thresholds: %s", [round(value, 3) for value in chosen])
    return chosen


def _predict_split(frame, split, cfg, loaded, image_size, device):
    subset = frame[frame["split"] == split].reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"The {split} split has no images.")
    dataset = ChestXrayDataset(
        subset,
        cfg.data.image_dir,
        build_transforms(image_size, train=False),
        labels=loaded.labels,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    return collect_predictions(loaded.model, loader, device)


def _write_table(report: dict, path: Path) -> None:
    rows = []
    for name, metrics in report["per_label"].items():
        rows.append({"label": name, **metrics})
    pd.DataFrame(rows).to_csv(path, index=False)


def _log_summary(report: dict) -> None:
    def _fmt(value: float | None) -> str:
        return "undefined" if value is None else f"{value:.4f}"

    logger.info(
        "split=%s n=%s mean_auroc=%s mean_sensitivity=%s mean_specificity=%s mean_f1=%s mean_ece=%s",
        report["split"],
        report["num_images"],
        _fmt(report["mean_auroc"]),
        _fmt(report["mean_sensitivity"]),
        _fmt(report["mean_specificity"]),
        _fmt(report["mean_f1"]),
        _fmt(report["mean_ece"]),
    )


def _warn_if_config_drift(cfg: ExperimentConfig, saved: dict) -> None:
    saved_data = saved.get("data") if isinstance(saved, dict) else None
    saved_seed = saved.get("seed") if isinstance(saved, dict) else None
    if not isinstance(saved_data, dict):
        return
    if saved_seed is not None and int(saved_seed) != cfg.seed:
        logger.warning(
            "Checkpoint seed is %s and this config seed is %s. The test patients may differ.",
            saved_seed,
            cfg.seed,
        )
    for key in ("train_ratio", "val_ratio", "test_ratio", "split_strategy"):
        if key in saved_data and saved_data[key] != getattr(cfg.data, key):
            logger.warning(
                "Checkpoint data.%s is %s and this config is %s.",
                key,
                saved_data[key],
                getattr(cfg.data, key),
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a chest X-ray triage checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default=None, help="auto, cpu, or cuda. Defaults to the config.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run_evaluation(
        load_config(args.config),
        args.checkpoint,
        split=args.split,
        device_name=args.device,
    )
    mean = report["mean_auroc"]
    logger.info("mean AUROC: %s", "undefined" if mean is None else f"{mean:.4f}")


if __name__ == "__main__":
    main()
