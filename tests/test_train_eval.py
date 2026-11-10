"""One CPU epoch on the synthetic fixture, then the evaluation report."""

from __future__ import annotations

from pathlib import Path

from cxr_triage.config import load_config
from cxr_triage.evaluation.evaluate import run_evaluation
from cxr_triage.labels import CHEST14_LABELS, CLINICAL_DISCLAIMER
from cxr_triage.training.train import run_training


def test_smoke_training_logs_mlflow_and_writes_metrics(tmp_path: Path):
    cfg = load_config("configs/smoke.yaml")
    cfg.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.mlflow.tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg.mlflow.enabled = True
    cfg.eval.output_dir = str(tmp_path / "eval")

    result = run_training(cfg)
    assert result["epochs_ran"] == 1
    assert Path(result["best_checkpoint"]).is_file()
    assert Path(result["last_checkpoint"]).is_file()
    assert (tmp_path / "mlflow.db").is_file()

    report = run_evaluation(cfg, result["best_checkpoint"], split="test", device_name="cpu")
    assert report["split"] == "test"
    assert report["disclaimer"] == CLINICAL_DISCLAIMER
    assert set(report["per_label"]) == set(CHEST14_LABELS)
    for name in ("mean_auroc", "mean_sensitivity", "mean_specificity", "mean_f1", "mean_ece"):
        assert name in report
    assert (tmp_path / "eval" / "metrics.json").is_file()
    assert (tmp_path / "eval" / "per_label.csv").is_file()
    assert (tmp_path / "eval" / "reliability.png").is_file()
