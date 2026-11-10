"""Config-driven training with checkpoints and optional MLflow tracking."""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cxr_triage.config import ExperimentConfig, load_config
from cxr_triage.data.dataset import ChestXrayDataset
from cxr_triage.data.prepare import prepare_experiment_frame
from cxr_triage.data.transforms import build_transforms
from cxr_triage.evaluation.metrics import mean_auroc
from cxr_triage.labels import CHEST14_LABELS, CLINICAL_DISCLAIMER, encode_labels
from cxr_triage.models.checkpoint import save_checkpoint
from cxr_triage.models.densenet import build_model
from cxr_triage.seed import resolve_device, set_seed
from cxr_triage.training.loss import build_loss, compute_pos_weight

logger = logging.getLogger(__name__)


def run_training(cfg: ExperimentConfig) -> dict[str, object]:
    """Train DenseNet121 and return the checkpoint paths and selection score."""
    logger.info(CLINICAL_DISCLAIMER)
    set_seed(cfg.seed)
    device = resolve_device(cfg.train.device)
    frame = prepare_experiment_frame(cfg)
    logger.info(
        "Split sizes: %s",
        {name: int((frame["split"] == name).sum()) for name in ("train", "val", "test")},
    )

    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    val_frame = frame[frame["split"] == "val"].reset_index(drop=True)
    pos_weight = _resolve_pos_weight(cfg, train_frame, device)
    if pos_weight is not None:
        logger.info("pos_weight: %s", [round(float(value), 3) for value in pos_weight.tolist()])

    train_loader = _loader(train_frame, cfg, train=True, device=device)
    val_loader = _loader(val_frame, cfg, train=False, device=device)
    model = build_model(
        num_classes=len(CHEST14_LABELS),
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
    ).to(device)
    criterion = build_loss(pos_weight)
    optimizer = _optimizer(model, cfg)
    scheduler = _scheduler(optimizer, cfg)
    use_amp = bool(cfg.train.amp and device.type == "cuda")

    checkpoint_dir = Path(cfg.checkpoint_dir)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    best_score = float("-inf")
    stale_epochs = 0
    epochs_ran = 0
    config_dict = asdict(cfg)
    if pos_weight is None:
        weight_list = None
    else:
        weight_list = [float(value) for value in pos_weight.detach().cpu().tolist()]

    tracking = _tracking(cfg)
    with tracking as run:
        if run is not None:
            import mlflow

            mlflow.log_params(_mlflow_params(cfg))
            if weight_list is not None:
                mlflow.log_dict(
                    {"pos_weight": weight_list, "labels": list(CHEST14_LABELS)},
                    "pos_weight.json",
                )

        for epoch in range(1, cfg.train.epochs + 1):
            epochs_ran = epoch
            train_loss = _run_epoch(
                model, train_loader, criterion, optimizer, device, train=True, cfg=cfg, use_amp=use_amp
            )
            val_loss, val_targets, val_probs = _run_epoch(
                model, val_loader, criterion, optimizer, device, train=False, cfg=cfg, use_amp=False
            )
            val_auroc = mean_auroc(val_targets, val_probs)
            score = _selection_score(val_auroc, val_loss)
            logger.info(
                "epoch %s/%s train_loss=%.4f val_loss=%.4f val_mean_auroc=%s",
                epoch,
                cfg.train.epochs,
                train_loss,
                val_loss,
                "undefined" if val_auroc is None else f"{val_auroc:.4f}",
            )
            if run is not None:
                import mlflow

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                        **({} if val_auroc is None else {"val_mean_auroc": val_auroc}),
                    },
                    step=epoch,
                )
            if scheduler is not None:
                scheduler.step()

            payload_score = None if val_auroc is None else float(val_auroc)
            save_checkpoint(
                last_path,
                model=model,
                labels=CHEST14_LABELS,
                image_size=cfg.data.image_size,
                dropout=cfg.model.dropout,
                epoch=epoch,
                score=payload_score,
                config=config_dict,
                pos_weight=weight_list,
                optimizer=optimizer,
            )
            if score > best_score:
                best_score = score
                stale_epochs = 0
                save_checkpoint(
                    best_path,
                    model=model,
                    labels=CHEST14_LABELS,
                    image_size=cfg.data.image_size,
                    dropout=cfg.model.dropout,
                    epoch=epoch,
                    score=payload_score,
                    config=config_dict,
                    pos_weight=weight_list,
                    optimizer=optimizer,
                )
                logger.info("Saved new best checkpoint to %s", best_path)
            else:
                stale_epochs += 1
                patience = cfg.train.early_stopping_patience
                if patience and stale_epochs >= patience:
                    logger.info("Early stopping after %s epochs without improvement.", patience)
                    break

        if run is not None and best_path.is_file():
            import mlflow

            mlflow.log_artifact(str(best_path))

    reported = None if best_score == float("-inf") else best_score
    return {
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "best_score": reported,
        "epochs_ran": epochs_ran,
    }


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    *,
    train: bool,
    cfg: ExperimentConfig,
    use_amp: bool,
):
    model.train(train)
    # enabled=False is a no-op, so the CPU smoke path does not need a GPU.
    scaler = _grad_scaler(use_amp)
    total_loss = 0.0
    total_count = 0
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for images, batch_targets in loader:
        if train and images.size(0) < 2:
            logger.warning(
                "Skipping a training batch of size %s. BatchNorm needs at least 2 images.",
                images.size(0),
            )
            continue
        images = images.to(device)
        batch_targets = batch_targets.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, batch_targets)
            if train:
                scaler.scale(loss).backward()
                if cfg.train.grad_clip and cfg.train.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().item()) * images.size(0)
        total_count += images.size(0)
        if not train:
            targets.append(batch_targets.detach().cpu().numpy())
            probabilities.append(torch.sigmoid(logits.detach()).cpu().numpy())

    if total_count == 0:
        role = "Training" if train else "Validation"
        raise RuntimeError(
            f"{role} produced no usable batches. DenseNet BatchNorm needs at least two "
            "images in a training batch, and every split needs at least one image."
        )
    mean_loss = total_loss / total_count
    if train:
        return mean_loss
    return mean_loss, np.concatenate(targets), np.concatenate(probabilities)


def _loader(frame, cfg: ExperimentConfig, *, train: bool, device: torch.device) -> DataLoader:
    dataset = ChestXrayDataset(
        frame,
        cfg.data.image_dir,
        build_transforms(cfg.data.image_size, train=train),
    )
    generator = None
    if train:
        generator = torch.Generator()
        generator.manual_seed(cfg.seed)
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=train,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )


def _resolve_pos_weight(cfg: ExperimentConfig, train_frame, device: torch.device) -> torch.Tensor | None:
    setting = cfg.train.pos_weight
    if setting == "none":
        return None
    if setting == "auto":
        targets = np.stack([encode_labels(value) for value in train_frame["finding_labels"]])
        return compute_pos_weight(targets, cfg.train.pos_weight_max).to(device)
    return torch.tensor(setting, dtype=torch.float32, device=device)


def _optimizer(model: torch.nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    if cfg.train.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    if cfg.train.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    raise ValueError(f"Unknown optimizer {cfg.train.optimizer!r}.")


def _scheduler(optimizer: torch.optim.Optimizer, cfg: ExperimentConfig):
    if cfg.train.scheduler == "none":
        return None
    if cfg.train.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    raise ValueError(f"Unknown scheduler {cfg.train.scheduler!r}.")


def _grad_scaler(enabled: bool):
    """AMP scaler. Disabled on CPU, where ``enabled`` is false and this is a no-op."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _selection_score(val_auroc: float | None, val_loss: float) -> float:
    """Prefer mean validation AUROC, and fall back to loss when AUROC is undefined."""
    if val_auroc is None or np.isnan(val_auroc):
        return -float(val_loss)
    return float(val_auroc)


def _tracking(cfg: ExperimentConfig):
    if not cfg.mlflow.enabled:
        return nullcontext(None)
    import mlflow

    os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    _prepare_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    return mlflow.start_run(run_name="densenet121")


def _prepare_tracking_uri(uri: str) -> None:
    """Create the parent directory for a sqlite store.

    MLflow 3 rejects the old ``./mlruns`` directory backend unless
    ``MLFLOW_ALLOW_FILE_STORE`` is set. SQLite is the supported local default.
    """
    if uri.startswith("sqlite:"):
        path = uri.removeprefix("sqlite:///")
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        return
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    Path(uri).mkdir(parents=True, exist_ok=True)


def _mlflow_params(cfg: ExperimentConfig) -> dict[str, str | int | float | bool]:
    pos_weight = cfg.train.pos_weight if isinstance(cfg.train.pos_weight, str) else "list"
    return {
        "seed": cfg.seed,
        "model": cfg.model.name,
        "pretrained": cfg.model.pretrained,
        "dropout": cfg.model.dropout,
        "image_size": cfg.data.image_size,
        "batch_size": cfg.data.batch_size,
        "split_strategy": cfg.data.split_strategy,
        "epochs": cfg.train.epochs,
        "lr": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
        "optimizer": cfg.train.optimizer,
        "scheduler": cfg.train.scheduler,
        "pos_weight": pos_weight,
        "pos_weight_max": -1 if cfg.train.pos_weight_max is None else cfg.train.pos_weight_max,
        "device": cfg.train.device,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the chest X-ray triage model.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    result = run_training(cfg)
    logger.info(
        "Finished %s epoch(s). Best checkpoint: %s",
        result["epochs_ran"],
        result["best_checkpoint"],
    )


if __name__ == "__main__":
    main()
