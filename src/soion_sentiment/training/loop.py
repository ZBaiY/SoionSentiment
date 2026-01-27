from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from soion_sentiment.config import Config
from soion_sentiment.training.checkpoint import save_checkpoint
from soion_sentiment.training.metrics import compute_metrics
from soion_sentiment.training.optim import build_optimizer, build_scheduler


def _maybe_oom(e: RuntimeError) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "mps backend out of memory" in msg


def _log_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _build_loss(cfg: Config, class_weights: torch.Tensor | None):
    kwargs = {"reduction": "mean"}
    if class_weights is not None:
        kwargs["weight"] = class_weights
    if cfg.training.label_smoothing > 0:
        kwargs["label_smoothing"] = cfg.training.label_smoothing
    return torch.nn.CrossEntropyLoss(**kwargs)


def _class_weights(cfg: Config, train_dataset, device: torch.device) -> torch.Tensor | None:
    if cfg.training.imbalance_strategy != "reweight":
        return None
    num_labels = len(cfg.model.labels)
    counts = torch.zeros(num_labels, dtype=torch.float32)
    labels = train_dataset["label"]
    for i in range(num_labels):
        counts[i] = sum(1 for v in labels if v == i)
    total = counts.sum().item()
    weights = torch.zeros_like(counts)
    for i, c in enumerate(counts.tolist()):
        weights[i] = (total / (num_labels * c)) if c > 0 else 0.0
    return weights.to(device)


def _eval_epoch(model, loader: DataLoader, device: torch.device, cfg: Config) -> dict[str, Any]:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits
            batch_preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            labels_tensor = batch["labels"] if "labels" in batch else batch["label"]
            batch_labels = labels_tensor.detach().cpu().tolist()
            preds.extend(batch_preds)
            labels.extend(batch_labels)
    return compute_metrics(preds, labels, cfg)


def train(
    cfg: Config,
    *,
    run_dir: Path,
    model,
    tokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    precision: str,
) -> dict[str, Any]:
    model.to(device)
    optimizer = build_optimizer(cfg, model)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.training.grad_accum_steps)
    if cfg.training.max_steps is not None:
        total_steps = cfg.training.max_steps
        total_epochs = math.ceil(total_steps / steps_per_epoch)
    else:
        total_epochs = cfg.training.epochs
        total_steps = total_epochs * steps_per_epoch

    scheduler = build_scheduler(cfg, optimizer, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and device.type == "cuda"))

    class_weights = _class_weights(cfg, train_loader.dataset, device)
    loss_fn = _build_loss(cfg, class_weights)

    metrics_path = run_dir / cfg.logging.metrics_filename
    best_metric = None
    best_epoch = 0
    best_step = 0
    no_improve = 0
    best_early_metric = None

    global_step = 0
    for epoch in range(1, total_epochs + 1):
        model.train()
        running_loss = 0.0
        step_in_epoch = 0

        for batch in train_loader:
            step_in_epoch += 1
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels_tensor = batch["labels"] if "labels" in batch else batch["label"]
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    out = model(**batch)
                    logits = out.logits
                    loss = loss_fn(logits, labels_tensor)
                    loss = loss / cfg.training.grad_accum_steps
                scaler.scale(loss).backward()
            except RuntimeError as e:
                if _maybe_oom(e):
                    raise RuntimeError(
                        "Out of memory during training step. Reduce batch size or max_length."
                    ) from e
                raise

            if step_in_epoch % cfg.training.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                if cfg.training.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                running_loss += loss.item() * cfg.training.grad_accum_steps
                if global_step % cfg.logging.log_every_steps == 0:
                    _log_jsonl(
                        metrics_path,
                        {
                            "split": "train",
                            "epoch": epoch,
                            "step": global_step,
                            "loss": running_loss / cfg.logging.log_every_steps,
                        },
                    )
                    running_loss = 0.0

                if cfg.training.eval_every_steps is not None and global_step % cfg.training.eval_every_steps == 0:
                    eval_metrics = _eval_epoch(model, val_loader, device, cfg)
                    eval_metrics.update({"split": "val", "epoch": epoch, "step": global_step})
                    _log_jsonl(metrics_path, eval_metrics)

            if cfg.training.max_steps is not None and global_step >= cfg.training.max_steps:
                break

        eval_metrics = _eval_epoch(model, val_loader, device, cfg)
        eval_metrics.update({"split": "val", "epoch": epoch, "step": global_step})
        _log_jsonl(metrics_path, eval_metrics)

        metric_key = cfg.eval.metric
        current = eval_metrics.get(metric_key)
        improved = False
        if current is not None:
            if best_metric is None:
                improved = True
            elif cfg.eval.mode == "max" and current > best_metric:
                improved = True
            elif cfg.eval.mode == "min" and current < best_metric:
                improved = True

        if improved:
            best_metric = current
            best_epoch = epoch
            best_step = global_step
            no_improve = 0
            if cfg.logging.save_best:
                save_checkpoint(
                    run_dir / "best",
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    step=global_step,
                    epoch=epoch,
                    cfg=cfg,
                    metrics=eval_metrics,
                )
        if cfg.training.early_stopping.enabled:
            early_key = cfg.training.early_stopping.metric
            early_val = eval_metrics.get(early_key)
            improved_early = False
            if early_val is not None:
                if best_early_metric is None:
                    improved_early = True
                elif cfg.training.early_stopping.mode == "max" and (
                    early_val > best_early_metric + cfg.training.early_stopping.min_delta
                ):
                    improved_early = True
                elif cfg.training.early_stopping.mode == "min" and (
                    early_val < best_early_metric - cfg.training.early_stopping.min_delta
                ):
                    improved_early = True
            if improved_early:
                best_early_metric = early_val
                no_improve = 0
            else:
                no_improve += 1

        if cfg.logging.save_last:
            save_checkpoint(
                run_dir / "last",
                model,
                tokenizer,
                optimizer,
                scheduler,
                step=global_step,
                epoch=epoch,
                cfg=cfg,
                metrics=eval_metrics,
            )

        if cfg.training.early_stopping.enabled and no_improve >= cfg.training.early_stopping.patience:
            break

        if cfg.training.max_steps is not None and global_step >= cfg.training.max_steps:
            break

    summary = {
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "total_steps": global_step,
    }
    (run_dir / cfg.logging.summary_filename).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
