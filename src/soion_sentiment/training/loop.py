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
        # apply label smoothing if specified
        # "I am not sure the label is correct (human agreement), so I will smooth the one-hot labels"
        kwargs["label_smoothing"] = cfg.training.label_smoothing
    return torch.nn.CrossEntropyLoss(**kwargs)


def _class_weights(cfg: Config, train_dataset, device: torch.device) -> torch.Tensor | None:
    # rescale loss by class weights to handle class imbalance
    # only used if cfg.training.imbalance_strategy == "reweight"
    # like 1 is more frequent, 0 is less frequent
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


def _compute_training_steps(cfg: Config, train_loader_len: int) -> tuple[int, int, int]:
    steps_per_epoch = math.ceil(train_loader_len / cfg.training.grad_accum_steps)
    if cfg.training.max_steps is not None:
        total_steps = cfg.training.max_steps
        total_epochs = math.ceil(total_steps / steps_per_epoch)
    else:
        total_epochs = cfg.training.epochs
        total_steps = total_epochs * steps_per_epoch
    return steps_per_epoch, total_steps, total_epochs


def _eval_epoch(model, loader: DataLoader, device: torch.device, cfg: Config) -> dict[str, Any]:
    was_training = model.training
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
    metrics = compute_metrics(preds, labels, cfg)
    if was_training:
        model.train()
    return metrics


def _handle_eval(
    cfg: Config,
    *,
    eval_metrics: dict[str, Any],
    run_dir: Path,
    model,
    tokenizer,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    state: dict[str, Any],
    save_last: bool,
) -> bool:
    state.setdefault("best_metric", None) ## -- .get + if not exists -> None, align schema
    state.setdefault("best_epoch", 0)
    state.setdefault("best_step", 0)
    state.setdefault("early_best_metric", None)
    state.setdefault("early_no_improve", 0)
    metric_key = cfg.eval.metric
    current = eval_metrics.get(metric_key)
    improved = False
    if current is not None:
        if state["best_metric"] is None:
            improved = True
        elif cfg.eval.mode == "max" and current > state["best_metric"]:
            improved = True
        elif cfg.eval.mode == "min" and current < state["best_metric"]:
            improved = True

    if improved:
        state["best_metric"] = current
        state["best_epoch"] = epoch
        state["best_step"] = step
        if cfg.logging.save_best:
            save_checkpoint(
                run_dir / "best",
                model,
                tokenizer,
                optimizer,
                scheduler,
                step=step,
                epoch=epoch,
                cfg=cfg,
                metrics=eval_metrics,
            )

    if cfg.training.early_stopping.enabled:
        early_key = cfg.training.early_stopping.metric
        early_val = eval_metrics.get(early_key)
        improved_early = False
        if early_val is not None:
            if state["early_best_metric"] is None:
                improved_early = True
            elif cfg.training.early_stopping.mode == "max" and (
                early_val > state["early_best_metric"] + cfg.training.early_stopping.min_delta
            ):
                improved_early = True
            elif cfg.training.early_stopping.mode == "min" and (
                early_val < state["early_best_metric"] - cfg.training.early_stopping.min_delta
            ):
                improved_early = True
        if improved_early:
            state["early_best_metric"] = early_val
            state["early_no_improve"] = 0
        else:
            state["early_no_improve"] += 1

    if save_last and cfg.logging.save_last:
        save_checkpoint(
            run_dir / "last",
            model,
            tokenizer,
            optimizer,
            scheduler,
            step=step,
            epoch=epoch,
            cfg=cfg,
            metrics=eval_metrics,
        )

    if cfg.training.early_stopping.enabled and state["early_no_improve"] >= cfg.training.early_stopping.patience:
        return True
    return False


def train(
    cfg: Config,
    *,
    run_dir: Path,                          # the dir for running the training, to save checkpoints, logs, etc.
    model,                                  # the model to be trained
    tokenizer,                              # the tokenizer used for the model and data
    train_loader: DataLoader,               # the dataloader for training data
    val_loader: DataLoader,                 # the dataloader for validation data
    device: torch.device,                   # the device to run the training on
    autocast_dtype: torch.dtype | None,     # the dtype for autocasting, e.g., torch.float16
    precision: str,                         # the precision mode, e.g., "amp", "bf16", "fp32"
) -> dict[str, Any]:
    
    model.to(device) 
    optimizer = build_optimizer(cfg, model)  # build optimizer from config, usually AdamW

    steps_per_epoch, total_steps, total_epochs = _compute_training_steps(cfg, len(train_loader))

    scheduler = build_scheduler(cfg, optimizer, total_steps)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(precision == "fp16" and torch.device.type == "cuda"),
    )  # gradient scaler for mixed precision training, temporarily scales up gradients to avoid underflow

    # compute class weights if needed
    class_weights = _class_weights(cfg, train_loader.dataset, device) 
    # build loss function with class weights if needed
    loss_fn = _build_loss(cfg, class_weights)  # -- from torch.nn.CrossEntropyLoss

    metrics_path = run_dir / cfg.logging.metrics_filename
    state = {
        "best_metric": None,
        "best_epoch": 0,
        "best_step": 0,
        "early_best_metric": None,
        "early_no_improve": 0,
    }

    global_step = 0
    for epoch in range(1, total_epochs + 1):
        model.train() # -- set model to training mode
        running_loss = 0.0
        step_in_epoch = 0
        should_stop = False

        for batch in train_loader:
            step_in_epoch += 1

            ## ---- The initial training step: generating gradients for micro-batches ---- ##
            try:
                batch = {k: v.to(device) for k, v in batch.items()} # everything needs to be on the right device

                labels_tensor = batch["labels"] if "labels" in batch else batch["label"]
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    out = model(**batch)
                    logits = out.logits         # -- the last layer output before softmax, shape (batch_size, num_labels)
                    loss = loss_fn(logits, labels_tensor) # -- compute loss, scalar loss; defines d(loss)/d(logits) etc
                    loss = loss / cfg.training.grad_accum_steps # -- normalize loss for gradient accumulation
                scaler.scale(loss).backward()  # -- accumulate them in param.grad，通过param.grad存储梯度到每个参数，scaler.scale用于混合精度训练时放大梯度以防止下溢

            ## ---- Monitor for out-of-memory errors ---- ##
            except RuntimeError as e:
                if _maybe_oom(e):
                    raise RuntimeError(
                        "Out of memory during training step. Reduce batch size or max_length."
                    ) from e
                raise

            ## ---- Gradient accumulation and optimization step, core learning loop ---- ##
            if step_in_epoch % cfg.training.grad_accum_steps == 0:
                scaler.unscale_(optimizer)  # unscale 到真实的梯度值
                if cfg.training.max_grad_norm is not None: # 梯度方向不变，但幅度被裁剪到max_grad_norm以内，控制学习率防止发散
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                scaler.step(optimizer) # 更新参数
                scaler.update() # 更新scaler的缩放因子
                optimizer.zero_grad(set_to_none=True) # 清空梯度，为下一次累积做准备（param.grad是加和累积，需要清零）
                scheduler.step() # 更新学习率调度器
                global_step += 1 # 训练的全局步数

                running_loss += loss.item() * cfg.training.grad_accum_steps # accumulate actual loss value
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

                ## -- periodic evaluation during training -- ##
                if cfg.training.eval_every_steps is not None and global_step % cfg.training.eval_every_steps == 0:
                    eval_metrics = _eval_epoch(model, val_loader, device, cfg)
                    eval_metrics.update({"split": "val", "epoch": epoch, "step": global_step})
                    _log_jsonl(metrics_path, eval_metrics)  ## To monitor overfitting or other issues during training
                    if _handle_eval(
                        cfg,
                        eval_metrics=eval_metrics,
                        run_dir=run_dir,
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=global_step,
                        state=state,
                        save_last=False,
                    ):
                        should_stop = True
                        break

            if cfg.training.max_steps is not None and global_step >= cfg.training.max_steps:
                break
            ## -- finish one epoch --
            
        if should_stop:
            break

        ## ---- End of epoch evaluation and checkpointing ---- ##
        eval_metrics = _eval_epoch(model, val_loader, device, cfg)
        eval_metrics.update({"split": "val", "epoch": epoch, "step": global_step})
        _log_jsonl(metrics_path, eval_metrics)
        if _handle_eval(
            cfg,
            eval_metrics=eval_metrics,
            run_dir=run_dir,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=global_step,
            state=state,
            save_last=True,
        ):
            break

        if cfg.training.max_steps is not None and global_step >= cfg.training.max_steps:
            break

    summary = {
        "best_metric": state["best_metric"],
        "best_epoch": state["best_epoch"],
        "best_step": state["best_step"],
        "total_steps": global_step,
    }
    (run_dir / cfg.logging.summary_filename).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
