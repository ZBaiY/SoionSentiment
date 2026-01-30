from __future__ import annotations

import gc
import json
import math
import resource
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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


def _safe_log_jsonl(path: Path, record: dict[str, Any]) -> None:
    try:
        _log_jsonl(path, record)
    except Exception:
        return



def _rss_current_gb() -> float | None:
    """Return current process RSS in GiB if available.

    Prefer psutil (true current RSS). Fallback to ru_maxrss (peak) if psutil is unavailable.
    """
    try:
        import os
        import psutil  # type: ignore

        rss_bytes = psutil.Process(os.getpid()).memory_info().rss
        return float(rss_bytes) / (1024.0**3)
    except Exception:
        # Fallback: return peak RSS if that's all we can access.
        return _rss_peak_gb()


def _rss_peak_gb() -> float | None:
    """Return peak RSS (ru_maxrss) in GiB if available.

    Note: on macOS ru_maxrss is bytes; on Linux it is kilobytes.
    """
    try:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            rss_bytes = float(r)
        else:
            rss_bytes = float(r) * 1024.0
        return rss_bytes / (1024.0**3)
    except Exception:
        return None


def _mps_mem_gb() -> tuple[float | None, float | None]:
    """Return (current_allocated_gb, driver_allocated_gb) for MPS if available."""
    try:
        if not torch.backends.mps.is_available():
            return None, None
        cur = torch.mps.current_allocated_memory() / (1024.0**3)
        drv = torch.mps.driver_allocated_memory() / (1024.0**3)
        return float(cur), float(drv)
    except Exception:
        return None, None


def _vms_current_gb() -> float | None:
    try:
        import os
        import psutil  # type: ignore

        vms_bytes = psutil.Process(os.getpid()).memory_info().vms
        return float(vms_bytes) / (1024.0**3)
    except Exception:
        return None


def _uss_current_gb() -> float | None:
    try:
        import os
        import psutil  # type: ignore

        uss_bytes = psutil.Process(os.getpid()).memory_full_info().uss
        return float(uss_bytes) / (1024.0**3)
    except Exception:
        return None


def _vm_stat_gb() -> dict[str, float]:
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False)
    except Exception:
        return {}
    if out.returncode != 0:
        return {}
    lines = out.stdout.splitlines()
    page_size = 4096
    if lines:
        header = lines[0]
        if "page size of" in header and "bytes" in header:
            try:
                page_size = int(header.split("page size of")[1].split("bytes")[0].strip())
            except Exception:
                page_size = 4096
    out_map: dict[str, float] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        val = val.strip().rstrip(".")
        try:
            pages = int(val)
        except ValueError:
            continue
        out_map[key] = pages * page_size / (1024.0**3)
    return out_map


def _mem_snapshot(device: torch.device, *, step: int | None = None, vm_stat_every_steps: int | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"gc_objects": len(gc.get_objects())}
    rss = _rss_current_gb()
    if rss is not None:
        snapshot["rss_gb"] = rss
    vms = _vms_current_gb()
    if vms is not None:
        snapshot["vms_gb"] = vms
    uss = _uss_current_gb()
    if uss is not None:
        snapshot["uss_gb"] = uss
    if device.type == "mps":
        mps_cur, mps_drv = _mps_mem_gb()
        if mps_cur is not None:
            snapshot["mps_current_gb"] = mps_cur
        if mps_drv is not None:
            snapshot["mps_driver_gb"] = mps_drv
    if vm_stat_every_steps is not None and step is not None and step % vm_stat_every_steps == 0:
        vm = _vm_stat_gb()
        if vm:
            for k in ("pages_free", "pages_active", "pages_speculative", "pages_wired_down", "pages_compressed"):
                if k in vm:
                    snapshot[k] = vm[k]
    return snapshot


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
    if cfg.train.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        # 前向传播时不保存中间激活（activations/计算数值，backwardation的时候链式法则会推出来），反向传播时再把需要的那段前向重算一遍，从而显著降低峰值显存/内存；代价是反向更慢（多一次重算）。
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) # enable gradient checkpointing to save memory
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            if cfg.train.use_cache is None:
                model.config.use_cache = False
            else:
                model.config.use_cache = cfg.train.use_cache
    elif cfg.train.use_cache is not None and hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = cfg.train.use_cache
    optimizer = build_optimizer(cfg, model)  # build optimizer from config, usually AdamW

    steps_per_epoch, total_steps, total_epochs = _compute_training_steps(cfg, len(train_loader))

    scheduler = build_scheduler(cfg, optimizer, total_steps)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(cfg.train.precision == "fp16" and cfg.train.grad_scaler and device.type == "cuda"),
    )  # gradient scaler for mixed precision training, temporarily scales up gradients to avoid underflow
    if cfg.train.precision == "fp16":
        autocast_dtype = torch.float16
    elif cfg.train.precision == "bf16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = None

    # compute class weights if needed
    class_weights = _class_weights(cfg, train_loader.dataset, device) 
    # build loss function with class weights if needed
    loss_fn = _build_loss(cfg, class_weights)  # -- from torch.nn.CrossEntropyLoss

    metrics_path = run_dir / cfg.logging.metrics_filename
    train_log_path = run_dir / cfg.logging.train_log_filename
    events_path = run_dir / cfg.logging.events_filename
    state = {
        "best_metric": None,
        "best_epoch": 0,
        "best_step": 0,
        "early_best_metric": None,
        "early_no_improve": 0,
    }
    ema_loss = None
    ema_alpha = 0.1
    update_every = max(1, cfg.logging.log_every_steps)
    train_log_every = max(1, cfg.logging.train_log_every_steps)
    train_tail = deque(maxlen=50)
    last_step_time = time.perf_counter()
    empty_cache_every = cfg.train.mps_empty_cache_every_steps
    gc_every = cfg.train.gc_collect_every_steps
    if device.type == "mps":
        if empty_cache_every is None:
            empty_cache_every = 1
        if gc_every is None:
            gc_every = 25

    global_step = 0
    for epoch in range(1, total_epochs + 1):
        model.train() # -- set model to training mode
        running_loss = 0.0
        step_in_epoch = 0
        should_stop = False
        try:
            total_batches = len(train_loader)
        except TypeError:
            total_batches = None
        pbar = tqdm(
            total=total_batches,
            desc=f"epoch {epoch}/{total_epochs}",
            leave=False,
            unit="batch",
        )

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
                if torch.isnan(loss).any():
                    event = {
                        "event": "nan_loss",
                        "epoch": epoch,
                        "step": global_step,
                        "step_in_epoch": step_in_epoch,
                        "mem": _mem_snapshot(device, step=global_step, vm_stat_every_steps=cfg.train.vm_stat_every_steps),
                        "tail": list(train_tail),
                    }
                    _safe_log_jsonl(events_path, event)
                    raise RuntimeError("NaN loss encountered")
                if torch.isinf(loss).any():
                    event = {
                        "event": "inf_loss",
                        "epoch": epoch,
                        "step": global_step,
                        "step_in_epoch": step_in_epoch,
                        "mem": _mem_snapshot(device, step=global_step, vm_stat_every_steps=cfg.train.vm_stat_every_steps),
                        "tail": list(train_tail),
                    }
                    _safe_log_jsonl(events_path, event)
                    raise RuntimeError("Inf loss encountered")
                if scaler.is_enabled():
                    scaler.scale(loss).backward()  # -- accumulate them in param.grad，通过param.grad存储梯度到每个参数，scaler.scale用于混合精度训练时放大梯度以防止下溢
                else:
                    loss.backward()
                del out, logits
                if cfg.train.detect_anomaly:
                    has_bad_grad = False
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        if not torch.isfinite(p.grad).all():
                            has_bad_grad = True
                            break
                    if has_bad_grad:
                        event = {
                            "event": "nan_grad",
                            "epoch": epoch,
                            "step": global_step,
                            "step_in_epoch": step_in_epoch,
                            "mem": _mem_snapshot(device, step=global_step, vm_stat_every_steps=cfg.train.vm_stat_every_steps),
                            "tail": list(train_tail),
                        }
                        _safe_log_jsonl(events_path, event)
                        raise RuntimeError("Non-finite gradients encountered")

            ## ---- Monitor for out-of-memory errors ---- ##
            except RuntimeError as e:
                if _maybe_oom(e):
                    snapshot = _mem_snapshot(device, step=global_step, vm_stat_every_steps=cfg.train.vm_stat_every_steps)
                    snapshot.update({"split": "mem_oom", "epoch": epoch, "step": global_step})
                    _safe_log_jsonl(metrics_path, snapshot)
                    _safe_log_jsonl(
                        events_path,
                        {
                            "event": "oom",
                            "epoch": epoch,
                            "step": global_step,
                            "mem": snapshot,
                            "tail": list(train_tail),
                        },
                    )
                    rss = snapshot.get("rss_gb")
                    rss_peak = _rss_peak_gb()
                    mps_cur = snapshot.get("mps_current_gb")
                    mps_drv = snapshot.get("mps_driver_gb")
                    raise RuntimeError(
                        "Out of memory during training step. Reduce batch size or max_length. "
                        f"rss_gb={rss if rss is not None else 'n/a'} "
                        f"rss_peak_gb={rss_peak if rss_peak is not None else 'n/a'} "
                        f"mps_cur_gb={mps_cur if mps_cur is not None else 'n/a'} "
                        f"mps_drv_gb={mps_drv if mps_drv is not None else 'n/a'}"
                    ) from e
                raise

            ## ---- Gradient accumulation and optimization step, core learning loop ---- ##
            if step_in_epoch % cfg.training.grad_accum_steps == 0:
                grad_norm = None
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)  # unscale 到真实的梯度值
                    if cfg.logging.log_grad_norm:
                        if cfg.training.max_grad_norm is not None: # 梯度方向不变，但幅度被裁剪到max_grad_norm以内，控制学习率防止发散
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                    else:
                        if cfg.training.max_grad_norm is not None: # 梯度方向不变，但幅度被裁剪到max_grad_norm以内，控制学习率防止发散
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                    scaler.step(optimizer) # 更新参数
                    scaler.update() # 更新scaler的缩放因子
                    optimizer.zero_grad(set_to_none=True) # 清空梯度，为下一次累积做准备（param.grad是加和累积，需要清零）
                else:
                    if cfg.logging.log_grad_norm:
                        if cfg.training.max_grad_norm is not None: # 梯度方向不变，但幅度被裁剪到max_grad_norm以内，控制学习率防止发散
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                    else:
                        if cfg.training.max_grad_norm is not None: # 梯度方向不变，但幅度被裁剪到max_grad_norm以内，控制学习率防止发散
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                scheduler.step() # 更新学习率调度器
                global_step += 1 # 训练的全局步数
                if gc_every is not None and global_step % gc_every == 0:
                    gc.collect()
                if empty_cache_every is not None and global_step % empty_cache_every == 0:
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
                if device.type == "mps":
                    mps_cur, mps_drv = _mps_mem_gb()
                    if mps_drv is not None and mps_drv > 16.0:
                        gc.collect()
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
                        _log_jsonl(
                            metrics_path,
                            {"split": "mem_event", "epoch": epoch, "step": global_step, "event": "mps_driver_runaway"},
                        )

                running_loss += loss.item() * cfg.training.grad_accum_steps # accumulate actual loss value
                current_loss = loss.item() * cfg.training.grad_accum_steps
                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = ema_alpha * current_loss + (1.0 - ema_alpha) * ema_loss
                if global_step % train_log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    seq_len = None
                    if "input_ids" in batch:
                        seq_len = int(batch["input_ids"].shape[-1])
                    batch_size = int(next(iter(batch.values())).shape[0]) if batch else 0
                    tokens = batch_size * (seq_len if seq_len is not None else 0)
                    now = time.perf_counter()
                    step_time = now - last_step_time
                    last_step_time = now
                    rec = {
                        "split": "train",
                        "epoch": epoch,
                        "step": global_step,
                        "step_in_epoch": step_in_epoch,
                        "loss": float(ema_loss if ema_loss is not None else current_loss),
                        "lr": lr,
                        "grad_norm": float(grad_norm) if (cfg.logging.log_grad_norm and grad_norm is not None) else None,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "tokens_per_step": tokens,
                    }
                    if cfg.logging.log_param_norm:
                        total = 0.0
                        for p in model.parameters():
                            total += float(torch.norm(p.detach(), p=2).item()) ** 2
                        rec["param_norm"] = total ** 0.5
                    if cfg.logging.log_throughput:
                        rec["step_time_s"] = step_time
                        rec["steps_per_s"] = (1.0 / step_time) if step_time > 0 else None
                        rec["tokens_per_s"] = (tokens / step_time) if step_time > 0 else None
                    mem = _mem_snapshot(device, step=global_step, vm_stat_every_steps=cfg.train.vm_stat_every_steps)
                    for k in ("mps_current_gb", "mps_driver_gb", "rss_gb", "uss_gb"):
                        if k in mem:
                            rec[k] = mem[k]
                    _safe_log_jsonl(train_log_path, rec)
                    train_tail.append(rec)
                del loss, labels_tensor, batch
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
                if global_step % update_every == 0:
                    mem = _mem_snapshot(device, step=global_step, vm_stat_every_steps=cfg.train.vm_stat_every_steps)
                    mem.update({"split": "mem", "epoch": epoch, "step": global_step})
                    _log_jsonl(metrics_path, mem)

                ## -- periodic evaluation during training -- ##
                if cfg.training.eval_every_steps is not None and global_step % cfg.training.eval_every_steps == 0:
                    pbar.close()
                    eval_metrics = _eval_epoch(model, val_loader, device, cfg)
                    eval_metrics.update({"split": "val", "epoch": epoch, "step": global_step})
                    _log_jsonl(metrics_path, eval_metrics)  ## To monitor overfitting or other issues during training
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
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
                    pbar = tqdm(
                        total=total_batches,
                        initial=step_in_epoch,
                        desc=f"epoch {epoch}/{total_epochs}",
                        leave=False,
                        unit="batch",
                    )

            if cfg.training.max_steps is not None and global_step >= cfg.training.max_steps:
                break
            
            ## -- finish one epoch --
            if step_in_epoch % update_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                postfix: dict[str, Any] = {
                    "gstep": global_step,
                    "lr": f"{lr:.3e}",
                    "loss": f"{ema_loss:.4f}" if ema_loss is not None else "n/a",
                }
                mem = _mem_snapshot(device, step=global_step, vm_stat_every_steps=cfg.train.vm_stat_every_steps)
                if "rss_gb" in mem:
                    postfix["rss_gb"] = f"{mem['rss_gb']:.2f}"
                if "vms_gb" in mem:
                    postfix["vms_gb"] = f"{mem['vms_gb']:.2f}"
                if "uss_gb" in mem:
                    postfix["uss_gb"] = f"{mem['uss_gb']:.2f}"
                if "mps_current_gb" in mem:
                    postfix["mps"] = f"{mem['mps_current_gb']:.2f}/{mem.get('mps_driver_gb', 0):.2f}"
                pbar.set_postfix(postfix, refresh=True)
            pbar.update(1)

        pbar.close()
        if should_stop:
            break

        ## ---- End of epoch evaluation and checkpointing ---- ##
        eval_metrics = _eval_epoch(model, val_loader, device, cfg)
        eval_metrics.update({"split": "val", "epoch": epoch, "step": global_step})
        _log_jsonl(metrics_path, eval_metrics)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
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
