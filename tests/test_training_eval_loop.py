from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from soion_sentiment.config import load_config
import soion_sentiment.training.loop as train_loop


class _DummyDataset(Dataset):
    def __init__(self, n: int, seq_len: int, num_labels: int) -> None:
        self.n = n
        self.seq_len = seq_len
        self.num_labels = num_labels

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return {
            "input_ids": torch.randn(self.seq_len),
            "labels": torch.tensor(idx % self.num_labels, dtype=torch.long),
        }


class _DummyModel(torch.nn.Module):
    def __init__(self, seq_len: int, num_labels: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(seq_len, num_labels)

    def forward(self, input_ids, labels=None, **kwargs):
        logits = self.proj(input_ids.float())
        return type("Out", (), {"logits": logits})


class _DummyTokenizer:
    def save_pretrained(self, path: str) -> None:
        return None


def _run_tiny_train(tmp_path: Path, overrides: dict, *, n_train: int = 4, n_val: int = 2) -> tuple[Path, dict]:
    cfg = load_config(
        "configs/base.yaml",
        overrides={
            "training.epochs": 1,
            "training.batch_size": 1,
            "training.eval_batch_size": 1,
            "training.grad_accum_steps": 1,
            "training.eval_every_steps": None,
            "training.eval_log_every_steps": None,
            "training.eval_stop_every_steps": None,
            "training.eval_stop_every_epochs": 1,
            "training.max_steps": None,
            "training.imbalance_strategy": "none",
            "training.early_stopping.enabled": True,
            "training.early_stopping.patience": 10,
            "training.early_stopping.min_delta": 0.0,
            "logging.run_dir": str(tmp_path),
            "logging.log_every_steps": 1,
            "logging.train_log_every_steps": 1,
            "logging.metrics_filename": "metrics.jsonl",
            "logging.train_log_filename": "train.jsonl",
            "logging.events_filename": "events.jsonl",
            "logging.save_best": False,
            "logging.save_last": False,
            "train.precision": "fp32",
            "train.grad_scaler": False,
            "scheduler.name": "linear",
            "scheduler.warmup_ratio": 0.0,
            "eval.compute_confusion_matrix": False,
            **overrides,
        },
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    model = _DummyModel(seq_len=4, num_labels=len(cfg.model.labels))
    tokenizer = _DummyTokenizer()
    train_ds = _DummyDataset(n=n_train, seq_len=4, num_labels=len(cfg.model.labels))
    val_ds = _DummyDataset(n=n_val, seq_len=4, num_labels=len(cfg.model.labels))
    train_loader = DataLoader(train_ds, batch_size=1)
    val_loader = DataLoader(val_ds, batch_size=1)

    summary = train_loop.train(
        cfg,
        run_dir=run_dir,
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        autocast_dtype=None,
        precision="fp32",
    )
    return run_dir, summary


def test_logging_eval_does_not_trigger_early_stopping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fixed_eval(*args, **kwargs):
        return {"acc": 0.0, "macro_f1": 0.0, "eval_loss": 0.0, "eval_samples": 1}

    monkeypatch.setattr(train_loop, "_eval_epoch", _fixed_eval)
    run_dir, summary = _run_tiny_train(
        tmp_path,
        overrides={
            "training.eval_log_every_steps": 1,
            "training.eval_stop_every_epochs": 1,
            "training.early_stopping.patience": 2,
        },
    )
    train_log = run_dir / "train.jsonl"
    assert train_log.exists()
    with train_log.open("r", encoding="utf-8") as f:
        steps = [json.loads(line)["step"] for line in f if line.strip() and "step" in line]
    assert summary["total_steps"] >= 4
    assert max(steps) >= 4


def test_training_continues_after_eval_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fixed_eval(*args, **kwargs):
        return {"acc": 0.0, "macro_f1": 0.0, "eval_loss": 0.0, "eval_samples": 1}

    monkeypatch.setattr(train_loop, "_eval_epoch", _fixed_eval)
    _, summary = _run_tiny_train(
        tmp_path,
        overrides={
            "training.eval_log_every_steps": 1,
            "training.eval_stop_every_epochs": 1,
            "training.early_stopping.patience": 5,
        },
    )
    assert summary["total_steps"] > 1


def test_per_kind_eval_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, _ = _run_tiny_train(
        tmp_path,
        overrides={
            "training.eval_log_every_steps": 1,
            "training.eval_stop_every_steps": 1,
            "training.eval_stop_every_epochs": None,
            "training.eval_log_max_samples": 8,
            "training.eval_stop_max_samples": None,
            "data.max_eval_samples": None,
            "training.early_stopping.enabled": False,
        },
        n_val=10,
    )
    with (run_dir / "metrics.jsonl").open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    log_rows = [r for r in rows if r.get("split") == "eval_log"]
    stop_rows = [r for r in rows if r.get("split") == "eval_stop"]
    assert log_rows
    assert stop_rows
    assert log_rows[0].get("max_samples") == 8
    assert log_rows[0].get("eval_samples") == 8
    assert stop_rows[0].get("max_samples") is None
    assert stop_rows[0].get("eval_samples") == 10


def test_backward_compat_eval_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, _ = _run_tiny_train(
        tmp_path,
        overrides={
            "training.eval_log_every_steps": 1,
            "training.eval_stop_every_steps": 1,
            "training.eval_stop_every_epochs": None,
            "training.eval_log_max_samples": None,
            "training.eval_stop_max_samples": None,
            "data.max_eval_samples": 3,
            "training.early_stopping.enabled": False,
        },
        n_val=10,
    )
    with (run_dir / "metrics.jsonl").open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    log_rows = [r for r in rows if r.get("split") == "eval_log"]
    stop_rows = [r for r in rows if r.get("split") == "eval_stop"]
    assert log_rows[0].get("eval_samples") == 3
    assert stop_rows[0].get("eval_samples") == 3


def test_deterministic_log_eval_subset(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = load_config(
        "configs/base.yaml",
        overrides={
            "training.batch_size": 1,
            "training.eval_batch_size": 1,
            "training.eval_log_max_samples": 4,
            "training.eval_stop_max_samples": None,
            "data.max_eval_samples": None,
            "training.imbalance_strategy": "none",
            "eval.compute_confusion_matrix": False,
        },
    )
    model = _DummyModel(seq_len=4, num_labels=len(cfg.model.labels))
    val_ds = _DummyDataset(n=6, seq_len=4, num_labels=len(cfg.model.labels))
    val_loader = DataLoader(val_ds, batch_size=1)
    loss_fn = train_loop._build_loss(cfg, None)
    metrics_a = train_loop._eval_epoch(model, val_loader, torch.device("cpu"), cfg, loss_fn=loss_fn, max_samples=4)
    metrics_b = train_loop._eval_epoch(model, val_loader, torch.device("cpu"), cfg, loss_fn=loss_fn, max_samples=4)
    assert metrics_a["eval_samples"] == 4
    assert metrics_a == metrics_b


def test_metrics_jsonl_matches_train_eval_events(tmp_path: Path) -> None:
    run_dir, _ = _run_tiny_train(
        tmp_path,
        overrides={
            "training.eval_log_every_steps": 1,
            "training.eval_stop_every_steps": 2,
            "training.eval_stop_every_epochs": None,
            "training.eval_log_max_samples": 4,
            "training.eval_stop_max_samples": None,
            "data.max_eval_samples": None,
            "training.early_stopping.enabled": False,
        },
        n_val=6,
    )
    with (run_dir / "train.jsonl").open("r", encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f if line.strip()]
    eval_events = [r for r in train_rows if r.get("event") == "eval"]
    train_steps = {r.get("step") for r in train_rows if r.get("split") == "train"}
    eval_map = {(r.get("eval_kind"), r.get("step")): r for r in eval_events}
    for r in eval_events:
        assert r.get("step") in train_steps

    with (run_dir / "metrics.jsonl").open("r", encoding="utf-8") as f:
        metrics_rows = [json.loads(line) for line in f if line.strip()]
    eval_metrics_rows = [r for r in metrics_rows if r.get("split") in {"eval_log", "eval_stop"}]
    assert eval_metrics_rows
    for row in eval_metrics_rows:
        kind = "log" if row.get("split") == "eval_log" else "stop"
        key = (kind, row.get("step"))
        assert key in eval_map
        train_metrics = eval_map[key]["metrics"]
        for metric_key in ["acc", "macro_f1", "eval_loss"]:
            assert metric_key in train_metrics
            assert metric_key in row
            assert row[metric_key] == train_metrics[metric_key]


def test_eval_epoch_schema_no_loss_alias(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = load_config(
        "configs/base.yaml",
        overrides={
            "training.batch_size": 1,
            "training.eval_batch_size": 1,
            "training.eval_log_max_samples": 4,
            "training.eval_stop_max_samples": None,
            "data.max_eval_samples": None,
            "training.imbalance_strategy": "none",
            "eval.compute_confusion_matrix": False,
        },
    )
    model = _DummyModel(seq_len=4, num_labels=len(cfg.model.labels))
    val_ds = _DummyDataset(n=6, seq_len=4, num_labels=len(cfg.model.labels))
    val_loader = DataLoader(val_ds, batch_size=1)
    loss_fn = train_loop._build_loss(cfg, None)
    metrics = train_loop._eval_epoch(model, val_loader, torch.device("cpu"), cfg, loss_fn=loss_fn, max_samples=4)
    assert "eval_loss" in metrics
    assert "eval_samples" in metrics
    assert "loss" not in metrics


def test_stop_eval_triggers_early_stopping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fixed_eval(*args, **kwargs):
        return {"acc": 0.0, "macro_f1": 0.0, "eval_loss": 0.0, "eval_samples": 1}

    monkeypatch.setattr(train_loop, "_eval_epoch", _fixed_eval)
    run_dir, summary = _run_tiny_train(
        tmp_path,
        overrides={
            "training.eval_stop_every_steps": 1,
            "training.eval_stop_every_epochs": None,
            "training.early_stopping.patience": 2,
        },
    )
    assert summary["total_steps"] <= 3


def test_eval_events_logged_in_train_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fixed_eval(*args, **kwargs):
        return {"acc": 0.0, "macro_f1": 0.0}

    monkeypatch.setattr(train_loop, "_eval_epoch", _fixed_eval)
    run_dir, _ = _run_tiny_train(
        tmp_path,
        overrides={
            "training.eval_log_every_steps": 1,
            "training.eval_stop_every_steps": 2,
            "training.eval_stop_every_epochs": None,
            "training.early_stopping.patience": 5,
        },
    )
    train_log = run_dir / "train.jsonl"
    assert train_log.exists()
    with train_log.open("r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    eval_events = [e for e in events if e.get("event") == "eval"]
    assert eval_events
    kinds = {e.get("eval_kind") for e in eval_events}
    assert "log" in kinds
    assert "stop" in kinds
    sample = eval_events[0]
    for key in [
        "event",
        "eval_kind",
        "ts_ms",
        "epoch",
        "step",
        "step_in_epoch",
        "split",
        "metrics",
        "max_samples",
        "did_improve",
        "should_stop",
        "best_metric",
        "best_step",
        "early_no_improve",
    ]:
        assert key in sample
    assert "eval_loss" in sample["metrics"]
