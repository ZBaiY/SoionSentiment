from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from soion_sentiment.analyze_train.runner import run_analysis
import soion_sentiment.training.loop as train_loop
from soion_sentiment.config import load_config


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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def test_missing_train_jsonl_raises(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "run_001"
    run_dir.mkdir(parents=True)
    _write_jsonl(run_dir / "metrics.jsonl", [{"step": 1, "macro_f1": 0.5}])

    with pytest.raises(RuntimeError):
        run_analysis("run_001", runs_root=runs_root)


def test_missing_metrics_jsonl_raises(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "run_001"
    run_dir.mkdir(parents=True)
    _write_jsonl(run_dir / "train.jsonl", [{"step": 1, "loss": 1.0}])

    with pytest.raises(RuntimeError):
        run_analysis("run_001", runs_root=runs_root)


def test_invalid_jsonl_raises(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "run_001"
    run_dir.mkdir(parents=True)

    (run_dir / "train.jsonl").write_text("{bad json}\n", encoding="utf-8")
    _write_jsonl(run_dir / "metrics.jsonl", [{"step": 1, "macro_f1": 0.5}])

    with pytest.raises(ValueError):
        run_analysis("run_001", runs_root=runs_root)


def test_minimal_valid_logs_produce_outputs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "run_001"
    run_dir.mkdir(parents=True)

    _write_jsonl(run_dir / "train.jsonl", [{"step": 1, "loss": 1.0}])
    _write_jsonl(run_dir / "metrics.jsonl", [{"step": 1, "macro_f1": 0.5}])

    run_analysis("run_001", runs_root=runs_root)

    analyze_dir = run_dir / "analyze"
    assert (analyze_dir / "summary.json").exists()
    assert (analyze_dir / "index.md").exists()


def test_eval_events_plot(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "run_001"
    run_dir.mkdir(parents=True)

    train_rows = [
        {"step": 1, "loss": 1.0, "epoch": 1, "step_in_epoch": 1},
        {
            "event": "eval",
            "eval_kind": "log",
            "ts_ms": 123,
            "epoch": 1,
            "step": 1,
            "step_in_epoch": 1,
            "split": "eval",
            "metrics": {"macro_f1": 0.5, "acc": 0.6, "eval_loss": 0.9},
            "did_improve": None,
            "should_stop": None,
            "best_metric": None,
            "best_step": None,
            "early_no_improve": None,
        },
        {
            "event": "eval",
            "eval_kind": "stop",
            "ts_ms": 124,
            "epoch": 1,
            "step": 2,
            "step_in_epoch": 2,
            "split": "eval",
            "metrics": {"macro_f1": 0.6, "acc": 0.7, "eval_loss": 0.8},
            "did_improve": True,
            "should_stop": False,
            "best_metric": 0.6,
            "best_step": 2,
            "early_no_improve": 0,
        },
    ]
    _write_jsonl(run_dir / "train.jsonl", train_rows)
    _write_jsonl(run_dir / "metrics.jsonl", [{"step": 1, "macro_f1": 0.5, "acc": 0.6, "split": "eval_stop"}])

    cfg_path = tmp_path / "analyze.yaml"
    cfg_path.write_text(
        """---\npaths:\n  runs_root: "runs"\n  run_id: "run_001"\n\ncore:\n  train_log: "train.jsonl"\n  metrics_log: "metrics.jsonl"\n\nstrict:\n  max_bad_lines: 0\n  require_step_axis: true\n\nanalysis:\n  enabled: true\n  rolling_window: 1\n  x_axis_preference: [\"step\"]\n  try_construct_step_from: [\"epoch\", \"step_in_epoch\"]\n  best_metric_preference: [\"macro_f1\", \"acc\", \"loss\"]\n\noutputs:\n  write_schemas: false\n  write_tables: false\n  write_plots: true\n  write_index_md: true\n\nplots:\n  public_only: true\n  make_pretty: false\n  include_debug_plots: false\n  loss_train_vs_eval: true\n  macro_f1_eval_log_vs_stop: true\n  acc_eval_log_vs_stop: false\n  combine_eval_log_and_stop: false\n  per_class_f1_plots: false\n  legacy_metrics_jsonl_plots: false\n\nextras:\n  read_best_last_metrics_json: false\n  read_resolved_config_yaml: false\n  read_summary_json: false\n  read_data_manifest_json: false\n  read_env_txt: false\n""",
        encoding="utf-8",
    )

    run_analysis(None, config_path=cfg_path, runs_root=runs_root)
    analyze_dir = run_dir / "analyze"
    assert (analyze_dir / "summary.json").exists()
    assert (analyze_dir / "index.md").exists()
    assert (analyze_dir / "plots" / "loss__train_vs_eval_log.png").exists()
    assert (analyze_dir / "plots" / "loss__eval_stop.png").exists()
    assert (analyze_dir / "plots" / "macro_f1__eval_log.png").exists()
    assert (analyze_dir / "plots" / "macro_f1__eval_stop.png").exists()


def test_logging_and_analyzer_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fixed_eval(*args, **kwargs):
        return {"acc": 0.0, "macro_f1": 0.0, "eval_loss": 0.0, "eval_samples": 1}

    monkeypatch.setattr(train_loop, "_eval_epoch", _fixed_eval)

    runs_root = tmp_path / "runs"
    run_id = "run_001"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(
        "configs/base.yaml",
        overrides={
            "training.epochs": 1,
            "training.batch_size": 1,
            "training.eval_batch_size": 1,
            "training.grad_accum_steps": 1,
            "training.eval_every_steps": None,
            "training.eval_log_every_steps": 1,
            "training.eval_stop_every_steps": 2,
            "training.eval_stop_every_epochs": None,
            "training.imbalance_strategy": "none",
            "training.early_stopping.enabled": True,
            "training.early_stopping.patience": 5,
            "logging.run_dir": str(runs_root),
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
        },
    )

    model = _DummyModel(seq_len=4, num_labels=len(cfg.model.labels))
    tokenizer = _DummyTokenizer()
    train_ds = _DummyDataset(n=4, seq_len=4, num_labels=len(cfg.model.labels))
    val_ds = _DummyDataset(n=2, seq_len=4, num_labels=len(cfg.model.labels))
    train_loader = DataLoader(train_ds, batch_size=1)
    val_loader = DataLoader(val_ds, batch_size=1)

    train_loop.train(
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

    with (run_dir / "train.jsonl").open("r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    eval_events = [e for e in events if e.get("event") == "eval"]
    kinds = {e.get("eval_kind") for e in eval_events}
    assert "log" in kinds
    assert "stop" in kinds

    cfg_path = tmp_path / "analyze.yaml"
    cfg_path.write_text(
        f"\"\"\"---\npaths:\n  runs_root: \"{runs_root}\"\n  run_id: \"{run_id}\"\n\ncore:\n  train_log: \"train.jsonl\"\n  metrics_log: \"metrics.jsonl\"\n\nstrict:\n  max_bad_lines: 0\n  require_step_axis: true\n\nanalysis:\n  enabled: true\n  rolling_window: 1\n  x_axis_preference: [\"step\"]\n  try_construct_step_from: [\"epoch\", \"step_in_epoch\"]\n  best_metric_preference: [\"macro_f1\", \"acc\", \"loss\"]\n\noutputs:\n  write_schemas: false\n  write_tables: false\n  write_plots: true\n  write_index_md: true\n\nplots:\n  public_only: true\n  make_pretty: false\n  include_debug_plots: false\n  loss_train_vs_eval: true\n  macro_f1_eval_log_vs_stop: true\n  acc_eval_log_vs_stop: false\n  combine_eval_log_and_stop: false\n  per_class_f1_plots: false\n  legacy_metrics_jsonl_plots: false\n\nextras:\n  read_best_last_metrics_json: false\n  read_resolved_config_yaml: false\n  read_summary_json: false\n  read_data_manifest_json: false\n  read_env_txt: false\n\"\"\",\n        encoding=\"utf-8\",\n    ")

    run_analysis(None, config_path=cfg_path, runs_root=runs_root)
    analyze_dir = run_dir / "analyze"
    assert (analyze_dir / "plots" / "loss__train_vs_eval_log.png").exists()
    assert (analyze_dir / "plots" / "loss__eval_stop.png").exists()
    assert (analyze_dir / "plots" / "macro_f1__eval_log.png").exists()
    assert (analyze_dir / "plots" / "macro_f1__eval_stop.png").exists()
