from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunsConfig:
    run_dir: str
    checkpoint: str


@dataclass(frozen=True)
class SuiteEntry:
    name: str
    data_ref: str
    split: str


@dataclass(frozen=True)
class CalibrationConfig:
    n_bins: int = 15


@dataclass(frozen=True)
class LoggingConfig:
    out_dir: str = "runs/<run_id>/eval"
    out_csv: str = "eval_suite.csv"
    out_jsonl: str = "eval_suite.jsonl"
    write_index_md: bool = True
    write_confusion_matrices: bool = True
    write_plots: bool = True


@dataclass(frozen=True)
class EvalConfig:
    batch_size: int = 64
    max_samples: int | None = None
    metrics: list[str] | None = None
    calibration: CalibrationConfig = CalibrationConfig()
    logging: LoggingConfig = LoggingConfig()
    mistake_path: str | None = "sample_mistake.jsonl"
    mistake_max_n: int | None = None
    mistake_seed: int = 1234
    overrides: dict[str, Any] | None = None


@dataclass(frozen=True)
class EvalSuiteConfig:
    runs: RunsConfig
    suite: list[SuiteEntry]
    eval: EvalConfig

    def validate(self) -> None:
        if not self.runs.run_dir:
            raise ValueError("runs.run_dir must be set")
        if not self.runs.checkpoint:
            raise ValueError("runs.checkpoint must be set")
        if not self.suite:
            raise ValueError("suite must contain at least one entry")
        for entry in self.suite:
            if not entry.name:
                raise ValueError("suite entry name must be set")
            if not entry.data_ref:
                raise ValueError(f"suite entry {entry.name} missing data_ref")
            if not entry.split:
                raise ValueError(f"suite entry {entry.name} missing split")
        if self.eval.batch_size <= 0:
            raise ValueError("eval.batch_size must be > 0")
        if self.eval.max_samples is not None and self.eval.max_samples <= 0:
            raise ValueError("eval.max_samples must be > 0 when set")
        if self.eval.calibration.n_bins <= 1:
            raise ValueError("eval.calibration.n_bins must be > 1")
        if self.eval.overrides is not None and not isinstance(self.eval.overrides, dict):
            raise ValueError("eval.overrides must be a mapping when set")
        if self.eval.mistake_max_n is not None and self.eval.mistake_max_n <= 0:
            raise ValueError("eval.mistake_max_n must be > 0 when set")
        if self.eval.mistake_seed < 0:
            raise ValueError("eval.mistake_seed must be >= 0")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return raw


def _load_suite_entries(data: Any) -> list[SuiteEntry]:
    if not isinstance(data, list):
        raise ValueError("suite must be a list of entries")
    entries: list[SuiteEntry] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"suite[{idx}] must be a mapping")
        entries.append(
            SuiteEntry(
                name=str(item.get("name", "")),
                data_ref=str(item.get("data_ref", "")),
                split=str(item.get("split", "")),
            )
        )
    return entries


def load_eval_suite_config(path: str | Path) -> EvalSuiteConfig:
    cfg_path = Path(path)
    raw = _load_yaml(cfg_path)

    runs_raw = raw.get("runs")
    if not isinstance(runs_raw, dict):
        raise ValueError("runs must be a mapping")
    runs = RunsConfig(
        run_dir=str(runs_raw.get("run_dir", "")),
        checkpoint=str(runs_raw.get("checkpoint", "")),
    )

    suite = _load_suite_entries(raw.get("suite", []))

    eval_raw = raw.get("eval")
    if eval_raw is None:
        eval_raw = {}
    if not isinstance(eval_raw, dict):
        raise ValueError("eval must be a mapping")

    calib_raw = eval_raw.get("calibration")
    if calib_raw is None:
        calib_raw = {}
    if not isinstance(calib_raw, dict):
        raise ValueError("eval.calibration must be a mapping")
    calibration = CalibrationConfig(n_bins=int(calib_raw.get("n_bins", 15)))

    logging_raw = eval_raw.get("logging")
    if logging_raw is None:
        logging_raw = {}
    if not isinstance(logging_raw, dict):
        raise ValueError("eval.logging must be a mapping")
    logging = LoggingConfig(
        out_dir=str(logging_raw.get("out_dir", "runs/<run_id>/eval")),
        out_csv=str(logging_raw.get("out_csv", "eval_suite.csv")),
        out_jsonl=str(logging_raw.get("out_jsonl", "eval_suite.jsonl")),
        write_index_md=bool(logging_raw.get("write_index_md", True)),
        write_confusion_matrices=bool(logging_raw.get("write_confusion_matrices", True)),
        write_plots=bool(logging_raw.get("write_plots", True)),
    )

    eval_cfg = EvalConfig(
        batch_size=int(eval_raw.get("batch_size", 64)),
        max_samples=eval_raw.get("max_samples", None),
        metrics=eval_raw.get("metrics", None),
        calibration=calibration,
        logging=logging,
        mistake_path=eval_raw.get("mistake_path", "sample_mistake.jsonl"),
        mistake_max_n=eval_raw.get("mistake_max_n", None),
        mistake_seed=int(eval_raw.get("mistake_seed", 1234)),
        overrides=eval_raw.get("overrides", None),
    )

    cfg = EvalSuiteConfig(runs=runs, suite=suite, eval=eval_cfg)
    cfg.validate()
    return cfg
