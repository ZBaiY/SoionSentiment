from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    name: str
    agree: str | None
    split_protocol: str
    hf_dataset_root: str | None
    processed_root: str | None
    text_field: str
    label_field: str
    max_length: int
    padding: str
    truncation: bool
    shuffle_train: bool
    max_train_samples: int | None
    max_eval_samples: int | None
    max_test_samples: int | None
    hash_files: bool
    hash_max_bytes: int | None


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    labels: list[str]
    dropout_override: float | None


@dataclass(frozen=True)
class TokenizerConfig:
    use_fast: bool
    padding_side: str
    truncation_side: str


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool
    metric: str
    mode: str
    patience: int
    min_delta: float


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    eval_batch_size: int
    grad_accum_steps: int
    max_grad_norm: float | None
    max_steps: int | None
    eval_every_steps: int | None
    label_smoothing: float
    imbalance_strategy: str
    early_stopping: EarlyStoppingConfig


@dataclass(frozen=True)
class OptimConfig:
    lr: float
    weight_decay: float
    betas: list[float]
    eps: float


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    warmup_ratio: float
    warmup_steps: int | None
    num_cycles: float


@dataclass(frozen=True)
class EvalConfig:
    metric: str
    mode: str
    compute_confusion_matrix: bool


@dataclass(frozen=True)
class LoggingConfig:
    run_dir: str
    run_name: str | None
    log_every_steps: int
    metrics_filename: str
    summary_filename: str
    save_best: bool
    save_last: bool


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    precision: str
    num_workers: int
    pin_memory: bool
    deterministic: bool
    hf_cache_dir: str | None
    hf_offline: bool


@dataclass(frozen=True)
class Config:
    project: str
    seed: int
    data_ref: str | None
    model_ref: str | None
    preset_ref: str | None
    data: DataConfig
    model: ModelConfig
    tokenizer: TokenizerConfig
    training: TrainingConfig
    optim: OptimConfig
    scheduler: SchedulerConfig
    eval: EvalConfig
    logging: LoggingConfig
    runtime: RuntimeConfig

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.training.epochs < 1:
            raise ValueError("training.epochs must be >= 1")
        if self.training.batch_size <= 0:
            raise ValueError("training.batch_size must be > 0")
        if self.training.eval_batch_size <= 0:
            raise ValueError("training.eval_batch_size must be > 0")
        if self.training.grad_accum_steps < 1:
            raise ValueError("training.grad_accum_steps must be >= 1")
        if self.training.eval_every_steps is not None and self.training.eval_every_steps <= 0:
            raise ValueError("training.eval_every_steps must be positive when set")
        if self.data.max_length <= 0:
            raise ValueError("data.max_length must be > 0")
        if self.optim.lr <= 0:
            raise ValueError("optim.lr must be > 0")
        if self.scheduler.warmup_steps is not None and self.scheduler.warmup_ratio not in (0.0,):
            raise ValueError("scheduler.warmup_steps and warmup_ratio cannot both be set")
        if self.scheduler.warmup_ratio < 0 or self.scheduler.warmup_ratio > 1:
            raise ValueError("scheduler.warmup_ratio must be in [0, 1]")
        if self.data.name == "phrasebank":
            if self.data.split_protocol != "precomputed":
                raise ValueError("phrasebank requires data.split_protocol=precomputed")
            if self.data.agree is None:
                raise ValueError("phrasebank requires data.agree to be set")
        if not self.model.labels or len(self.model.labels) < 2:
            raise ValueError("model.labels must contain at least 2 labels")
        if len(set(self.model.labels)) != len(self.model.labels):
            raise ValueError("model.labels must be unique")
        metric_names = {"acc", "macro_f1"} | {f"f1_{l}" for l in self.model.labels}
        if self.eval.metric not in metric_names:
            raise ValueError(f"eval.metric must be one of {sorted(metric_names)}")
        if self.training.early_stopping.metric not in metric_names:
            raise ValueError(f"training.early_stopping.metric must be one of {sorted(metric_names)}")
        if self.training.imbalance_strategy not in {"none", "reweight"}:
            raise ValueError("training.imbalance_strategy must be one of: none, reweight")
        if self.training.early_stopping.enabled:
            if self.training.early_stopping.patience < 1:
                raise ValueError("training.early_stopping.patience must be >= 1")
            if self.training.early_stopping.mode not in {"min", "max"}:
                raise ValueError("training.early_stopping.mode must be one of: min, max")
        if self.eval.mode not in {"min", "max"}:
            raise ValueError("eval.mode must be one of: min, max")
        if self.runtime.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("runtime.precision must be one of: fp32, fp16, bf16")
        if self.runtime.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("runtime.device must be one of: auto, cpu, cuda, mps")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.as_dict(), sort_keys=True)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=2)

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _from_dict(cls: type[Any], data: dict[str, Any]) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping for {cls.__name__}, got {type(data).__name__}")
    field_map = {f.name: f for f in fields(cls)}
    extra = set(data.keys()) - set(field_map.keys())
    if extra:
        raise ValueError(f"unexpected keys for {cls.__name__}: {sorted(extra)}")
    kwargs: dict[str, Any] = {}
    for name, f in field_map.items():
        if name not in data:
            raise ValueError(f"missing required key {cls.__name__}.{name}")
        val = data[name]
        if hasattr(f.type, "__dataclass_fields__"):
            kwargs[name] = _from_dict(f.type, val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def _set_by_path(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = root
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def parse_overrides(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw_val = item.split("=", 1)
        out[key] = yaml.safe_load(raw_val)
    return out


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must be a mapping: {path}")
    return data


def _registry_path(base_path: Path, group: str, name: str) -> Path:
    return base_path.parent / group / f"{name}.yaml"


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
    *,
    data_ref: str | None = None,
    model_ref: str | None = None,
    preset_ref: str | None = None,
) -> Config:
    cfg_path = Path(path)
    base = _load_yaml(cfg_path)

    if data_ref is not None:
        base["data_ref"] = data_ref
    if model_ref is not None:
        base["model_ref"] = model_ref
    if preset_ref is not None:
        base["preset_ref"] = preset_ref

    merged = dict(base)
    if base.get("data_ref"):
        data_path = _registry_path(cfg_path, "data", base["data_ref"])
        merged = _deep_merge(merged, _load_yaml(data_path))
    if base.get("model_ref"):
        model_path = _registry_path(cfg_path, "models", base["model_ref"])
        merged = _deep_merge(merged, _load_yaml(model_path))
    if base.get("preset_ref"):
        preset_path = _registry_path(cfg_path, "presets", base["preset_ref"])
        merged = _deep_merge(merged, _load_yaml(preset_path))

    if overrides:
        for k, v in overrides.items():
            _set_by_path(merged, k, v)

    cfg = _from_dict(Config, merged)
    cfg.validate()
    return cfg
