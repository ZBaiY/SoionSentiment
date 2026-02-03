from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from soion_sentiment.config import Config, _deep_merge, _from_dict, _set_by_path, load_config


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return raw


def _load_data_ref(data_ref: str, repo_root: Path) -> dict[str, Any]:
    path = repo_root / "configs" / "data" / f"{data_ref}.yaml"
    return _load_yaml(path)


def load_run_config(
    run_dir: Path,
    *,
    data_ref: str | None,
    overrides: dict[str, Any] | None,
) -> Config:
    resolved_path = run_dir / "resolved_config.yaml"
    if not resolved_path.exists():
        raise FileNotFoundError(f"resolved_config.yaml missing: {resolved_path}")

    base_cfg = load_config(resolved_path)
    merged = base_cfg.as_dict()
    if data_ref:
        repo_root = _repo_root_from_here()
        data_cfg = _load_data_ref(data_ref, repo_root)
        merged = _deep_merge(merged, data_cfg)
    if overrides:
        for key, value in overrides.items():
            _set_by_path(merged, key, value)
    cfg = _from_dict(Config, merged)
    cfg.validate()
    return cfg
