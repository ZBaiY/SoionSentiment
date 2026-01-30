from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_run_paths(run_id: str, runs_root: Path, cfg: dict[str, Any]) -> dict[str, Path]:
    run_dir = runs_root / run_id
    core = cfg.get("core", {})

    train_name = core.get("train_log", "train.jsonl")
    metrics_name = core.get("metrics_log", "metrics.jsonl")

    paths = {
        "run_dir": run_dir,
        "analyze_dir": run_dir / "analyze",
        "train_jsonl": run_dir / train_name,
        "metrics_jsonl": run_dir / metrics_name,
        "resolved_config_yaml": run_dir / "resolved_config.yaml",
        "summary_json": run_dir / "summary.json",
        "data_manifest_json": run_dir / "data_manifest.json",
        "env_txt": run_dir / "env.txt",
        "best_metrics_json": run_dir / "best" / "metrics.json",
        "last_metrics_json": run_dir / "last" / "metrics.json",
    }
    return paths
