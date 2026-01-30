from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from .analysis import analyze_training, build_step_axis, split_train_events
from .config import load_config
from .loaders import load_jsonl_df
from .paths import resolve_run_paths
from .report import write_index_md
from .schema import (
    file_metadata,
    infer_schema_json,
    infer_schema_jsonl,
    infer_schema_text,
    infer_schema_yaml,
)


def _warn(msg: str, warnings_list: list[str], seen: set[str]) -> None:
    if msg in seen:
        return
    seen.add(msg)
    warnings_list.append(msg)
    warnings.warn(msg)


def _read_optional_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_optional_yaml(path: Path) -> Any:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_optional_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_schema(outdir: Path, name: str, schema: dict[str, Any], path: Path) -> None:
    outpath = outdir / f"{name}.schema.json"
    payload = {
        "path": str(path),
        "metadata": file_metadata(path),
        "schema": schema,
    }
    outpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_table(
    df: pd.DataFrame,
    outpath: Path,
    warnings_list: list[str],
    seen: set[str],
) -> None:
    try:
        df.to_parquet(outpath, index=False)
    except Exception as exc:
        _warn(f"Failed to write parquet {outpath.name}: {exc}", warnings_list, seen)
        fallback = outpath.with_suffix(".csv")
        df.to_csv(fallback, index=False)


def run_analysis(
    run_id: str | None,
    config_path: Path = Path("configs/analyze_train.yaml"),
    runs_root: Path | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    if run_id is None:
        run_id = cfg.get("paths", {}).get("run_id")
    if not run_id:
        raise RuntimeError("run_id must be provided in config paths.run_id")
    root = Path(runs_root) if runs_root is not None else Path(cfg["paths"]["runs_root"])
    paths = resolve_run_paths(run_id, root, cfg)

    if not paths["train_jsonl"].exists():
        raise RuntimeError(f"Missing required train.jsonl: {paths['train_jsonl']}")
    if not paths["metrics_jsonl"].exists():
        raise RuntimeError(f"Missing required metrics.jsonl: {paths['metrics_jsonl']}")

    analyze_dir = paths["analyze_dir"]
    analyze_dir.mkdir(parents=True, exist_ok=True)
    (analyze_dir / "plots").mkdir(parents=True, exist_ok=True)
    (analyze_dir / "tables").mkdir(parents=True, exist_ok=True)
    (analyze_dir / "schemas").mkdir(parents=True, exist_ok=True)

    max_bad_lines = int(cfg.get("strict", {}).get("max_bad_lines", 0))

    train_df, train_stats = load_jsonl_df(paths["train_jsonl"], max_bad_lines=max_bad_lines)
    metrics_df, metrics_stats = load_jsonl_df(paths["metrics_jsonl"], max_bad_lines=max_bad_lines)

    train_df, eval_events_df = split_train_events(train_df)
    train_df = build_step_axis(train_df, cfg)
    metrics_df = build_step_axis(metrics_df, cfg)
    eval_events_df = build_step_axis(eval_events_df, cfg) if not eval_events_df.empty else eval_events_df

    labels: list[str] | None = None
    resolved_config_err: str | None = None
    if cfg.get("extras", {}).get("read_resolved_config_yaml", True):
        path = paths["resolved_config_yaml"]
        if path.exists():
            try:
                resolved_cfg = _read_optional_yaml(path)
                if isinstance(resolved_cfg, dict):
                    model_cfg = resolved_cfg.get("model", {})
                    if isinstance(model_cfg, dict):
                        labels_val = model_cfg.get("labels")
                        if isinstance(labels_val, list) and labels_val:
                            labels = [str(x) for x in labels_val]
            except Exception as exc:
                resolved_config_err = str(exc)

    if cfg.get("analysis", {}).get("enabled", True):
        summary = analyze_training(train_df, metrics_df, eval_events_df, analyze_dir, cfg, labels=labels)
    else:
        summary = {
            "rows": {
                "train": int(len(train_df)),
                "metrics": int(len(metrics_df)),
                "eval_events": int(len(eval_events_df)),
            },
            "best": {},
            "health_warnings": [],
        }

    warnings_list: list[str] = []
    warnings_seen: set[str] = set()
    summary["parse_stats"] = {"train": train_stats, "metrics": metrics_stats}
    for w in summary.get("health_warnings", []):
        _warn(w, warnings_list, warnings_seen)

    extras_cfg = cfg.get("extras", {})
    extras: dict[str, Any] = {}

    if extras_cfg.get("read_resolved_config_yaml", True):
        path = paths["resolved_config_yaml"]
        if resolved_config_err:
            _warn(f"Failed to parse resolved_config.yaml: {resolved_config_err}", warnings_list, warnings_seen)
        elif path.exists():
            try:
                extras["resolved_config"] = _read_optional_yaml(path)
            except Exception as exc:
                _warn(f"Failed to parse resolved_config.yaml: {exc}", warnings_list, warnings_seen)
        else:
            _warn("Missing optional resolved_config.yaml", warnings_list, warnings_seen)

    if extras_cfg.get("read_summary_json", True):
        path = paths["summary_json"]
        if path.exists():
            try:
                extras["summary"] = _read_optional_json(path)
            except Exception as exc:
                _warn(f"Failed to parse summary.json: {exc}", warnings_list, warnings_seen)
        else:
            _warn("Missing optional summary.json", warnings_list, warnings_seen)

    if extras_cfg.get("read_data_manifest_json", True):
        path = paths["data_manifest_json"]
        if path.exists():
            try:
                extras["data_manifest"] = _read_optional_json(path)
            except Exception as exc:
                _warn(f"Failed to parse data_manifest.json: {exc}", warnings_list, warnings_seen)
        else:
            _warn("Missing optional data_manifest.json", warnings_list, warnings_seen)

    if extras_cfg.get("read_env_txt", True):
        path = paths["env_txt"]
        if path.exists():
            try:
                extras["env_txt"] = _read_optional_text(path)
            except Exception as exc:
                _warn(f"Failed to read env.txt: {exc}", warnings_list, warnings_seen)
        else:
            _warn("Missing optional env.txt", warnings_list, warnings_seen)

    if extras_cfg.get("read_best_last_metrics_json", True):
        path = paths["best_metrics_json"]
        if path.exists():
            try:
                extras["best_metrics"] = _read_optional_json(path)
            except Exception as exc:
                _warn(f"Failed to parse best/metrics.json: {exc}", warnings_list, warnings_seen)
        else:
            _warn("Missing optional best/metrics.json", warnings_list, warnings_seen)

        path = paths["last_metrics_json"]
        if path.exists():
            try:
                extras["last_metrics"] = _read_optional_json(path)
            except Exception as exc:
                _warn(f"Failed to parse last/metrics.json: {exc}", warnings_list, warnings_seen)
        else:
            _warn("Missing optional last/metrics.json", warnings_list, warnings_seen)

    summary["extras"] = extras
    summary["warnings"] = warnings_list
    if "resolved_config" in extras and isinstance(extras["resolved_config"], dict):
        training_cfg = extras["resolved_config"].get("training", {})
        if isinstance(training_cfg, dict):
            summary["log_eval_max_samples"] = training_cfg.get("eval_log_max_samples")
            summary["stop_eval_max_samples"] = training_cfg.get("eval_stop_max_samples")

    if cfg.get("outputs", {}).get("write_tables", True):
        _write_table(train_df, analyze_dir / "tables" / "train.parquet", warnings_list, warnings_seen)
        _write_table(metrics_df, analyze_dir / "tables" / "metrics.parquet", warnings_list, warnings_seen)

    if cfg.get("outputs", {}).get("write_schemas", True):
        _write_schema(
            analyze_dir / "schemas",
            "train.jsonl",
            infer_schema_jsonl(paths["train_jsonl"], None),
            paths["train_jsonl"],
        )
        _write_schema(
            analyze_dir / "schemas",
            "metrics.jsonl",
            infer_schema_jsonl(paths["metrics_jsonl"], None),
            paths["metrics_jsonl"],
        )

        optional_schema_targets = [
            (paths["resolved_config_yaml"], infer_schema_yaml, "resolved_config.yaml", "resolved_config.yaml"),
            (paths["summary_json"], infer_schema_json, "summary.json", "summary.json"),
            (paths["data_manifest_json"], infer_schema_json, "data_manifest.json", "data_manifest.json"),
            (paths["env_txt"], infer_schema_text, "env.txt", "env.txt"),
            (paths["best_metrics_json"], infer_schema_json, "best.metrics.json", "best/metrics.json"),
            (paths["last_metrics_json"], infer_schema_json, "last.metrics.json", "last/metrics.json"),
        ]
        for path, fn, out_name, warn_name in optional_schema_targets:
            if path.exists():
                try:
                    _write_schema(analyze_dir / "schemas", out_name, fn(path), path)
                except Exception as exc:
                    _warn(f"Failed to infer schema for {warn_name}: {exc}", warnings_list, warnings_seen)
            else:
                _warn(f"Missing optional {warn_name}", warnings_list, warnings_seen)

    if cfg.get("outputs", {}).get("write_index_md", True):
        write_index_md(analyze_dir, run_id, paths, summary, warnings_list, cfg)

    summary_path = analyze_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return summary
