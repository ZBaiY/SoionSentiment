from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from soion_sentiment.analyze_train.plotting import plot_confusion_matrix


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing eval jsonl: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    base_rows = []
    metrics_rows = []
    for rec in records:
        base = {k: v for k, v in rec.items() if k != "metrics"}
        base_rows.append(base)
        metrics_rows.append(rec.get("metrics", {}))
    base_df = pd.DataFrame(base_rows)
    metrics_df = pd.json_normalize(metrics_rows)
    df = pd.concat([base_df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _format_float(val: Any) -> str:
    try:
        if val is None:
            return "-"
        return f"{float(val):.4f}"
    except Exception:
        return "-"


def _write_index_md(records: list[dict[str, Any]], out_path: Path) -> None:
    run_id = records[0].get("run_id", "") if records else ""
    lines: list[str] = []
    lines.append(f"# Eval Suite: {run_id}")
    lines.append("")
    lines.append("| eval_name | dataset_ref | split | n_samples | loss | acc | macro_f1 | ece |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for rec in records:
        metrics = rec.get("metrics", {}) or {}
        loss = _format_float(metrics.get("loss"))
        acc = _format_float(metrics.get("acc"))
        macro_f1 = _format_float(metrics.get("macro_f1"))
        ece = _format_float(metrics.get("ece"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rec.get("eval_name", "")),
                    str(rec.get("dataset_ref", "")),
                    str(rec.get("split", "")),
                    str(rec.get("n_samples", 0)),
                    loss,
                    acc,
                    macro_f1,
                    ece,
                ]
            )
            + " |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_confusion_plots(
    records: list[dict[str, Any]],
    out_dir: Path,
    *,
    labels: list[str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        metrics = rec.get("metrics", {}) or {}
        cm = metrics.get("confusion_matrix")
        if cm is None:
            continue
        eval_name = rec.get("eval_name", "")
        title = f"{eval_name}"
        outpath = out_dir / f"confusion_matrix_{eval_name}.png"
        plot_confusion_matrix(cm, outpath, title=title, labels=labels)


def analyze_eval_suite(
    jsonl_path: Path,
    out_dir: Path,
    *,
    out_csv: Path,
    write_index_md: bool,
    write_plots: bool,
    write_confusion_matrices: bool,
) -> list[dict[str, Any]]:
    records = _read_jsonl(jsonl_path)
    _write_csv(records, out_csv)

    if write_index_md:
        _write_index_md(records, out_dir / "index.md")
    if write_plots and write_confusion_matrices:
        labels = None
        if records:
            id2label = records[0].get("id2label")
            if isinstance(id2label, dict):
                try:
                    keys = sorted(id2label.keys(), key=lambda v: int(v))
                except Exception:
                    keys = sorted(id2label.keys())
                labels = [id2label[k] for k in keys]
        _write_confusion_plots(records, out_dir / "plots", labels=labels)

    return records
