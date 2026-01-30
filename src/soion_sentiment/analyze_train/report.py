from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_index_md(
    outdir: Path,
    run_id: str,
    paths: dict[str, Path],
    summary: dict[str, Any],
    warnings: list[str],
    cfg: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append(f"# Training Analysis: {run_id}")
    lines.append("")
    lines.append("## Core paths")
    lines.append("")
    lines.append(f"- train.jsonl: `{paths['train_jsonl']}`")
    lines.append(f"- metrics.jsonl: `{paths['metrics_jsonl']}`")
    lines.append("")

    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    lines.append("```")
    lines.append("")

    outpath = outdir / "index.md"
    outpath.write_text("\n".join(lines), encoding="utf-8")
