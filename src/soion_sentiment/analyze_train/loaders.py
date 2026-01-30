from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_jsonl_df(
    path: Path,
    max_lines: int | None = None,
    max_bad_lines: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bad_lines = 0
    parsed_lines = 0

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            raw = line.strip()
            if not raw:
                bad_lines += 1
                if bad_lines > max_bad_lines:
                    raise ValueError(f"Too many bad JSONL lines in {path}")
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                bad_lines += 1
                if bad_lines > max_bad_lines:
                    raise ValueError(f"Too many bad JSONL lines in {path}")
                continue
            if not isinstance(obj, dict):
                bad_lines += 1
                if bad_lines > max_bad_lines:
                    raise ValueError(f"Too many bad JSONL lines in {path}")
                continue
            rows.append(obj)
            parsed_lines += 1

    stats = {"bad_lines": bad_lines, "parsed_lines": parsed_lines}
    df = pd.DataFrame(rows)
    return df, stats
