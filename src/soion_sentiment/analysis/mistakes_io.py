from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any


def write_mistakes_jsonl(
    mistakes: list[dict[str, Any]],
    path: Path,
    *,
    max_n: int | None,
    seed: int,
    run_id: str,
    split: str,
) -> None:
    if max_n is not None:
        if max_n <= 0:
            raise ValueError("max_n must be > 0 when set")
        rng = random.Random(seed)
        n = min(max_n, len(mistakes))
        if n < len(mistakes):
            idxs = rng.sample(range(len(mistakes)), n)
            mistakes = [mistakes[i] for i in idxs]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in mistakes:
            rec = dict(rec)
            rec.setdefault("run_id", run_id)
            rec.setdefault("split", split)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_mistakes_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"mistakes jsonl not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
