from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from soion_sentiment.config import Config


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_data_manifest(cfg: Config) -> dict[str, Any]:
    repo_root = _repo_root_from_here()
    hf_root = Path(cfg.data.hf_dataset_root) if cfg.data.hf_dataset_root else repo_root / "data" / "phrasebank_local_v1"
    proc_root = (
        Path(cfg.data.processed_root) if cfg.data.processed_root else repo_root / "data" / "processed" / "phrasebank"
    )
    roots = [hf_root, proc_root]
    files: list[dict[str, Any]] = []

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            size = path.stat().st_size
            sha = None
            note = None
            if cfg.data.hash_files:
                if cfg.data.hash_max_bytes is None or size <= cfg.data.hash_max_bytes:
                    sha = _sha256(path)
                else:
                    note = "skipped_hash_too_large"
            files.append(
                {
                    "path": str(path.relative_to(repo_root)),
                    "size": size,
                    "sha256": sha,
                    "note": note,
                }
            )

    return {
        "dataset": cfg.data.name,
        "roots": [str(p.relative_to(repo_root)) for p in roots],
        "files": files,
    }


def write_data_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
