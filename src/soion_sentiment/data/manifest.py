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
    notes: list[str] = []

    agree_map = {
        "sentences_50agree": "50agree",
        "sentences_66agree": "66agree",
        "sentences_75agree": "75agree",
        "sentences_allagree": "allagree",
    }
    agree_key = agree_map.get(cfg.data.agree or "")
    if agree_key is None and cfg.data.agree is not None:
        notes.append(f"unknown agree '{cfg.data.agree}', manifest may be broad")

    def _skip(path: Path) -> bool:
        name = path.name
        if name.startswith("."):
            return True
        if name.startswith("cache-") and name.endswith(".arrow"):
            return True
        if name.endswith(".lock"):
            return True
        return False

    def _add_file(path: Path) -> None:
        if _skip(path):
            return
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

    if cfg.data.name == "phrasebank":
        if agree_key is None:
            for root in roots:
                if not root.exists():
                    continue
                for path in sorted(p for p in root.rglob("*") if p.is_file()):
                    _add_file(path)
        else:
            if hf_root.exists():
                dict_path = hf_root / "dataset_dict.json"
                if dict_path.exists():
                    _add_file(dict_path)
                if cfg.data.agree:
                    agree_dir = hf_root / cfg.data.agree
                    if agree_dir.exists():
                        for path in sorted(p for p in agree_dir.iterdir() if p.is_file()):
                            if path.name == "dataset_info.json" or path.name == "state.json" or path.name.startswith("data-"):
                                _add_file(path)
            if proc_root.exists():
                seed_dir = proc_root / agree_key / str(cfg.seed)
                if seed_dir.exists():
                    for path in sorted(p for p in seed_dir.rglob("*") if p.is_file()):
                        _add_file(path)
    else:
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                _add_file(path)

    manifest: dict[str, Any] = {
        "dataset": cfg.data.name,
        "roots": [str(p.relative_to(repo_root)) for p in roots],
        "files": files,
    }
    if notes:
        manifest["notes"] = notes
    return manifest


def write_data_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
