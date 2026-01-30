from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def file_metadata(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "size_bytes": int(st.st_size),
        "mtime": float(st.st_mtime),
    }


def _type_name(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def _update_schema(state: dict[str, Any], obj: dict[str, Any]) -> None:
    state["n_rows"] += 1
    for k, v in obj.items():
        state["key_counts"][k] = state["key_counts"].get(k, 0) + 1
        tname = _type_name(v)
        state["type_counts"].setdefault(k, {})
        state["type_counts"][k][tname] = state["type_counts"][k].get(tname, 0) + 1
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if k not in state["numeric_min"]:
                state["numeric_min"][k] = v
                state["numeric_max"][k] = v
            else:
                state["numeric_min"][k] = min(state["numeric_min"][k], v)
                state["numeric_max"][k] = max(state["numeric_max"][k], v)
        if len(state["examples"].setdefault(k, [])) < 3:
            state["examples"][k].append(v)


def _finalize_schema(state: dict[str, Any]) -> dict[str, Any]:
    n_rows = max(state["n_rows"], 1)
    keys = sorted(state["key_counts"].keys())
    return {
        "n_rows": state["n_rows"],
        "keys": keys,
        "coverage": {k: state["key_counts"][k] / n_rows for k in keys},
        "types": {k: state["type_counts"][k] for k in keys},
        "numeric_min": {k: state["numeric_min"][k] for k in state["numeric_min"]},
        "numeric_max": {k: state["numeric_max"][k] for k in state["numeric_max"]},
        "examples": {k: state["examples"][k] for k in keys},
    }


def _schema_state() -> dict[str, Any]:
    return {
        "n_rows": 0,
        "key_counts": {},
        "type_counts": {},
        "numeric_min": {},
        "numeric_max": {},
        "examples": {},
    }


def infer_schema_jsonl(path: Path, max_lines: int | None = None) -> dict[str, Any]:
    state = _schema_state()
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            _update_schema(state, obj)
    return _finalize_schema(state)


def infer_schema_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    state = _schema_state()
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                _update_schema(state, obj)
            else:
                _update_schema(state, {"value": obj})
    elif isinstance(data, dict):
        _update_schema(state, data)
    else:
        _update_schema(state, {"value": data})

    return _finalize_schema(state)


def infer_schema_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    state = _schema_state()
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                _update_schema(state, obj)
            else:
                _update_schema(state, {"value": obj})
    elif isinstance(data, dict):
        _update_schema(state, data)
    else:
        _update_schema(state, {"value": data})

    return _finalize_schema(state)


def infer_schema_text(path: Path) -> dict[str, Any]:
    state = _schema_state()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            _update_schema(state, {"line": line.rstrip("\n")})
    return _finalize_schema(state)
