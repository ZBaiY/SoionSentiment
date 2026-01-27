# src/soion_sentiment/data/phrasebank.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from datasets import Dataset, DatasetDict, load_from_disk


class PhraseBankContractError(RuntimeError):
    """Raised when PhraseBank data violates the training-entry contract."""


@dataclass(frozen=True)
class PhraseBankSplitSpec:
    agree: str
    seed: int
    indices_path: Path


def _repo_root_from_here() -> Path:
    # .../src/soion_sentiment/data/phrasebank.py -> repo root
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> Any: 
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise PhraseBankContractError(f"missing required file: {path}") from e
    except json.JSONDecodeError as e:
        raise PhraseBankContractError(f"invalid json: {path}") from e


def _as_int_list(x: Any, *, name: str) -> list[int]:
    # Validate that x is a list of non-negative integers.
    if not isinstance(x, list):
        raise PhraseBankContractError(f"{name} must be a list, got {type(x).__name__}")
    out: list[int] = []
    for i, v in enumerate(x):
        if not isinstance(v, int):
            raise PhraseBankContractError(f"{name}[{i}] must be int, got {type(v).__name__}")
        if v < 0:
            raise PhraseBankContractError(f"{name}[{i}] must be >= 0, got {v}")
        out.append(v)
    return out


def _ensure_disjoint(*named_splits: tuple[str, list[int]]) -> None:
    # Ensure that the provided splits are disjoint.
    seen: dict[int, str] = {}
    for split_name, idxs in named_splits:
        for v in idxs:
            if v in seen:
                raise PhraseBankContractError(
                    f"indices overlap: {v} appears in both {seen[v]} and {split_name}"
                )
            seen[v] = split_name


def _ensure_in_range(idxs: Iterable[int], *, n_total: int, name: str) -> None:
    # Ensure that all indices are within the valid range [0, n_total).
    for v in idxs:
        if v >= n_total:
            raise PhraseBankContractError(f"{name} contains out-of-range index {v} (n_total={n_total})")


def _schema_gate(ds: Dataset) -> None:
    # Enforce that the dataset has exactly the required columns.
    cols = set(ds.column_names)
    required = {"text", "label"}
    if cols != required:
        raise PhraseBankContractError(
            f"schema gate failed: columns must be exactly {sorted(required)}, got {sorted(cols)}"
        )


def _resolve_base_dataset(obj: Any, *, agree: str) -> Dataset:
    # two possible types returned by load_from_disk:
    # (A) DatasetDict: {"sentences_66agree": Dataset, ...}
    # (B) Dataset: single agree Dataset
    if isinstance(obj, DatasetDict):
        if agree not in obj:
            raise PhraseBankContractError(
                f"agree={agree!r} not found in dataset dict keys={list(obj.keys())}"
            )
        base = obj[agree]
        if not isinstance(base, Dataset):
            raise PhraseBankContractError(f"dataset dict entry {agree!r} is not a Dataset")
        return base

    if isinstance(obj, Dataset):
        return obj

    raise PhraseBankContractError(f"load_from_disk returned unsupported type: {type(obj).__name__}")


def load_phrasebank_splits(
    *,
    seed: int,
    agree: str = "sentences_66agree",
    hf_dataset_root: str | Path | None = None,
    processed_root: str | Path | None = None,
) -> DatasetDict:
    """
    Training-entry contract for PhraseBank.

    - Loads HF dataset snapshot from disk (source-of-truth storage).
    - Replays immutable split indices from data/processed/phrasebank/{agree}/{seed}/indices.json.
    - Returns DatasetDict(train/val/test) with hard schema gate: columns exactly {"text","label"}.
    """
    if not isinstance(seed, int):
        raise PhraseBankContractError(f"seed must be int, got {type(seed).__name__}")

    repo_root = _repo_root_from_here()

    hf_root = Path(hf_dataset_root) if hf_dataset_root is not None else repo_root / "data" / "phrasebank_local_v1"
    proc_root = Path(processed_root) if processed_root is not None else repo_root / "data" / "processed" / "phrasebank"

    # Load base dataset
    loaded = load_from_disk(str(hf_root))
    base = _resolve_base_dataset(loaded, agree=agree)
    n_total = len(base)

    # Load indices truth
    indices_path = proc_root / agree / str(seed) / "indices.json"
    indices_obj = _read_json(indices_path)

    if not isinstance(indices_obj, Mapping):
        raise PhraseBankContractError(f"indices.json must be an object with keys train/val/test: {indices_path}")

    train_idx = _as_int_list(indices_obj.get("train"), name="train")
    val_idx = _as_int_list(indices_obj.get("val"), name="val")
    test_idx = _as_int_list(indices_obj.get("test"), name="test")

    # Hard gates: disjoint + in-range
    _ensure_disjoint(("train", train_idx), ("val", val_idx), ("test", test_idx))
    _ensure_in_range(train_idx, n_total=n_total, name="train")
    _ensure_in_range(val_idx, n_total=n_total, name="val")
    _ensure_in_range(test_idx, n_total=n_total, name="test")

    # Replay splits
    train_ds = base.select(train_idx)
    val_ds = base.select(val_idx)
    test_ds = base.select(test_idx)

    # Schema gate (after select, but before anything else)
    _schema_gate(train_ds)
    _schema_gate(val_ds)
    _schema_gate(test_ds)

    return DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds})