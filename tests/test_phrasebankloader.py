# tests/test_phrasebank_loader.py
from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from soion_sentiment.data.phrasebank import PhraseBankContractError, load_phrasebank_splits


def _make_hf_snapshot(root: Path, *, agree: str = "sentences_66agree", n: int = 10) -> None:
    base = Dataset.from_dict(
        {
            "text": [f"s{i}" for i in range(n)],
            "label": [i % 3 for i in range(n)],
        }
    )
    DatasetDict({agree: base}).save_to_disk(str(root))


def _write_indices(proc_root: Path, *, agree: str, seed: int, indices: dict) -> Path:
    d = proc_root / agree / str(seed)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "indices.json"
    p.write_text(json.dumps(indices), encoding="utf-8")
    return p


def test_phrasebank_loader_missing_indices_file(tmp_path: Path) -> None:
    hf_root = tmp_path / "hf_snapshot"
    proc_root = tmp_path / "processed"

    _make_hf_snapshot(hf_root)

    with pytest.raises(PhraseBankContractError, match="missing required file"):
        load_phrasebank_splits(
            seed=5768,
            agree="sentences_66agree",
            hf_dataset_root=hf_root,
            processed_root=proc_root,
        )


def test_phrasebank_loader_happy_path(tmp_path: Path) -> None:
    hf_root = tmp_path / "hf_snapshot"
    proc_root = tmp_path / "processed"
    agree = "sentences_66agree"
    seed = 5768

    _make_hf_snapshot(hf_root, agree=agree, n=10)
    _write_indices(
        proc_root,
        agree=agree,
        seed=seed,
        indices={"train": [0, 1, 2], "val": [3], "test": [4, 5]},
    )

    ds = load_phrasebank_splits(
        seed=seed,
        agree=agree,
        hf_dataset_root=hf_root,
        processed_root=proc_root,
    )

    assert set(ds.keys()) == {"train", "val", "test"}
    assert len(ds["train"]) == 3
    assert len(ds["val"]) == 1
    assert len(ds["test"]) == 2
    assert set(ds["train"].column_names) == {"text", "label"}
    assert ds["train"][0]["text"] == "s0"
    assert isinstance(ds["train"][0]["label"], int)


def test_phrasebank_loader_rejects_invalid_indices_format(tmp_path: Path) -> None:
    hf_root = tmp_path / "hf_snapshot"
    proc_root = tmp_path / "processed"
    agree = "sentences_66agree"
    seed = 5768

    _make_hf_snapshot(hf_root, agree=agree, n=10)
    _write_indices(
        proc_root,
        agree=agree,
        seed=seed,
        indices={"train": "not a list", "val": [3], "test": [4, 5]},
    )

    with pytest.raises(PhraseBankContractError):
        load_phrasebank_splits(
            seed=seed,
            agree=agree,
            hf_dataset_root=hf_root,
            processed_root=proc_root,
        )


def test_phrasebank_loader_rejects_overlap_between_splits(tmp_path: Path) -> None:
    hf_root = tmp_path / "hf_snapshot"
    proc_root = tmp_path / "processed"
    agree = "sentences_66agree"
    seed = 5768
    _make_hf_snapshot(hf_root, agree=agree, n=10)
    _write_indices(
        proc_root,
        agree=agree,
        seed=seed,
        indices={"train": [0, 1, 2], "val": [2], "test": [4, 5]},
    )

    with pytest.raises(PhraseBankContractError, match="overlap"):
        load_phrasebank_splits(
            seed=seed,
            agree=agree,
            hf_dataset_root=hf_root,
            processed_root=proc_root,
        )


def test_phrasebank_loader_rejects_out_of_range_indices(tmp_path: Path) -> None:
    hf_root = tmp_path / "hf_snapshot"
    proc_root = tmp_path / "processed"
    agree = "sentences_66agree"
    seed = 5768

    _make_hf_snapshot(hf_root, agree=agree, n=6)
    _write_indices(
        proc_root,
        agree=agree,
        seed=seed,
        indices={"train": [0, 1, 2], "val": [3], "test": [999]},
    )

    with pytest.raises(PhraseBankContractError, match="out-of-range"):
        load_phrasebank_splits(
            seed=seed,
            agree=agree,
            hf_dataset_root=hf_root,
            processed_root=proc_root,
        )


def test_phrasebank_loader_schema_gate(tmp_path: Path) -> None:
    # Build a snapshot with wrong column name ("sentence" instead of "text")
    agree = "sentences_66agree"
    seed = 5768
    hf_root = tmp_path / "hf_snapshot"
    proc_root = tmp_path / "processed"

    base = Dataset.from_dict({"sentence": [f"s{i}" for i in range(10)], "label": [0] * 10})
    DatasetDict({agree: base}).save_to_disk(str(hf_root))
    _write_indices(proc_root, agree=agree, seed=seed, indices={"train": [0], "val": [1], "test": [2]})

    with pytest.raises(PhraseBankContractError, match="schema gate failed"):
        load_phrasebank_splits(
            seed=seed,
            agree=agree,
            hf_dataset_root=hf_root,
            processed_root=proc_root,
        )