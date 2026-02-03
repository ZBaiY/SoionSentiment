from __future__ import annotations

import hashlib

from datasets import Dataset, DatasetDict

from soion_sentiment.data.tokenize import _ensure_dataset_row, _ensure_sample_id


def _make_dataset() -> DatasetDict:
    texts = ["alpha", "beta", "gamma", "delta"]
    labels = [0, 1, 0, 1]
    split_ds = Dataset.from_dict({"text": texts, "label": labels})
    return DatasetDict({"train": split_ds, "val": split_ds, "test": split_ds})


def _collect_sample_ids(raw_ds, batch_size: int) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    rows = list(range(len(raw_ds)))
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i : i + batch_size]
        for idx in batch_rows:
            row = raw_ds[idx]
            items.append((row["text"], row["sample_id"]))
    return items


def test_sample_id_stable_across_batch_sizes() -> None:
    ds = _ensure_sample_id(_ensure_dataset_row(_make_dataset()))
    raw_ds = ds["val"].with_format(None)

    items_a = _collect_sample_ids(raw_ds, batch_size=1)
    items_b = _collect_sample_ids(raw_ds, batch_size=2)
    items_c = _collect_sample_ids(raw_ds, batch_size=2)

    map_a = {text: sample_id for text, sample_id in items_a}
    map_b = {text: sample_id for text, sample_id in items_b}
    map_c = {text: sample_id for text, sample_id in items_c}

    assert map_a == map_b
    assert map_b == map_c

    for text, sample_id in map_a.items():
        assert sample_id == hashlib.sha1(text.encode("utf-8")).hexdigest()
        assert isinstance(sample_id, str)
        assert len(sample_id) == 40
