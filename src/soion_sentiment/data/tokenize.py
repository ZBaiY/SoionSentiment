from __future__ import annotations

from typing import Any

import hashlib

from datasets import DatasetDict
from transformers import AutoTokenizer

from soion_sentiment.config import Config


def build_tokenizer(cfg: Config):
    kwargs = {"use_fast": cfg.tokenizer.use_fast, "local_files_only": cfg.runtime.hf_offline}
    if cfg.runtime.hf_cache_dir is not None:
        kwargs["cache_dir"] = cfg.runtime.hf_cache_dir
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone, **kwargs)
    tokenizer.padding_side = cfg.tokenizer.padding_side
    tokenizer.truncation_side = cfg.tokenizer.truncation_side
    return tokenizer


def _apply_max_samples(ds: DatasetDict, cfg: Config) -> DatasetDict:
    if cfg.data.max_train_samples is not None:
        ds["train"] = ds["train"].select(range(min(cfg.data.max_train_samples, len(ds["train"]))))
    if cfg.data.max_eval_samples is not None:
        ds["val"] = ds["val"].select(range(min(cfg.data.max_eval_samples, len(ds["val"]))))
    if cfg.data.max_test_samples is not None:
        ds["test"] = ds["test"].select(range(min(cfg.data.max_test_samples, len(ds["test"]))))
    return ds


def _check_label_range(ds: DatasetDict, cfg: Config) -> None:
    num_labels = len(cfg.model.labels)
    for split in ["train", "val", "test"]:
        labels = ds[split]["label"]
        if labels and (min(labels) < 0 or max(labels) >= num_labels):
            raise ValueError(f"labels out of range for split={split}, num_labels={num_labels}")


def _ensure_example_id(ds: DatasetDict) -> DatasetDict:
    for split in ["train", "val", "test"]:
        cols = set(ds[split].column_names)
        if "example_id" in cols:
            continue
        id_source = None
        for key in ("id", "idx", "index"):
            if key in cols:
                id_source = key
                break
        if id_source is not None:
            ds[split] = ds[split].map(lambda row: {"example_id": str(row[id_source])})
            continue
        ds[split] = ds[split].map(
            lambda row: {"example_id": hashlib.sha1(row["text"].encode("utf-8")).hexdigest()}
        )
    return ds


def _ensure_sample_id(ds: DatasetDict) -> DatasetDict:
    for split in ["train", "val", "test"]:
        cols = set(ds[split].column_names)
        if "sample_id" in cols:
            continue
        ds[split] = ds[split].map(
            lambda row: {"sample_id": hashlib.sha1(row["text"].encode("utf-8")).hexdigest()}
        )
    return ds


def _ensure_dataset_row(ds: DatasetDict) -> DatasetDict:
    for split in ["train", "val", "test"]:
        ds[split] = ds[split].map(lambda _row, idx: {"dataset_row": idx}, with_indices=True)
    return ds


def tokenize_dataset(
    cfg: Config,
    ds: DatasetDict,
    tokenizer,
    *,
    keep_text: bool = False,
    add_example_id: bool = False,
    add_sample_id: bool = False,
    add_dataset_row: bool = False,
) -> DatasetDict:
    text_field = cfg.data.text_field
    label_field = cfg.data.label_field
    if label_field != "label":
        ds = ds.rename_column(label_field, "label")
    if text_field != "text":
        ds = ds.rename_column(text_field, "text")

    if add_example_id:
        ds = _ensure_example_id(ds)
    if add_sample_id:
        ds = _ensure_sample_id(ds)
    if add_dataset_row:
        ds = _ensure_dataset_row(ds)

    ds = _apply_max_samples(ds, cfg)
    _check_label_range(ds, cfg)

    def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=cfg.data.truncation,
            max_length=cfg.data.max_length,
            padding=False,
        )

    ds = ds.map(_tokenize, batched=True)
    if not keep_text:
        ds = ds.remove_columns(["text"])

    for split in ["train", "val", "test"]:
        cols = ["input_ids", "attention_mask", "label"]
        if "token_type_ids" in ds[split].column_names:
            cols.append("token_type_ids")
        if add_dataset_row and "dataset_row" in ds[split].column_names:
            cols.append("dataset_row")
        ds[split] = ds[split].with_format("torch", columns=cols)

    return ds
