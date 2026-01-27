from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from soion_sentiment.config import Config
from soion_sentiment.data.phrasebank import load_phrasebank_splits


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


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


def build_dataset(cfg: Config, tokenizer) -> DatasetDict:
    if cfg.data.name != "phrasebank":
        raise ValueError(f"unsupported dataset: {cfg.data.name}")
    if cfg.data.split_protocol != "precomputed":
        raise ValueError("phrasebank requires data.split_protocol=precomputed")

    ds = load_phrasebank_splits(
        seed=cfg.seed,
        agree=cfg.data.agree or "sentences_66agree",
        hf_dataset_root=cfg.data.hf_dataset_root,
        processed_root=cfg.data.processed_root,
    )

    text_field = cfg.data.text_field
    label_field = cfg.data.label_field
    if label_field != "label":
        ds = ds.rename_column(label_field, "label")
    if text_field != "text":
        ds = ds.rename_column(text_field, "text")

    ds = _apply_max_samples(ds, cfg)

    num_labels = len(cfg.model.labels)
    for split in ["train", "val", "test"]:
        labels = ds[split]["label"]
        if labels and (min(labels) < 0 or max(labels) >= num_labels):
            raise ValueError(f"labels out of range for split={split}, num_labels={num_labels}")

    def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=cfg.data.truncation,
            max_length=cfg.data.max_length,
            padding=False,
        )

    ds = ds.map(_tokenize, batched=True)
    ds = ds.remove_columns(["text"])

    for split in ["train", "val", "test"]:
        cols = ["input_ids", "attention_mask", "label"]
        if "token_type_ids" in ds[split].column_names:
            cols.append("token_type_ids")
        ds[split] = ds[split].with_format("torch", columns=cols)

    return ds


def build_dataloader(cfg: Config, dataset, tokenizer, split: str) -> DataLoader:
    is_train = split == "train"
    batch_size = cfg.training.batch_size if is_train else cfg.training.eval_batch_size
    collator = DataCollatorWithPadding(
        tokenizer,
        padding=cfg.data.padding,
        max_length=cfg.data.max_length,
        return_tensors="pt",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train and cfg.data.shuffle_train,
        num_workers=cfg.runtime.num_workers,
        pin_memory=cfg.runtime.pin_memory,
        collate_fn=collator,
    )


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
