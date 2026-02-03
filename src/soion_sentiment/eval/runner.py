from __future__ import annotations

import inspect
import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from soion_sentiment.data.collate import build_collator
from soion_sentiment.data.phrasebank import load_phrasebank_splits
from soion_sentiment.data.tokenize import build_tokenizer, tokenize_dataset
from soion_sentiment.analysis.mistakes_io import write_mistakes_jsonl
from soion_sentiment.training.device import get_device
from soion_sentiment.training.seed import set_seed

from .config import EvalSuiteConfig
from .loader import load_run_config
from .metrics import compute_classification_metrics, finalize_ece, init_ece_bins, update_ece_bins


_EVAL_SPLIT_ALIASES = {
    "val": "val",
    "validation": "val",
    "valid": "val",
    "dev": "val",
    "test": "test",
    "train": "train",
}


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_split(name: str) -> str:
    key = str(name).strip().lower()
    return _EVAL_SPLIT_ALIASES.get(key, key)


def _filter_model_inputs(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(model.forward).parameters
    for param in params.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return batch
    allowed = {name for name in params if name != "self"}
    return {key: value for key, value in batch.items() if key in allowed}


def _resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    if checkpoint in {"best", "last"}:
        candidate = run_dir / checkpoint
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"checkpoint not found: {candidate}")
    raw = Path(checkpoint)
    if raw.is_absolute():
        if raw.exists():
            return raw
        raise FileNotFoundError(f"checkpoint not found: {raw}")
    candidate = run_dir / raw
    if candidate.exists():
        return candidate
    cwd_candidate = Path.cwd() / raw
    if cwd_candidate.exists():
        return cwd_candidate
    raise FileNotFoundError(f"checkpoint not found: {checkpoint}")


def _resolve_out_dir(template: str, run_id: str, root: Path) -> Path:
    expanded = template.replace("<run_id>", run_id)
    out_path = Path(expanded)
    if out_path.is_absolute():
        return out_path
    prefix = Path("runs") / run_id
    if out_path == prefix:
        return root
    if prefix in out_path.parents:
        return root / out_path.relative_to(prefix)
    return root / out_path


def _build_dataset(
    cfg,
    *,
    split: str,
    keep_text: bool,
    add_example_id: bool,
    add_sample_id: bool,
    add_dataset_row: bool,
):
    if cfg.data.name != "phrasebank":
        raise ValueError(f"unsupported dataset: {cfg.data.name}")
    if cfg.data.split_protocol != "precomputed":
        raise ValueError("phrasebank requires data.split_protocol=precomputed")

    tokenizer = build_tokenizer(cfg)
    ds = load_phrasebank_splits(
        seed=cfg.seed,
        agree=cfg.data.agree or "sentences_66agree",
        hf_dataset_root=cfg.data.hf_dataset_root,
        processed_root=cfg.data.processed_root,
    )
    ds = tokenize_dataset(
        cfg,
        ds,
        tokenizer,
        keep_text=keep_text,
        add_example_id=add_example_id,
        add_sample_id=add_sample_id,
        add_dataset_row=add_dataset_row,
    )
    return ds, tokenizer


def _build_eval_loader(cfg, dataset, tokenizer, batch_size: int) -> DataLoader:
    collator = build_collator(cfg, tokenizer)
    num_workers = cfg.train.dataloader.num_workers
    pin_memory = cfg.train.dataloader.pin_memory
    persistent_workers = cfg.train.dataloader.persistent_workers if num_workers > 0 else False
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "collate_fn": collator,
    }
    if num_workers > 0 and cfg.train.dataloader.prefetch_factor is not None:
        kwargs["prefetch_factor"] = cfg.train.dataloader.prefetch_factor
    return DataLoader(dataset, **kwargs)


def _slice_dataset(dataset, max_samples: int | None):
    if max_samples is None:
        return dataset
    n = min(max_samples, len(dataset))
    return dataset.select(range(n))


def _loss_fn(cfg) -> torch.nn.Module:
    kwargs: dict[str, Any] = {"reduction": "mean"}
    if cfg.training.label_smoothing > 0:
        kwargs["label_smoothing"] = cfg.training.label_smoothing
    return torch.nn.CrossEntropyLoss(**kwargs)


def _eval_one(
    cfg,
    *,
    checkpoint_path: Path,
    run_id: str,
    eval_name: str,
    data_ref: str,
    split: str,
    batch_size: int,
    max_samples: int | None,
    metrics: list[str],
    n_bins: int,
    collect_mistakes: bool,
) -> dict[str, Any]:
    set_seed(cfg.seed, cfg.runtime.deterministic)
    device_spec = get_device(cfg.runtime, cfg.train.precision)

    ds_dict, tokenizer = _build_dataset(
        cfg,
        split=split,
        keep_text=collect_mistakes,
        add_example_id=collect_mistakes,
        add_sample_id=collect_mistakes,
        add_dataset_row=collect_mistakes,
    )
    split_key = _normalize_split(split)
    if split_key not in ds_dict:
        raise ValueError(f"split {split!r} not found in dataset")
    ds = _slice_dataset(ds_dict[split_key], max_samples)
    raw_ds = ds_dict[split_key].with_format(None) if collect_mistakes else None
    loader = _build_eval_loader(cfg, ds, tokenizer, batch_size)

    labels = cfg.model.labels
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    model_kwargs: dict[str, Any] = {
        "local_files_only": cfg.runtime.hf_offline,
        "num_labels": len(labels),
        "label2id": label2id,
        "id2label": id2label,
    }
    if cfg.runtime.hf_cache_dir is not None:
        model_kwargs["cache_dir"] = cfg.runtime.hf_cache_dir
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_path), **model_kwargs)
    model.to(device_spec.device)
    model.eval()

    loss_fn = _loss_fn(cfg)
    preds: list[int] = []
    labels_list: list[int] = []
    mistakes: list[dict[str, Any]] = []
    sample_id_text: dict[str, str] = {}
    seen = 0
    loss_sum = 0.0
    ece_state = init_ece_bins(n_bins) if "ece" in metrics else None

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device_spec.device) for k, v in batch.items()}
            model_inputs = _filter_model_inputs(model, batch)
            out = model(**model_inputs)
            logits = out.logits
            labels_tensor = batch["labels"] if "labels" in batch else batch["label"]
            dataset_rows = batch.get("dataset_row")

            remaining = None
            if max_samples is not None:
                remaining = max_samples - seen
                if remaining <= 0:
                    break
                if labels_tensor.shape[0] > remaining:
                    logits = logits[:remaining]
                    labels_tensor = labels_tensor[:remaining]
                    if dataset_rows is not None:
                        dataset_rows = dataset_rows[:remaining]

            loss = loss_fn(logits, labels_tensor)
            batch_size_actual = labels_tensor.shape[0]
            loss_sum += float(loss.item()) * batch_size_actual
            seen += batch_size_actual

            batch_preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            batch_labels = labels_tensor.detach().cpu().tolist()
            preds.extend(batch_preds)
            labels_list.extend(batch_labels)

            if ece_state is not None:
                probs = torch.softmax(logits, dim=-1)
                conf, pred = torch.max(probs, dim=-1)
                correct = (pred == labels_tensor).float()
                update_ece_bins(
                    ece_state,
                    conf.detach().cpu().numpy().astype(np.float64),
                    correct.detach().cpu().numpy().astype(np.float64),
                )
            if collect_mistakes and raw_ds is not None and dataset_rows is not None:
                probs = torch.softmax(logits, dim=-1).detach().cpu()
                top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
                margins = (top2[:, 0] - top2[:, 1]).tolist() if top2.shape[-1] > 1 else [0.0] * probs.shape[0]
                for i, (pred_id, true_id) in enumerate(zip(batch_preds, batch_labels)):
                    if pred_id == true_id:
                        continue
                    row_idx = int(dataset_rows[i].detach().cpu().item())
                    raw_row = raw_ds[row_idx]
                    text = raw_row.get("text", "")
                    sample_id = raw_row.get("sample_id")
                    if sample_id is None:
                        raise ValueError(
                            "sample_id missing from dataset; ensure add_sample_id=True during preprocessing"
                        )
                    if not isinstance(sample_id, str) or len(sample_id) != 40:
                        raise ValueError(f"invalid sample_id for dataset_row={row_idx}: {sample_id!r}")
                    prev_text = sample_id_text.get(sample_id)
                    if prev_text is not None and prev_text != text:
                        warnings.warn(
                            f"sample_id collision with different text for dataset_row={row_idx}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    else:
                        sample_id_text[sample_id] = text
                    token_len = None
                    if "input_ids" in raw_row:
                        token_len = len(raw_row["input_ids"])
                    truncated = None
                    if token_len is not None:
                        truncated = bool(cfg.data.truncation and token_len >= cfg.data.max_length)
                    mistakes.append(
                        {
                            "split": eval_name,
                            "y_true": id2label.get(true_id, str(true_id)),
                            "y_pred": id2label.get(pred_id, str(pred_id)),
                            "probs": probs[i].tolist(),
                            "label_order": list(cfg.model.labels),
                            "margin": float(margins[i]),
                            "text": text,
                            "token_len": token_len,
                            "truncated": truncated,
                            "dataset_row": row_idx,
                            "agreement_source": cfg.data.agree,
                            "run_id": run_id,
                            "sample_id": sample_id,
                        }
                    )

    include_conf = "confusion_matrix" in metrics
    metric_out = compute_classification_metrics(
        preds,
        labels_list,
        cfg,
        include_confusion_matrix=include_conf,
    )

    out_metrics: dict[str, Any] = {}
    if "loss" in metrics:
        out_metrics["loss"] = (loss_sum / seen) if seen > 0 else None
    if "acc" in metrics and "acc" in metric_out:
        out_metrics["acc"] = metric_out["acc"]
    if "macro_f1" in metrics and "macro_f1" in metric_out:
        out_metrics["macro_f1"] = metric_out["macro_f1"]
    if "per_class_f1" in metrics:
        for key, value in metric_out.items():
            if key.startswith("f1_"):
                out_metrics[key] = value
    if "confusion_matrix" in metrics and "confusion_matrix" in metric_out:
        out_metrics["confusion_matrix"] = metric_out["confusion_matrix"]
    if "ece" in metrics:
        ece_val, bins = finalize_ece(ece_state) if ece_state is not None else (None, [])
        out_metrics["ece"] = ece_val
        out_metrics["ece_bins"] = bins

    record = {
        "run_id": run_id,
        "checkpoint": str(checkpoint_path),
        "eval_name": eval_name,
        "dataset_ref": data_ref,
        "split": split_key,
        "n_samples": int(seen),
        "metrics": out_metrics,
        "label2id": label2id,
        "id2label": id2label,
        "mistakes": mistakes,
    }
    return record


def _safe_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def run_eval_suite(cfg: EvalSuiteConfig) -> tuple[list[dict[str, Any]], Path]:
    run_dir = Path(cfg.runs.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    run_id = run_dir.name
    checkpoint_path = _resolve_checkpoint(run_dir, cfg.runs.checkpoint)
    out_dir = _resolve_out_dir(cfg.eval.logging.out_dir, run_id, checkpoint_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = cfg.eval.metrics or [
        "loss",
        "macro_f1",
        "acc",
        "per_class_f1",
        "confusion_matrix",
        "ece",
    ]

    records: list[dict[str, Any]] = []
    all_mistakes: list[dict[str, Any]] = []
    for entry in cfg.suite:
        run_cfg = load_run_config(run_dir, data_ref=entry.data_ref, overrides=cfg.eval.overrides)
        record = _eval_one(
            run_cfg,
            checkpoint_path=checkpoint_path,
            run_id=run_id,
            eval_name=entry.name,
            data_ref=entry.data_ref,
            split=entry.split,
            batch_size=cfg.eval.batch_size,
            max_samples=cfg.eval.max_samples,
            metrics=metrics,
            n_bins=cfg.eval.calibration.n_bins,
            collect_mistakes=cfg.eval.mistake_path is not None,
        )
        all_mistakes.extend(record.pop("mistakes", []))
        records.append(record)

    jsonl_path = out_dir / cfg.eval.logging.out_jsonl
    _safe_write_jsonl(jsonl_path, records)

    if cfg.eval.mistake_path is not None:
        mistake_path = out_dir / cfg.eval.mistake_path
        write_mistakes_jsonl(
            all_mistakes,
            mistake_path,
            max_n=cfg.eval.mistake_max_n,
            seed=cfg.eval.mistake_seed,
            run_id=run_id,
            split="suite",
        )

    return records, out_dir
