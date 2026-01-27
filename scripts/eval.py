#!/usr/bin/env python3
"""Evaluation entrypoint.

Contract:
- Consumes a *standard schema* DatasetDict with exactly two columns: text, label.
- Applies a *replayable split* via indices.json under data/processed/phrasebank/{agree}/{seed}/.
- Evaluates a HF sequence classification model checkpoint and writes a JSON report.

This script is intentionally strict: if the schema is not exactly {text,label}, it exits.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _project_root_from_file(file: Path) -> Path:
    # scripts/eval.py -> project root
    return file.resolve().parents[1]


PROJECT_ROOT = _project_root_from_file(Path(__file__))
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_PHRASEBANK_ROOT = DEFAULT_DATA_ROOT / "phrasebank_local_v1"
DEFAULT_SPLITS_ROOT = DEFAULT_DATA_ROOT / "processed" / "phrasebank"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "reports" / "eval"


@dataclass(frozen=True)
class EvalConfig:
    agree: str
    seed: int
    split: str
    model: str
    batch_size: int
    max_length: int
    device: str
    out_dir: str
    data_root: str
    phrasebank_root: str
    splits_root: str
    num_errors: int


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def _schema_gate(ds: Any, *, name: str) -> None:
    cols = set(getattr(ds, "column_names", []))
    _require(cols == {"text", "label"}, f"Schema gate failed for {name}: columns={sorted(cols)} expected=['label','text']")


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> float:
    f1s: list[float] = []
    for c in range(num_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        denom = (2 * tp + fp + fn)
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(f1s))


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> list[list[int]]:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm.tolist()


def _device_str(requested: str) -> str:
    r = requested.lower().strip()
    if r in {"auto", "cpu"}:
        return r
    if r in {"cuda", "mps"}:
        return r
    return "auto"


def evaluate_split(
    *,
    ds,
    split: str,
    model_name_or_path: str,
    batch_size: int,
    max_length: int,
    device: str,
    num_errors: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _require(split in ds, f"Split {split!r} not in dataset; available={list(ds.keys())}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

    # Resolve device
    dev = device
    if dev == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"

    torch_device = torch.device(dev)
    model.to(torch_device)
    model.eval()

    num_labels = int(getattr(model.config, "num_labels", 0) or 0)
    _require(num_labels > 0, f"Model config has invalid num_labels={num_labels}")

    def tok(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    eval_ds = ds[split].map(tok, batched=True, remove_columns=[])
    # Keep only columns needed for torch
    keep_cols = [c for c in eval_ds.column_names if c in {"input_ids", "attention_mask", "token_type_ids", "label", "text"}]
    eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in keep_cols])
    eval_ds.set_format(type="torch", columns=[c for c in keep_cols if c not in {"text"}])

    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        # tokenizer.pad expects lists of dicts with input_ids/attention_mask[/token_type_ids]
        # labels are carried separately.
        labels = torch.stack([f["label"] for f in features])
        batch_inp = [{k: v for k, v in f.items() if k in {"input_ids", "attention_mask", "token_type_ids"}} for f in features]
        padded = tokenizer.pad(batch_inp, return_tensors="pt")
        padded["labels"] = labels
        return padded

    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    y_true: list[int] = []
    y_pred: list[int] = []

    # For error sampling we also need the raw text; fetch from original split by index.
    errors: list[dict[str, Any]] = []
    err_cap = max(0, int(num_errors))

    # Iterate with a running index to map back into the original split
    seen = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(torch_device)
            batch = {k: v.to(torch_device) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits
            preds = torch.argmax(logits, dim=-1)

            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

            # Collect misclassified examples (bounded)
            if err_cap and len(errors) < err_cap:
                preds_cpu = preds.detach().cpu().tolist()
                labels_cpu = labels.detach().cpu().tolist()
                for j, (t, p) in enumerate(zip(labels_cpu, preds_cpu)):
                    if t != p and len(errors) < err_cap:
                        text = ds[split][seen + j]["text"]
                        errors.append({"text": text, "true": int(t), "pred": int(p)})
            seen += int(labels.shape[0])

    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)

    acc = float(np.mean(yt == yp)) if yt.size else 0.0
    macro_f1 = _macro_f1(yt, yp, num_classes=num_labels) if yt.size else 0.0
    cm = _confusion(yt, yp, num_classes=num_labels) if yt.size else [[0] * num_labels for _ in range(num_labels)]

    metrics = {
        "n": int(yt.size),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "num_labels": num_labels,
        "device": dev,
    }
    return metrics, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agree", default="66agree", help="PhraseBank agree subset: 66agree/allagree/75agree/50agree")
    ap.add_argument("--seed", type=int, default=5768, help="Split seed directory under data/processed/phrasebank/{agree}/{seed}")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which split to evaluate")
    ap.add_argument("--model", required=True, help="HF model name or local checkpoint directory")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda/mps")
    ap.add_argument("--num-errors", type=int, default=50, help="Max misclassified examples to include in report")

    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--phrasebank-root", default=str(DEFAULT_PHRASEBANK_ROOT))
    ap.add_argument("--splits-root", default=str(DEFAULT_SPLITS_ROOT))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))

    args = ap.parse_args()

    cfg = EvalConfig(
        agree=str(args.agree),
        seed=int(args.seed),
        split=str(args.split),
        model=str(args.model),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        device=_device_str(str(args.device)),
        out_dir=str(Path(args.out_root).resolve()),
        data_root=str(Path(args.data_root).resolve()),
        phrasebank_root=str(Path(args.phrasebank_root).resolve()),
        splits_root=str(Path(args.splits_root).resolve()),
        num_errors=int(args.num_errors),
    )

    phrasebank_root = Path(cfg.phrasebank_root)
    splits_root = Path(cfg.splits_root)
    out_root = Path(cfg.out_dir)

    try:
        from soion_sentiment.data.phrasebank import load_phrasebank_splits  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Standard loader missing: implement soion_sentiment.data.phrasebank.load_phrasebank_splits"
        ) from e

    ds = load_phrasebank_splits(
        phrasebank_root=phrasebank_root,
        splits_root=splits_root,
        agree=cfg.agree,
        seed=cfg.seed,
    )

    for k, v in ds.items():
        _schema_gate(v, name=f"datasetdict[{k}]")

    metrics, errors = evaluate_split(
        ds=ds,
        split=cfg.split,
        model_name_or_path=cfg.model,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        device=cfg.device,
        num_errors=cfg.num_errors,
    )

    out_dir = out_root / "phrasebank" / cfg.agree / str(cfg.seed) / cfg.split
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "metrics": metrics,
        "errors": errors,
    }

    out_path = out_dir / "eval_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"OK wrote {out_path}")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
