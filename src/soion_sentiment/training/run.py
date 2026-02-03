from __future__ import annotations

import inspect
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import datasets
import torch
import transformers
from transformers import AutoModelForSequenceClassification

from soion_sentiment.config import Config, load_config
from soion_sentiment.data.collate import build_dataloader
from soion_sentiment.data.phrasebank import load_phrasebank_splits
from soion_sentiment.data.tokenize import build_tokenizer, tokenize_dataset
from soion_sentiment.training.device import get_device
from soion_sentiment.training.loop import train
from soion_sentiment.data.manifest import build_data_manifest, write_data_manifest
from soion_sentiment.training.metrics import compute_metrics
from soion_sentiment.model.model import build_model
from soion_sentiment.training.seed import set_seed
from soion_sentiment.analysis.mistakes_io import write_mistakes_jsonl


def _env_info() -> str:
    parts = [
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"transformers={transformers.__version__}",
        f"datasets={datasets.__version__}",
    ]
    return "\n".join(parts) + "\n"


def _filter_model_inputs(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(model.forward).parameters
    for param in params.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return batch
    allowed = {name for name in params if name != "self"}
    return {key: value for key, value in batch.items() if key in allowed}


def _run_dir(cfg: Config) -> Path:
    # Generate a run directory path based on timestamp and config hash.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_hash = cfg.config_hash()[:8]
    suffix = f"_{cfg.logging.run_name}" if cfg.logging.run_name else ""
    return Path(cfg.logging.run_dir) / f"{timestamp}_{cfg_hash}{suffix}"


def _build_dataset(
    cfg: Config,
    tokenizer,
    *,
    keep_text: bool = False,
    add_example_id: bool = False,
    add_sample_id: bool = False,
    add_dataset_row: bool = False,
):
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
    return tokenize_dataset(
        cfg,
        ds,
        tokenizer,
        keep_text=keep_text,
        add_example_id=add_example_id,
        add_sample_id=add_sample_id,
        add_dataset_row=add_dataset_row,
    )


def run_train(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    *,
    data_ref: str | None = None,
    model_ref: str | None = None,
    preset_ref: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(
        config_path,
        overrides,
        data_ref=data_ref,
        model_ref=model_ref,
        preset_ref=preset_ref,
    )
    set_seed(cfg.seed, cfg.runtime.deterministic) # seeds for data loading, numpy, torch, etc.
    device_spec = get_device(cfg.runtime, cfg.train.precision)

    run_dir = _run_dir(cfg) # create run directory, store resolved config and env info and etc.
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    env_text = _env_info()
    if device_spec.note:
        env_text += f"\n{device_spec.note}\n"
    (run_dir / "env.txt").write_text(env_text, encoding="utf-8")
    

    tokenizer = build_tokenizer(cfg)  # tokenizer from config, usually pretrained, together with model backbone
    ds = _build_dataset(cfg, tokenizer) # drag the dataset, get data, pass through tokenizer, return labeled datasets, with ref in dictionary and whether it is masked
    manifest = build_data_manifest(cfg) # to record dataset info, for logging and reproducibility

    write_data_manifest(run_dir / "data_manifest.json", manifest)
    train_loader = build_dataloader(cfg, ds["train"], tokenizer, "train")
    val_loader = build_dataloader(cfg, ds["val"], tokenizer, "val")

    model = build_model(cfg) # build model from config, usually pretrained backbone + classification head
    summary = train(
        cfg,
        run_dir=run_dir,                            # the dir just created
        model=model,                                # the model just built
        tokenizer=tokenizer,                        # the tokenizer just built
        train_loader=train_loader,                  # the training dataloader just built
        val_loader=val_loader,                      # the validation dataloader just built
        device=device_spec.device,                  # the device to run on, usually cuda or cpu    
        autocast_dtype=device_spec.autocast_dtype,  # the dtype for autocasting, usually float16 or bfloat16 or None
        precision=device_spec.precision,            # the precision mode, usually "amp" or "bf16" or "fp32"
    )

    return {"run_dir": str(run_dir), "summary": summary}


def run_eval(
    config_path: str | Path,
    overrides: dict[str, Any] | None,
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    data_ref: str | None = None,
    model_ref: str | None = None,
    preset_ref: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(
        config_path,
        overrides,
        data_ref=data_ref,
        model_ref=model_ref,
        preset_ref=preset_ref,
    )
    device_spec = get_device(cfg.runtime, cfg.train.precision)
    tokenizer = build_tokenizer(cfg)
    collect_mistakes = cfg.eval.mistake_path is not None
    ds = _build_dataset(
        cfg,
        tokenizer,
        keep_text=collect_mistakes,
        add_example_id=collect_mistakes,
        add_dataset_row=collect_mistakes,
    )
    loader = build_dataloader(cfg, ds[split], tokenizer, split)
    raw_ds = ds[split].with_format(None) if collect_mistakes else None

    labels = cfg.model.labels
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    model_kwargs = {
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

    run_dir = checkpoint_path.parent
    if not (run_dir / "resolved_config.yaml").exists():
        run_dir = checkpoint_path
    run_id = run_dir.name

    preds: list[int] = []
    labels: list[int] = []
    mistakes: list[dict[str, Any]] = []
    with torch.no_grad():            # deactivate gradient calculation for evaluation, save memory (no backprop tracking tables for autograd)
        for batch in loader:
            batch = {k: v.to(device_spec.device) for k, v in batch.items()}
            model_inputs = _filter_model_inputs(model, batch)
            out = model(**model_inputs)
            logits = out.logits
            preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            label_tensor = batch["labels"] if "labels" in batch else batch["label"]
            labels.extend(label_tensor.detach().cpu().tolist())
            if collect_mistakes and raw_ds is not None and "dataset_row" in batch:
                probs = torch.softmax(logits, dim=-1).detach().cpu()
                top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
                margins = (top2[:, 0] - top2[:, 1]).tolist() if top2.shape[-1] > 1 else [0.0] * probs.shape[0]
                batch_preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
                batch_labels = label_tensor.detach().cpu().tolist()
                dataset_rows = batch["dataset_row"].detach().cpu().tolist()
                for i, (pred_id, true_id) in enumerate(zip(batch_preds, batch_labels)):
                    if pred_id == true_id:
                        continue
                    row_idx = int(dataset_rows[i])
                    raw_row = raw_ds[row_idx]
                    text = raw_row.get("text", "")
                    token_len = None
                    if "input_ids" in raw_row:
                        token_len = len(raw_row["input_ids"])
                    truncated = None
                    if token_len is not None:
                        truncated = bool(cfg.data.truncation and token_len >= cfg.data.max_length)
                    mistakes.append(
                        {
                            "split": split,
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
                            "sample_id": raw_row.get("sample_id"),
                        }
                    )

    metrics = compute_metrics(preds, labels, cfg)
    metrics["split"] = split
    if collect_mistakes:
        mistake_path = Path(cfg.eval.mistake_path) if cfg.eval.mistake_path else None
        if mistake_path is not None:
            if not mistake_path.is_absolute():
                mistake_path = run_dir / mistake_path
            write_mistakes_jsonl(
                mistakes,
                mistake_path,
                max_n=cfg.eval.mistake_max_n,
                seed=cfg.eval.mistake_seed,
                run_id=run_id,
                split=split,
            )
    return metrics
