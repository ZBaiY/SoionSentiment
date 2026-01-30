from __future__ import annotations

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


def _env_info() -> str:
    parts = [
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"transformers={transformers.__version__}",
        f"datasets={datasets.__version__}",
    ]
    return "\n".join(parts) + "\n"


def _run_dir(cfg: Config) -> Path:
    # Generate a run directory path based on timestamp and config hash.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_hash = cfg.config_hash()[:8]
    suffix = f"_{cfg.logging.run_name}" if cfg.logging.run_name else ""
    return Path(cfg.logging.run_dir) / f"{timestamp}_{cfg_hash}{suffix}"


def _build_dataset(cfg: Config, tokenizer):
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
    return tokenize_dataset(cfg, ds, tokenizer)


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
    ds = _build_dataset(cfg, tokenizer)
    loader = build_dataloader(cfg, ds[split], tokenizer, split)

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

    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():            # deactivate gradient calculation for evaluation, save memory (no backprop tracking tables for autograd)
        for batch in loader:
            batch = {k: v.to(device_spec.device) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits
            preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            label_tensor = batch["labels"] if "labels" in batch else batch["label"]
            labels.extend(label_tensor.detach().cpu().tolist())

    metrics = compute_metrics(preds, labels, cfg)
    metrics["split"] = split
    return metrics
