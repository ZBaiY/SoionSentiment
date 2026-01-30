from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from soion_sentiment.config import Config


def build_collator(cfg: Config, tokenizer):
    return DataCollatorWithPadding(
        tokenizer,
        padding=cfg.data.padding,
        max_length=cfg.data.max_length,
        return_tensors="pt",
    )


def build_dataloader(cfg: Config, dataset, tokenizer, split: str) -> DataLoader:
    is_train = split == "train"
    batch_size = cfg.training.batch_size if is_train else cfg.training.eval_batch_size
    collator = build_collator(cfg, tokenizer) # batch size and padding strategy from config
    ## Dataloader will handle batching, shuffling, parallel loading, pin memory, collating etc.
    num_workers = cfg.train.dataloader.num_workers
    pin_memory = cfg.train.dataloader.pin_memory
    persistent_workers = cfg.train.dataloader.persistent_workers if num_workers > 0 else False
    kwargs = {
        "batch_size": batch_size,
        "shuffle": is_train and cfg.data.shuffle_train,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "collate_fn": collator,
    }
    if num_workers > 0 and cfg.train.dataloader.prefetch_factor is not None:
        kwargs["prefetch_factor"] = cfg.train.dataloader.prefetch_factor
    return DataLoader(
        dataset,
        **kwargs,
    )
