from __future__ import annotations

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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train and cfg.data.shuffle_train,
        num_workers=cfg.runtime.num_workers,
        pin_memory=cfg.runtime.pin_memory,
        collate_fn=collator,
    )
