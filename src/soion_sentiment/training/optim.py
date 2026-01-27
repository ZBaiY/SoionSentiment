from __future__ import annotations

import torch
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from soion_sentiment.config import Config


def _decay_param_names(model) -> set[str]:
    decay = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            continue
        for param_name, _ in module.named_parameters(recurse=False):
            if param_name.endswith("bias"):
                continue
            if name:
                decay.add(f"{name}.{param_name}")
            else:
                decay.add(param_name)
    return decay


def build_optimizer(cfg: Config, model) -> torch.optim.Optimizer:
    decay_names = _decay_param_names(model)
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in decay_names:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.optim.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
        eps=cfg.optim.eps,
    )


def build_scheduler(cfg: Config, optimizer, num_training_steps: int):
    if cfg.scheduler.warmup_steps is not None:
        warmup_steps = cfg.scheduler.warmup_steps
    else:
        warmup_steps = int(num_training_steps * cfg.scheduler.warmup_ratio)

    name = cfg.scheduler.name.lower()
    if name == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    if name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps,
            num_training_steps,
            num_cycles=cfg.scheduler.num_cycles,
        )
    raise ValueError(f"unsupported scheduler: {cfg.scheduler.name}")
