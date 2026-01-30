from __future__ import annotations

import torch
import warnings

from transformers import Adafactor, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from soion_sentiment.config import Config


def _is_layernorm_name(name: str) -> bool:
    lowered = name.lower()
    return "layernorm" in lowered or "layer_norm" in lowered or "rmsnorm" in lowered


def _is_bias_name(name: str) -> bool:
    return name.endswith(".bias") or name == "bias"


def _get_norm_module_types() -> tuple[type, ...]:
    types: list[type] = [torch.nn.LayerNorm]
    if hasattr(torch.nn, "RMSNorm"):
        types.append(torch.nn.RMSNorm)  # type: ignore[attr-defined]
    try:
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

        types.extend(list(ALL_LAYERNORM_LAYERS))
    except Exception:
        pass
    return tuple(dict.fromkeys(types))


def _collect_module_param_names(model, module_types: tuple[type, ...]) -> set[str]:
    names: set[str] = set()
    if not module_types:
        return names
    for module_name, module in model.named_modules():
        if isinstance(module, module_types):
            for param_name, _ in module.named_parameters(recurse=False):
                if module_name:
                    names.add(f"{module_name}.{param_name}")
                else:
                    names.add(param_name)
    return names


def _collect_embedding_param_names(model) -> set[str]:
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            for param_name, _ in module.named_parameters(recurse=False):
                if module_name:
                    names.add(f"{module_name}.{param_name}")
                else:
                    names.add(param_name)
    return names


def split_decay_no_decay_params(
    model, decay_embeddings: bool
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], dict]:
    # Prefer HF utilities to identify decay candidates; fallback to module-type traversal.
    try:
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        from transformers.trainer_pt_utils import get_parameter_names

        decay_names = set(get_parameter_names(model, ALL_LAYERNORM_LAYERS))
        hf_used = True
    except Exception:
        decay_names = {name for name, _ in model.named_parameters()}
        hf_used = False

    norm_types = _get_norm_module_types()
    norm_param_names = _collect_module_param_names(model, norm_types)
    if not norm_param_names:
        norm_param_names = {n for n, _ in model.named_parameters() if _is_layernorm_name(n)}

    embedding_names = _collect_embedding_param_names(model)

    no_decay_names = {name for name, _ in model.named_parameters() if _is_bias_name(name)}
    no_decay_names.update(norm_param_names)
    if not decay_embeddings:
        no_decay_names.update(embedding_names)

    decay_names = {n for n in decay_names if n not in no_decay_names and not _is_bias_name(n)}

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    moved_from_decay: list[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in decay_names:
            if name in no_decay_names:
                moved_from_decay.append(name)
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        else:
            no_decay_params.append(param)

    debug_info = {
        "hf_used": hf_used,
        "decay_count": len(decay_params),
        "no_decay_count": len(no_decay_params),
        "decay_sample": [
            n for n, p in model.named_parameters() if id(p) in {id(x) for x in decay_params}
        ][:5], ## show some param names for debugging
        "no_decay_sample": [
            n for n, p in model.named_parameters() if id(p) in {id(x) for x in no_decay_params}
        ][:5],
        "moved_from_decay": moved_from_decay[:5],
        "decay_embeddings": decay_embeddings,
    }
    return decay_params, no_decay_params, debug_info


def build_optimizer(cfg: Config, model) -> torch.optim.Optimizer:
    decay_params, no_decay_params, debug_info = split_decay_no_decay_params(
        model, decay_embeddings=cfg.optim.decay_embeddings
    )
    warnings.warn(
        "optimizer params: "
        f"decay={debug_info['decay_count']} no_decay={debug_info['no_decay_count']} "
        f"hf_used={debug_info['hf_used']} decay_embeddings={debug_info['decay_embeddings']}",
        stacklevel=2,
    )
    if debug_info["moved_from_decay"]:
        warnings.warn(
            f"moved from decay to no_decay: {debug_info['moved_from_decay']}",
            stacklevel=2,
        )
    if not decay_params and not no_decay_params:
        warnings.warn("No trainable parameters found for optimizer.", stacklevel=2)

    name = cfg.train.optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(
            [
                {"params": decay_params, "weight_decay": cfg.optim.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=cfg.optim.lr,
        )
    if name == "adafactor":
        lr = cfg.train.adafactor_lr if cfg.train.adafactor_lr is not None else cfg.optim.lr
        return Adafactor(
            [
                {"params": decay_params, "weight_decay": cfg.optim.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            clip_threshold=1.0,
        )
    if name != "adamw":
        raise ValueError(f"unsupported optimizer: {cfg.train.optimizer}")
    try:
        device_type = next(model.parameters()).device.type
    except StopIteration:
        device_type = "cpu"
    if cfg.train.adamw_foreach is None:
        foreach = device_type == "mps"
    else:
        foreach = cfg.train.adamw_foreach
    fused = None
    if cfg.train.adamw_fused is not None:
        fused = cfg.train.adamw_fused and device_type != "mps"
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.optim.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
        eps=cfg.optim.eps,
        foreach=foreach,
        **({} if fused is None else {"fused": fused}),
    )


def build_scheduler(cfg: Config, optimizer, num_training_steps: int):
    # Build learning rate scheduler according to config.
    # warming up is designed to gradually increase learning rate from 0 to the target value, in order to stabilize training at the beginning.
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
