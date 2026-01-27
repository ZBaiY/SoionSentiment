from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from soion_sentiment.config import RuntimeConfig


@dataclass(frozen=True)
class DeviceSpec:
    device: torch.device
    autocast_dtype: torch.dtype | None
    precision: Literal["fp32", "fp16", "bf16"]
    note: str | None


def _pick_device(runtime_cfg: RuntimeConfig) -> torch.device:
    if runtime_cfg.device == "cpu":
        return torch.device("cpu")
    if runtime_cfg.device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if runtime_cfg.device == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device(runtime_cfg: RuntimeConfig) -> DeviceSpec:
    device = _pick_device(runtime_cfg)
    precision = runtime_cfg.precision
    autocast_dtype: torch.dtype | None = None
    note: str | None = None

    if precision == "fp16":
        if device.type == "cuda":
            autocast_dtype = torch.float16
        elif device.type == "mps":
            autocast_dtype = torch.float16
        else:
            precision = "fp32"
            note = "fp16 requested but unsupported on cpu; falling back to fp32"
    elif precision == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        elif device.type == "mps":
            precision = "fp32"
            note = "bf16 requested but unsupported on mps; falling back to fp32"
        else:
            precision = "fp32"
            note = "bf16 requested but unsupported on cpu; falling back to fp32"

    return DeviceSpec(device=device, autocast_dtype=autocast_dtype, precision=precision, note=note)
