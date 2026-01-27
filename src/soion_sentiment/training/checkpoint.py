from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from soion_sentiment.config import Config


def save_checkpoint(
    output_dir: Path,
    model,
    tokenizer,
    optimizer,
    scheduler,
    *,
    step: int,
    epoch: int,
    cfg: Config,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "config": cfg.as_dict(),
        },
        output_dir / "trainer_state.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    ckpt_path = Path(path)
    state_path = ckpt_path / "trainer_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"checkpoint missing trainer_state.pt: {state_path}")
    return torch.load(state_path, map_location="cpu")
