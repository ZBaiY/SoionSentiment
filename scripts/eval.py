from __future__ import annotations

import argparse
import json
from pathlib import Path

from soion_sentiment.config import parse_overrides
from soion_sentiment.training.run import run_eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--data-ref", type=str, default=None)
    p.add_argument("--model-ref", type=str, default=None)
    p.add_argument("--preset-ref", type=str, default=None)
    p.add_argument("--override", action="append", default=[], help="Override config: key=value")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--which", type=str, choices=["best", "last"], default="best")
    p.add_argument("--split", type=str, choices=["val", "test"], default="test")
    return p.parse_args()


def _resolve_checkpoint(path: Path, which: str) -> Path:
    if (path / "trainer_state.pt").exists():
        return path
    candidate = path / which
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"checkpoint not found: {path}")


def main() -> None:
    args = parse_args()
    overrides = parse_overrides(args.override)
    ckpt_path = _resolve_checkpoint(Path(args.checkpoint), args.which)
    metrics = run_eval(
        args.config,
        overrides,
        ckpt_path,
        split=args.split,
        data_ref=args.data_ref,
        model_ref=args.model_ref,
        preset_ref=args.preset_ref,
    )

    output_dir = ckpt_path
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    if "confusion_matrix" in metrics:
        (output_dir / "confusion_matrix.json").write_text(
            json.dumps(metrics["confusion_matrix"], indent=2), encoding="utf-8"
        )

    print(f"[eval] metrics written to {output_dir}")


if __name__ == "__main__":
    main()
