from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from soion_sentiment.config import load_config
from soion_sentiment.training.run import run_eval


def _require_hf_cache(cfg) -> None:
    kwargs = {"local_files_only": True}
    if cfg.runtime.hf_cache_dir is not None:
        kwargs["cache_dir"] = cfg.runtime.hf_cache_dir
    try:
        AutoTokenizer.from_pretrained(cfg.model.backbone, **kwargs)
        AutoModelForSequenceClassification.from_pretrained(cfg.model.backbone, **kwargs)
    except OSError as e:
        print(
            "[smoke_eval] missing local HF cache for model/tokenizer. "
            "Set runtime.hf_cache_dir or download first.",
            file=sys.stderr,
        )
        raise SystemExit(2) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--keep-artifacts", action="store_true")
    return p.parse_args()


def run_smoke_eval(args: argparse.Namespace) -> Path:
    tmp_dir = None
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        tmp_dir = tempfile.TemporaryDirectory()
        run_dir = Path(tmp_dir.name)
        run_dir.mkdir(parents=True, exist_ok=True)

    resolved_path = run_dir / "resolved_config.yaml"
    if args.run_dir:
        if not resolved_path.exists():
            raise SystemExit(f"[smoke_eval] resolved_config.yaml missing: {resolved_path}")
        config_path = resolved_path
    else:
        config_path = Path(args.config)

    overrides = {"runtime.hf_offline": True, "runtime.device": "cpu"}
    cfg = load_config(config_path, overrides=overrides)
    _require_hf_cache(cfg)

    metrics = run_eval(config_path, overrides, cfg.model.backbone, split="test")

    eval_path = run_dir / "eval_summary.json"
    eval_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if cfg.eval.compute_confusion_matrix and "confusion_matrix" in metrics:
        (run_dir / "confusion_matrix.json").write_text(
            json.dumps(metrics["confusion_matrix"], indent=2), encoding="utf-8"
        )

    if not eval_path.exists():
        raise SystemExit("[smoke_eval] eval_summary.json missing")
    if cfg.eval.compute_confusion_matrix and not (run_dir / "confusion_matrix.json").exists():
        raise SystemExit("[smoke_eval] confusion_matrix.json missing")

    if not args.keep_artifacts and tmp_dir is not None:
        shutil.rmtree(run_dir, ignore_errors=True)
        tmp_dir.cleanup()

    return run_dir


def main() -> None:
    args = parse_args()
    run_dir = run_smoke_eval(args)
    print(f"[smoke_eval] run_dir={run_dir}")


if __name__ == "__main__":
    main()
