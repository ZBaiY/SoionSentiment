from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from soion_sentiment.config import load_config
from soion_sentiment.training.run import run_eval, run_train


def _require_hf_cache(cfg) -> None:
    kwargs = {"local_files_only": True}
    if cfg.runtime.hf_cache_dir is not None:
        kwargs["cache_dir"] = cfg.runtime.hf_cache_dir
    try:
        AutoTokenizer.from_pretrained(cfg.model.backbone, **kwargs)
        AutoModelForSequenceClassification.from_pretrained(cfg.model.backbone, **kwargs)
    except OSError as e:
        print(
            "[smoke_train] missing local HF cache for model/tokenizer. "
            "Set runtime.hf_cache_dir or download first.",
            file=sys.stderr,
        )
        raise SystemExit(2) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--data-ref", type=str, default=None)
    p.add_argument("--model-ref", type=str, default=None)
    p.add_argument("--preset-ref", type=str, default=None)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--keep-artifacts", action="store_true")
    return p.parse_args()


def run_smoke_train(args: argparse.Namespace) -> Path:
    overrides = {
        "runtime.hf_offline": True,
        "runtime.device": "cpu",
        "training.epochs": 1,
        "training.batch_size": 2,
        "training.eval_batch_size": 2,
        "training.grad_accum_steps": 1,
        "training.max_steps": 8,
        "data.max_train_samples": 32,
        "data.max_eval_samples": 32,
        "data.max_test_samples": 32,
        "logging.run_name": "smoke_train",
        "logging.save_best": False,
        "logging.save_last": False,
    }
    tmp_dir = None
    if args.run_dir is not None:
        overrides["logging.run_dir"] = args.run_dir
    else:
        tmp_dir = tempfile.TemporaryDirectory()
        overrides["logging.run_dir"] = tmp_dir.name

    cfg = load_config(
        args.config,
        overrides=None,
        data_ref=args.data_ref,
        model_ref=args.model_ref,
        preset_ref=args.preset_ref,
    )
    _require_hf_cache(cfg)
    if cfg.tokenizer.use_fast:
        warnings.filterwarnings(
            "ignore",
            message=".*byte fallback option is not implemented in the fast tokenizers.*",
            category=UserWarning,
        )
    print("[smoke_train] note: classifier/pooler init warnings are expected for sequence classification")

    result = run_train(
        args.config,
        overrides,
        data_ref=args.data_ref,
        model_ref=args.model_ref,
        preset_ref=args.preset_ref,
    )
    run_dir = Path(result["run_dir"])

    if not run_dir.exists():
        raise SystemExit("[smoke_train] run_dir missing")
    resolved_path = run_dir / "resolved_config.yaml"
    if not resolved_path.exists() or resolved_path.stat().st_size == 0:
        raise SystemExit("[smoke_train] resolved_config.yaml missing or empty")
    manifest_path = run_dir / "data_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("[smoke_train] data_manifest.json missing")
    json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        raise SystemExit("[smoke_train] metrics.jsonl missing or empty")
    with metrics_path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line.strip():
            raise SystemExit("[smoke_train] metrics.jsonl has no JSON lines")
        json.loads(first_line)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit("[smoke_train] summary.json missing")
    json.loads(summary_path.read_text(encoding="utf-8"))

    eval_overrides = {"runtime.hf_offline": True, "runtime.device": "cpu"}
    eval_metrics = run_eval(
        resolved_path,
        eval_overrides,
        cfg.model.backbone,
        split="test",
    )
    eval_path = run_dir / "eval_summary.json"
    eval_path.write_text(json.dumps(eval_metrics, indent=2, sort_keys=True), encoding="utf-8")
    if cfg.eval.compute_confusion_matrix and "confusion_matrix" in eval_metrics:
        (run_dir / "confusion_matrix.json").write_text(
            json.dumps(eval_metrics["confusion_matrix"], indent=2), encoding="utf-8"
        )
    if not eval_path.exists():
        raise SystemExit("[smoke_train] eval_summary.json missing")
    json.loads(eval_path.read_text(encoding="utf-8"))
    if cfg.eval.compute_confusion_matrix and not (run_dir / "confusion_matrix.json").exists():
        raise SystemExit("[smoke_train] confusion_matrix.json missing")

    print(
        "[smoke_train] ARTIFACTS OK: "
        "resolved_config.yaml, data_manifest.json, metrics.jsonl, summary.json, eval_summary.json"
    )

    if tmp_dir is not None and not args.keep_artifacts:
        shutil.rmtree(run_dir, ignore_errors=True)
        tmp_dir.cleanup()

    return run_dir


def main() -> None:
    args = parse_args()
    run_dir = run_smoke_train(args)
    print(f"[smoke_train] run_dir={run_dir}")


if __name__ == "__main__":
    main()
