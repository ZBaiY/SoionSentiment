from __future__ import annotations

import argparse
from pathlib import Path

from soion_sentiment.eval import analyze_eval_suite, load_eval_suite_config, run_eval_suite


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/eval_suite.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_eval_suite_config(args.config)
    records, out_dir = run_eval_suite(cfg)

    jsonl_path = out_dir / cfg.eval.logging.out_jsonl
    out_csv = out_dir / cfg.eval.logging.out_csv
    analyze_eval_suite(
        jsonl_path,
        out_dir,
        out_csv=out_csv,
        write_index_md=cfg.eval.logging.write_index_md,
        write_plots=cfg.eval.logging.write_plots,
        write_confusion_matrices=cfg.eval.logging.write_confusion_matrices,
    )

    print(f"[eval_suite] wrote {len(records)} records to {out_dir}")


if __name__ == "__main__":
    main()
