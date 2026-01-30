#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from soion_sentiment.analyze_train.runner import run_analysis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/analyze_train.yaml"))
    ap.add_argument("--runs-root", type=Path, default=None)
    args = ap.parse_args()

    run_analysis(None, config_path=args.config, runs_root=args.runs_root)


if __name__ == "__main__":
    main()
