from __future__ import annotations

import argparse

from soion_sentiment.config import parse_overrides
from soion_sentiment.training.run import run_train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--data-ref", type=str, default=None)
    p.add_argument("--model-ref", type=str, default=None)
    p.add_argument("--preset-ref", type=str, default=None)
    p.add_argument("--override", action="append", default=[], help="Override config: key=value")
    p.add_argument("--run-name", type=str, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = parse_overrides(args.override)
    if args.run_name is not None:
        overrides["logging.run_name"] = args.run_name
    result = run_train(
        args.config,
        overrides,
        data_ref=args.data_ref,
        model_ref=args.model_ref,
        preset_ref=args.preset_ref,
    )
    print(f"[train] run_dir={result['run_dir']}")


if __name__ == "__main__":
    main()
