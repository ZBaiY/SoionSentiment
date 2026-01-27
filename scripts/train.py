from __future__ import annotations

import argparse
from pathlib import Path

from soion_sentiment.data.phrasebank import load_phrasebank_splits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data-entry contract knobs (ONLY these)
    p.add_argument("--agree", type=str, default="sentences_66agree")
    p.add_argument("--seed", type=int, required=True)

    # Optional path overrides (keep default = repo conventions)
    p.add_argument("--hf-dataset-root", type=str, default=None)
    p.add_argument("--processed-root", type=str, default=None)

    # The rest of your training knobs stay here (model, lr, epochs, batch, etc.)
    p.add_argument("--model-name", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    ds = load_phrasebank_splits(
        seed=args.seed,
        agree=args.agree,
        hf_dataset_root=args.hf_dataset_root,
        processed_root=args.processed_root,
    )

    # From here on, train.py MUST treat ds as the only truth:
    # ds["train"], ds["val"], ds["test"] each have exactly {"text","label"}.

    # ---- Example wiring point (keep your existing pipeline below) ----
    train_ds = ds["train"]
    val_ds = ds["val"]
    test_ds = ds["test"]

    # TODO: tokenizer/map/collator/trainer/eval/checkpoints
    # IMPORTANT: do NOT do any splitting here.
    # ---------------------------------------------------------------

    print(
        f"[data] agree={args.agree} seed={args.seed} "
        f"train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)} "
        f"cols={train_ds.column_names}"
    )


if __name__ == "__main__":
    main()