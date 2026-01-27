from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import datasets  # huggingface datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "phrasebank_local_v1"
DATA_WRITE_DIR = PROJECT_ROOT / "data" / "processed" / "phrasebank"
_ALLOWED_AGREE = {"allagree", "75agree", "66agree", "50agree"}


def _parse_agrees(s: str) -> list[str]:
    agrees = [x.strip() for x in s.split(",") if x.strip()]
    for a in agrees:
        if a not in _ALLOWED_AGREE:
            raise ValueError(f"Invalid --agree {a!r}. Allowed: {sorted(_ALLOWED_AGREE)}")
    return agrees

def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _compute_split_sizes(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    train_ratio, val_ratio, test_ratio = ratios
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test

def _label_counts(ds: datasets.Dataset, indices: list[int]) -> dict[int, int]:
    # ['text', 'label'] for frasebank
    labels = ds.select(indices)["label"]
    unique, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}

def make_splits(
    agree: str,
    seed: int,
    ratios: tuple[float, float, float],
    *,
    in_root: Path = DATA_DIR,
    out_root: Path = DATA_WRITE_DIR,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    src_path = in_root / f"sentences_{agree}"
    if not src_path.exists():
        raise FileNotFoundError(f"Missing source dataset directory: {src_path}")

    ds = datasets.load_from_disk(str(src_path))
    n_total = len(ds)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)

    n_train, n_val, n_test = _compute_split_sizes(n_total, ratios)

    train_idx = perm[:n_train].tolist()
    val_idx = perm[n_train : n_train + n_val].tolist()
    test_idx = perm[n_train + n_val :].tolist()

    # Hard checks
    s_train, s_val, s_test = set(train_idx), set(val_idx), set(test_idx)
    if (s_train & s_val) or (s_train & s_test) or (s_val & s_test):
        raise RuntimeError("split overlap detected")
    if len(s_train) + len(s_val) + len(s_test) != n_total:
        raise RuntimeError("split coverage mismatch")
    if (s_train | s_val | s_test) != set(range(n_total)):
        raise RuntimeError("split does not cover all indices [0..n_total-1]")

    out_dir = out_root / agree / str(seed)
    if out_dir.exists() and not force:
        raise FileExistsError(f"Output dir exists (immutable): {out_dir} (use --force to overwrite)")
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Metadata: use HF fingerprint from state.json if available via private attr
    fingerprint = getattr(ds, "_fingerprint", None)

    indices_obj = {
        "agree": agree,
        "seed": seed,
        "n_total": n_total,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "splits": {"train": train_idx, "val": val_idx, "test": test_idx},
    }

    info_obj = {
        "agree": agree,
        "seed": seed,
        "n_total": n_total,
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "label_counts": {
            "train": _label_counts(ds, train_idx),
            "val": _label_counts(ds, val_idx),
            "test": _label_counts(ds, test_idx),
        },
        "source_fingerprint": fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    if dry_run:
        print(f"[DRY] agree={agree} seed={seed} n_total={n_total} sizes={info_obj['split_sizes']} fingerprint={fingerprint}")
        return

    (out_dir / "indices.json").write_text(json.dumps(indices_obj, ensure_ascii=False, indent=2))
    (out_dir / "dataset_info.json").write_text(json.dumps(info_obj, ensure_ascii=False, indent=2))

    print(f"OK agree={agree} seed={seed} -> {out_dir}  n_total={n_total}  sizes={info_obj['split_sizes']}  fp={fingerprint}")




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agree", default="66agree,allagree,75agree,50agree", help="Comma-separated: 66agree,allagree,75agree,50agree")
    ap.add_argument("--seeds", default="12345,5768,78516,42", help="Comma-separated ints, e.g. 5768,78516")
    ap.add_argument("--ratios", default="0.8,0.1,0.1", help="train,val,test ratios, sum to 1")
    ap.add_argument("--in-root", default=str(DATA_DIR), help="Root containing phrasebank_local_v1")
    ap.add_argument("--out-root", default=str(DATA_WRITE_DIR), help="Root to write processed splits")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    agrees = _parse_agrees(args.agree)
    seeds = _parse_ints(args.seeds)
    r = tuple(float(x.strip()) for x in args.ratios.split(","))
    if len(r) != 3:
        raise ValueError(f"--ratios must have 3 values, got {args.ratios!r}")
    ratios = (r[0], r[1], r[2])

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    print("Making splits for PhraseBank...")
    for agree in agrees:
        for seed in seeds:
            make_splits(
                agree=agree,
                seed=seed,
                ratios=ratios,
                in_root=in_root,
                out_root=out_root,
                force=args.force,
                dry_run=args.dry_run,
            )

if __name__ == "__main__":
    main()
