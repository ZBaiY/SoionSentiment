from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _ema(values: list[float], alpha: float) -> list[float]:
    out = []
    ema = None
    for v in values:
        if ema is None:
            ema = v
        else:
            ema = alpha * v + (1 - alpha) * ema
        out.append(ema)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--smooth", type=float, default=None, help="EMA alpha, e.g. 0.1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _read_jsonl(run_dir / "train.jsonl")
    eval_rows = _read_jsonl(run_dir / "metrics.jsonl")

    train_steps = [r.get("step") for r in train_rows if r.get("split") == "train"]
    train_loss = [r.get("loss") for r in train_rows if r.get("split") == "train"]
    train_lr = [r.get("lr") for r in train_rows if r.get("split") == "train"]
    train_grad = [r.get("grad_norm") for r in train_rows if r.get("split") == "train"]
    train_mps_drv = [r.get("mps_driver_gb") for r in train_rows if r.get("split") == "train"]
    train_rss = [r.get("rss_gb") for r in train_rows if r.get("split") == "train"]
    train_tps = [r.get("tokens_per_s") for r in train_rows if r.get("split") == "train"]

    if args.smooth is not None and train_loss:
        train_loss = _ema(train_loss, args.smooth)

    if train_steps and train_loss:
        plt.figure()
        plt.plot(train_steps, train_loss)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("train loss vs step")
        plt.tight_layout()
        plt.savefig(out_dir / "loss_vs_step.png")
        plt.close()

    if train_steps and train_lr:
        plt.figure()
        plt.plot(train_steps, train_lr)
        plt.xlabel("step")
        plt.ylabel("lr")
        plt.title("lr vs step")
        plt.tight_layout()
        plt.savefig(out_dir / "lr_vs_step.png")
        plt.close()

    if train_steps and train_mps_drv:
        plt.figure()
        plt.plot(train_steps, train_mps_drv)
        plt.xlabel("step")
        plt.ylabel("mps_driver_gb")
        plt.title("mps driver vs step")
        plt.tight_layout()
        plt.savefig(out_dir / "mps_driver_vs_step.png")
        plt.close()

    eval_epochs = [r.get("epoch") for r in eval_rows if r.get("split") == "val" and "macro_f1" in r]
    eval_f1 = [r.get("macro_f1") for r in eval_rows if r.get("split") == "val" and "macro_f1" in r]
    if eval_epochs and eval_f1:
        plt.figure()
        plt.plot(eval_epochs, eval_f1)
        plt.xlabel("epoch")
        plt.ylabel("macro_f1")
        plt.title("macro_f1 vs epoch")
        plt.tight_layout()
        plt.savefig(out_dir / "macro_f1_vs_epoch.png")
        plt.close()

    # Write curves.csv
    csv_path = out_dir / "curves.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("step,epoch,loss,lr,grad_norm,mps_driver_gb,rss_gb,tokens_per_s\n")
        for r in train_rows:
            if r.get("split") != "train":
                continue
            f.write(
                f"{r.get('step')},{r.get('epoch')},{r.get('loss')},{r.get('lr')},{r.get('grad_norm')},{r.get('mps_driver_gb')},{r.get('rss_gb')},{r.get('tokens_per_s')}\n"
            )


if __name__ == "__main__":
    main()
