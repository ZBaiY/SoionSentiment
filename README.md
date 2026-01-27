

# SoionSentiment

**SoionSentiment** is a training-focused NLP research project under **SoionLab**.  
It uses sentiment classification as a *controlled task* to study **supervised fine-tuning dynamics**, **generalization**, and **robustness** in medium-scale encoder models.

This repository is **not** an application or monitoring system.  
The task is intentionally simple; the focus is on *training behavior and evidence*.

---

## Scope and Positioning

- Parent project: **SoionLab** (research engine / experimental platform)
- Role of this repo: **standalone training module**, later integrable into SoionLab
- Narrative focus:
  - training stability and convergence,
  - generalization under non-i.i.d. splits,
  - regularization and class imbalance,
  - reproducibility and auditability.

Sentiment classification is treated purely as a **training carrier**, not as an end product.

---

## Models and Training Regime

- Backbone models:
  - DeBERTa-v3-base
  - RoBERTa-base (or equivalent)
- Parameter scale: ~100–300M
- Training mode:
  - **full-parameter fine-tuning**
  - no LoRA / adapters / PEFT
- Optimizer and scheduling:
  - AdamW
  - warmup + cosine decay
- Hardware target:
  - single-device training (Apple M2 Pro class)

---

## Experimental Requirements

This project emphasizes **training evidence over task performance**.

Minimum required artifacts:

1. **Training dynamics**
   - train / validation loss vs. step or epoch
   - main metric (F1 / accuracy) vs. step or epoch
   - discussion of loss–metric alignment or divergence

2. **Ablation studies (≥3, motivation required)**
   - learning-rate schedule variants
   - regularization (weight decay, dropout, label smoothing)
   - class imbalance handling (reweighting, sampling, thresholds)

3. **Data splitting protocol**
   - time-based split preferred
   - otherwise cross-domain or cross-source split
   - random split is not used as the main experiment

4. **Reproducibility package**
   - fixed random seeds
   - configuration files (`configs/*.yaml`)
   - data manifest with hashes
   - exact commit SHA recorded per run

5. **Failure analysis**
   - structured error buckets (length, negation, domain terms, domain shift)
   - qualitative examples tied back to training choices

---

## Repository Structure (Planned)

```
soion-sentiment/
├── src/                # training and evaluation code
├── scripts/            # entry points (train / eval / analysis)
├── configs/            # experiment configurations
├── runs/               # outputs (metrics, checkpoints) [gitignored]
├── reports/            # analysis notes and conclusions
├── data/               # datasets (gitignored)
└── README.md
```

---

## Status

This repository is under **active development**.  
Results, ablations, and analysis will be added incrementally as training progresses.

The emphasis is on *clean experimental history*, not rapid iteration or leaderboard results.

---

## Training Entry Point

All training/eval/data/model parameters are driven by `configs/base.yaml`, with optional CLI overrides.

Run training:

```bash
python scripts/train.py --config configs/base.yaml
```

Override config values:

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --override training.epochs=1 \
  --override training.batch_size=8 \
  --override model.backbone=microsoft/deberta-v3-base
```

Run evaluation (best/last checkpoint):

```bash
python scripts/eval.py --config configs/base.yaml --checkpoint runs/<run_dir> --which best
```

Artifacts per run (under `runs/`):

- `resolved_config.yaml` (fully resolved config)
- `env.txt` (python/torch/transformers versions)
- `data_manifest.json` (file list + hashes)
- `metrics.jsonl` (train/eval metrics)
- `best/` and `last/` checkpoints (model/tokenizer + `trainer_state.pt`)
