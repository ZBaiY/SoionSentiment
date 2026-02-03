# Docs

Optional notes, references, and design docs.
Keep training evidence in runs/ and reports/.

## Configs Usage Guide

The configuration system uses a registry-based scheme under `configs/`. The resolved config (after all overlays) is the single source of truth used by training and evaluation.

### Layout

```
configs/
  base.yaml
  data/
    phrasebank_66agree.yaml
  models/
    deberta_v3_base.yaml
  presets/
    baseline.yaml
```

### Resolution Order (strict)

1. `configs/base.yaml`
2. `configs/data/<data_ref>.yaml`
3. `configs/models/<model_ref>.yaml`
4. `configs/presets/<preset_ref>.yaml` (if not null)
5. CLI overrides (`--override key=value`)

### Base Pointers

`configs/base.yaml` must define the registry pointers:

```yaml
data_ref: phrasebank_66agree
model_ref: deberta_v3_base
preset_ref: baseline
```

### What Goes Where

- `base.yaml`: global defaults (training/optim/scheduler/eval/logging/runtime, plus general data fields).
- `configs/data/*.yaml`: dataset-specific settings (e.g., `data.name`, `data.agree`, `data.split_protocol`).
- `configs/models/*.yaml`: model backbone + labels + dropout overrides.
- `configs/presets/*.yaml`: experiment overlays (training/optim/scheduler tweaks). Can be empty.

### CLI Overrides

Override any field with nested keys:

```
--override training.epochs=1 --override optim.lr=2e-5
```

### Examples

Use defaults from base:

```
python scripts/train.py --config configs/base.yaml
```

Select a different registry entry:

```
python scripts/train.py \
  --config configs/base.yaml \
  --data-ref phrasebank_66agree \
  --model-ref deberta_v3_base \
  --preset-ref baseline
```

Add experiment overrides:

```
python scripts/train.py \
  --config configs/base.yaml \
  --preset-ref baseline \
  --override training.epochs=3 \
  --override optim.lr=3e-5
```

## Eval Mistake Logs

Evaluation can optionally write a `sample_mistake.jsonl` under the run directory with misclassified examples.
Configure via `eval.mistake_path` (null disables), `eval.mistake_max_n` (null = all), and `eval.mistake_seed`.
