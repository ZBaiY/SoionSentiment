# scripts/smoke_infer.py

import os
from pathlib import Path

# Cache HF models/tokenizers inside the repo (so the project is self-contained).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face hub cache layout
HF_HUB_CACHE = HF_CACHE_DIR / "hub"
HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)

# Prefer environment variables (covers transformers + huggingface_hub).
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "microsoft/deberta-v3-base"
NUM_LABELS = 3  # 先假设三分类，占位即可

def main():
    print("torch:", torch.__version__)
    print("mps available:", torch.backends.mps.is_available())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(HF_HUB_CACHE), use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        cache_dir=str(HF_HUB_CACHE),
    )

    # Select execution device:
    # - Use MPS (Apple Silicon GPU) if available
    # - Fallback to CPU otherwise
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Move all model parameters and buffers to the selected device.
    # This must match the device of input tensors to avoid device-mismatch errors.
    model.to(device)

    # Switch the model to evaluation mode.
    # This disables training-specific behaviors such as Dropout
    # and ensures deterministic outputs during inference. (batchnorm is not used here but when used, it is usually exponential moving average batch statistics vs. single batch statistics)
    model.eval()

    texts = [
        "Bitcoin price surged after ETF approval.",
        "The company reported a significant loss this quarter.",
    ]

    # Tokenize a batch of variable-length texts.
    # Padding makes all sequences the same length so they can be stacked into a tensor.
    # Truncation + max_length bound the sequence length to control memory and compute.
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Disable autograd during inference.
    # No computation graph is built, which reduces memory usage and speeds up execution.
    # This explicitly marks the forward pass as non-training.
    with torch.no_grad():
        outputs = model(**inputs)

    print("logits shape:", outputs.logits.shape)
    print("logits:", outputs.logits)


if __name__ == "__main__":
    main()