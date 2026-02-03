from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from soion_sentiment.config import Config
from soion_sentiment.training.metrics import compute_metrics


@dataclass
class ECEBins:
    n_bins: int
    counts: np.ndarray
    conf_sum: np.ndarray
    acc_sum: np.ndarray


def init_ece_bins(n_bins: int) -> ECEBins:
    return ECEBins(
        n_bins=n_bins,
        counts=np.zeros(n_bins, dtype=np.int64),
        conf_sum=np.zeros(n_bins, dtype=np.float64),
        acc_sum=np.zeros(n_bins, dtype=np.float64),
    )


def update_ece_bins(state: ECEBins, confidences: np.ndarray, correct: np.ndarray) -> None:
    if confidences.size == 0:
        return
    bins = np.linspace(0.0, 1.0, state.n_bins + 1)
    bin_ids = np.digitize(confidences, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, state.n_bins - 1)
    for b in range(state.n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        state.counts[b] += int(mask.sum())
        state.conf_sum[b] += float(confidences[mask].sum())
        state.acc_sum[b] += float(correct[mask].sum())


def finalize_ece(state: ECEBins) -> tuple[float | None, list[dict[str, Any]]]:
    total = int(state.counts.sum())
    if total == 0:
        return None, []
    ece = 0.0
    bins: list[dict[str, Any]] = []
    for b in range(state.n_bins):
        count = int(state.counts[b])
        if count == 0:
            acc = 0.0
            conf = 0.0
        else:
            acc = float(state.acc_sum[b] / count)
            conf = float(state.conf_sum[b] / count)
        ece += abs(acc - conf) * (count / total)
        bins.append({"bin": b, "count": count, "acc": acc, "conf": conf})
    return float(ece), bins


def compute_classification_metrics(
    preds: list[int],
    labels: list[int],
    cfg: Config,
    *,
    include_confusion_matrix: bool,
) -> dict[str, Any]:
    if include_confusion_matrix and not cfg.eval.compute_confusion_matrix:
        from dataclasses import replace

        cfg = replace(cfg, eval=replace(cfg.eval, compute_confusion_matrix=True))
    return compute_metrics(preds, labels, cfg)
