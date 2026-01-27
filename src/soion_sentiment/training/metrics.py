from __future__ import annotations

from typing import Any

from soion_sentiment.config import Config


def compute_metrics(preds: list[int], labels: list[int], cfg: Config) -> dict[str, Any]:
    num_labels = len(cfg.model.labels)
    conf = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for p, l in zip(preds, labels):
        if 0 <= l < num_labels and 0 <= p < num_labels:
            conf[l][p] += 1

    total = sum(sum(row) for row in conf)
    correct = sum(conf[i][i] for i in range(num_labels))
    acc = (correct / total) if total > 0 else 0.0

    per_class_f1: dict[str, float] = {}
    f1s: list[float] = []
    for i, label in enumerate(cfg.model.labels):
        tp = conf[i][i]
        fp = sum(conf[r][i] for r in range(num_labels) if r != i)
        fn = sum(conf[i][c] for c in range(num_labels) if c != i)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_f1[f"f1_{label}"] = f1
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    metrics: dict[str, Any] = {"acc": acc, "macro_f1": macro_f1}
    metrics.update(per_class_f1)
    if cfg.eval.compute_confusion_matrix:
        metrics["confusion_matrix"] = conf
    return metrics
