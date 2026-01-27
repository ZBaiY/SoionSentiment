from __future__ import annotations

from transformers import AutoConfig, AutoModelForSequenceClassification

from soion_sentiment.config import Config


def build_model(cfg: Config):
    labels = cfg.model.labels
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    model_cfg = AutoConfig.from_pretrained(
        cfg.model.backbone,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )
    if cfg.model.dropout_override is not None:
        if hasattr(model_cfg, "hidden_dropout_prob"):
            model_cfg.hidden_dropout_prob = cfg.model.dropout_override
        if hasattr(model_cfg, "attention_probs_dropout_prob"):
            model_cfg.attention_probs_dropout_prob = cfg.model.dropout_override

    return AutoModelForSequenceClassification.from_pretrained(cfg.model.backbone, config=model_cfg)
