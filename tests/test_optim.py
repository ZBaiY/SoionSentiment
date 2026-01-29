import pytest
import torch

from soion_sentiment.config import load_config
from soion_sentiment.training.loop import _compute_training_steps
from soion_sentiment.training.optim import build_optimizer, split_decay_no_decay_params


class CustomNorm(torch.nn.LayerNorm):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.ln = torch.nn.LayerNorm(4)
        self.custom_norm = CustomNorm(4)


def _names_in_group(model: torch.nn.Module, group) -> set[str]:
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    return {name_by_id[id(p)] for p in group["params"]}


def test_build_optimizer_decay_groups() -> None:
    cfg = load_config("configs/base.yaml", overrides={"optim.weight_decay": 0.1})
    model = ToyModel()
    opt = build_optimizer(cfg, model)

    assert len(opt.param_groups) == 2
    decay_group, no_decay_group = opt.param_groups
    decay_names = _names_in_group(model, decay_group)
    no_decay_names = _names_in_group(model, no_decay_group)

    assert "linear.weight" in decay_names
    assert "linear.bias" in no_decay_names
    assert "ln.weight" in no_decay_names
    assert "ln.bias" in no_decay_names
    assert "custom_norm.weight" in no_decay_names
    assert "custom_norm.bias" in no_decay_names
    assert "custom_norm.weight" not in decay_names


def test_split_decay_hf_path_if_available() -> None:
    try:
        from transformers.trainer_pt_utils import get_parameter_names  # noqa: F401
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS  # noqa: F401
    except Exception:
        pytest.skip("transformers HF utilities not available")

    model = ToyModel()
    decay_params, no_decay_params, debug_info = split_decay_no_decay_params(model, decay_embeddings=True)
    assert debug_info["hf_used"] is True
    decay_names = _names_in_group(model, {"params": decay_params})
    no_decay_names = _names_in_group(model, {"params": no_decay_params})
    assert "linear.weight" in decay_names
    assert "linear.bias" in no_decay_names
    assert "ln.weight" in no_decay_names


def test_compute_training_steps() -> None:
    cfg = load_config(
        "configs/base.yaml",
        overrides={"training.epochs": 2, "training.grad_accum_steps": 4, "training.max_steps": None},
    )
    steps_per_epoch, total_steps, total_epochs = _compute_training_steps(cfg, train_loader_len=10)
    assert steps_per_epoch == 3
    assert total_steps == 6
    assert total_epochs == 2

    cfg = load_config(
        "configs/base.yaml",
        overrides={"training.epochs": 2, "training.grad_accum_steps": 4, "training.max_steps": 5},
    )
    steps_per_epoch, total_steps, total_epochs = _compute_training_steps(cfg, train_loader_len=10)
    assert steps_per_epoch == 3
    assert total_steps == 5
    assert total_epochs == 2
