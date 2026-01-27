from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from soion_sentiment.config import load_config
from soion_sentiment.data.collate import build_dataloader
from soion_sentiment.data.phrasebank import load_phrasebank_splits
from soion_sentiment.data.tokenize import build_tokenizer, tokenize_dataset
from soion_sentiment.training.manifest import build_data_manifest, write_data_manifest
from soion_sentiment.training.run import run_train


def _make_hf_snapshot(root: Path, *, agree: str = "sentences_66agree", n: int = 12) -> None:
    base = Dataset.from_dict(
        {
            "text": [f"s{i}" for i in range(n)],
            "label": [i % 3 for i in range(n)],
        }
    )
    DatasetDict({agree: base}).save_to_disk(str(root))


def _write_indices(proc_root: Path, *, agree: str, seed: int, indices: dict) -> None:
    d = proc_root / agree / str(seed)
    d.mkdir(parents=True, exist_ok=True)
    (d / "indices.json").write_text(json.dumps(indices), encoding="utf-8")


def _model_cached(model_name: str) -> bool:
    try:
        AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        return True
    except OSError:
        return False


def test_smoke_training(tmp_path: Path) -> None:
    model_name = "hf-internal-testing/tiny-random-bert"
    if not _model_cached(model_name):
        pytest.skip("tiny model not cached; skipping smoke training")

    hf_root = tmp_path / "hf_snapshot"
    proc_root = tmp_path / "processed"
    agree = "sentences_66agree"
    seed = 123

    _make_hf_snapshot(hf_root, agree=agree, n=12)
    _write_indices(
        proc_root,
        agree=agree,
        seed=seed,
        indices={"train": [0, 1, 2, 3, 4, 5], "val": [6, 7], "test": [8, 9]},
    )

    base_cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    base_cfg["seed"] = seed
    base_cfg["data"]["hf_dataset_root"] = str(hf_root)
    base_cfg["data"]["processed_root"] = str(proc_root)
    base_cfg["data"]["max_train_samples"] = 6
    base_cfg["data"]["max_eval_samples"] = 2
    base_cfg["data"]["max_test_samples"] = 2
    base_cfg["model"]["backbone"] = model_name
    base_cfg["training"]["epochs"] = 1
    base_cfg["training"]["batch_size"] = 2
    base_cfg["training"]["eval_batch_size"] = 2
    base_cfg["logging"]["log_every_steps"] = 1
    base_cfg["logging"]["run_dir"] = str(tmp_path / "runs")
    base_cfg["scheduler"]["warmup_ratio"] = 0.0

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=True), encoding="utf-8")

    result = run_train(cfg_path, overrides=None)
    run_dir = Path(result["run_dir"])
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "last" / "trainer_state.pt").exists()
    assert (run_dir / "best" / "trainer_state.pt").exists()


def test_smoke_components(tmp_path: Path) -> None:
    model_name = "hf-internal-testing/tiny-random-bert"
    if not _model_cached(model_name):
        pytest.skip("tiny model not cached; skipping component smoke test")

    repo_root = Path(__file__).resolve().parents[1]
    work_root = repo_root / "data" / f"tmp_smoke_{tmp_path.name}"
    hf_root = work_root / "hf_snapshot"
    proc_root = work_root / "processed"
    agree = "sentences_66agree"
    seed = 321

    work_root.mkdir(parents=True, exist_ok=True)
    try:
        _make_hf_snapshot(hf_root, agree=agree, n=12)
        _write_indices(
            proc_root,
            agree=agree,
            seed=seed,
            indices={"train": [0, 1, 2, 3, 4, 5], "val": [6, 7], "test": [8, 9]},
        )

        base_cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
        base_cfg["seed"] = seed
        base_cfg["data"]["hf_dataset_root"] = str(hf_root)
        base_cfg["data"]["processed_root"] = str(proc_root)
        base_cfg["model"]["backbone"] = model_name
        base_cfg["training"]["batch_size"] = 2
        base_cfg["training"]["eval_batch_size"] = 2

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=True), encoding="utf-8")
        cfg = load_config(cfg_path)

        tokenizer = build_tokenizer(cfg)
        ds = load_phrasebank_splits(
            seed=cfg.seed,
            agree=cfg.data.agree or "sentences_66agree",
            hf_dataset_root=cfg.data.hf_dataset_root,
            processed_root=cfg.data.processed_root,
        )
        ds = tokenize_dataset(cfg, ds, tokenizer)
        loader = build_dataloader(cfg, ds["train"], tokenizer, "train")

        batch = next(iter(loader))
        assert "input_ids" in batch and "attention_mask" in batch

        manifest = build_data_manifest(cfg)
        out_path = tmp_path / "data_manifest.json"
        write_data_manifest(out_path, manifest)
        assert out_path.exists()
        assert manifest["dataset"] == cfg.data.name
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def test_config_registry_merge_and_hash(tmp_path: Path) -> None:
    base_cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    base_cfg["data_ref"] = "phrasebank_66agree"
    base_cfg["model_ref"] = "deberta_v3_base"
    base_cfg["preset_ref"] = "baseline"
    base_cfg["data"]["name"] = "fake_dataset"
    base_cfg["data"]["agree"] = None
    base_cfg["model"]["backbone"] = "fake-backbone"
    base_cfg["model"]["labels"] = ["a", "b"]

    cfg_path = tmp_path / "base.yaml"
    cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=True), encoding="utf-8")

    cfg_a = load_config(cfg_path)
    cfg_b = load_config(cfg_path)
    assert cfg_a.data.name == "phrasebank"
    assert cfg_a.data.agree == "sentences_66agree"
    assert cfg_a.model.backbone == "microsoft/deberta-v3-base"
    assert cfg_a.model.labels == ["negative", "neutral", "positive"]
    assert cfg_a.config_hash() == cfg_b.config_hash()

    cfg_c = load_config(cfg_path, overrides={"training.epochs": cfg_a.training.epochs + 1})
    assert cfg_c.config_hash() != cfg_a.config_hash()


def test_hf_offline_cache_wiring(tmp_path: Path) -> None:
    base_cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    base_cfg["runtime"]["hf_offline"] = True
    base_cfg["runtime"]["hf_cache_dir"] = str(tmp_path / "hf_cache")

    cfg_path = tmp_path / "base.yaml"
    cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=True), encoding="utf-8")

    cfg = load_config(cfg_path)
    assert cfg.runtime.hf_offline is True
    assert cfg.runtime.hf_cache_dir == str(tmp_path / "hf_cache")
