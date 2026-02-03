from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from soion_sentiment.config import load_config
from soion_sentiment.eval import analyze_eval_suite, run_eval_suite
from soion_sentiment.eval.config import EvalConfig, EvalSuiteConfig, LoggingConfig, RunsConfig, SuiteEntry


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


def test_eval_suite_smoke(tmp_path: Path) -> None:
    model_name = "hf-internal-testing/tiny-random-bert"
    if not _model_cached(model_name):
        pytest.skip("tiny model not cached; skipping eval suite smoke test")

    run_root = tmp_path / "runs"
    run_dir = run_root / "run_eval_suite"
    run_dir.mkdir(parents=True, exist_ok=True)

    hf_root = tmp_path / "hf_snapshot"
    proc_root = tmp_path / "processed"
    agree = "sentences_66agree"
    indices_agree = "66agree"
    seed = 123

    _make_hf_snapshot(hf_root, agree=agree, n=12)
    _write_indices(
        proc_root,
        agree=indices_agree,
        seed=seed,
        indices={"train": [0, 1, 2, 3, 4, 5], "val": [6, 7, 8], "test": [9, 10, 11]},
    )

    base_cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    base_cfg["seed"] = seed
    base_cfg["data_ref"] = None
    base_cfg["model_ref"] = None
    base_cfg["preset_ref"] = None
    base_cfg["data"]["agree"] = agree
    base_cfg["data"]["hf_dataset_root"] = str(hf_root)
    base_cfg["data"]["processed_root"] = str(proc_root)
    base_cfg["data"]["max_eval_samples"] = None
    base_cfg["model"]["backbone"] = model_name
    base_cfg["training"]["eval_batch_size"] = 2
    base_cfg["runtime"]["hf_offline"] = True
    base_cfg["logging"]["run_dir"] = str(run_root)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=True), encoding="utf-8")
    cfg = load_config(cfg_path)
    (run_dir / "resolved_config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")

    ckpt_dir = run_dir / "best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
    model.save_pretrained(str(ckpt_dir))

    eval_cfg = EvalSuiteConfig(
        runs=RunsConfig(run_dir=str(run_dir), checkpoint="best"),
        suite=[
            SuiteEntry(name="pb66_a", data_ref="phrasebank_66agree", split="validation"),
            SuiteEntry(name="pb66_b", data_ref="phrasebank_66agree", split="validation"),
        ],
        eval=EvalConfig(
            batch_size=2,
            max_samples=8,
            metrics=["loss", "macro_f1", "acc"],
            logging=LoggingConfig(
                out_dir=str(run_dir / "eval"),
                out_csv="eval_suite.csv",
                out_jsonl="eval_suite.jsonl",
                write_index_md=True,
                write_confusion_matrices=False,
                write_plots=False,
            ),
        ),
    )

    records, out_dir = run_eval_suite(eval_cfg)
    analyze_eval_suite(
        out_dir / "eval_suite.jsonl",
        out_dir,
        out_csv=out_dir / "eval_suite.csv",
        write_index_md=True,
        write_plots=False,
        write_confusion_matrices=False,
    )

    assert (out_dir / "eval_suite.jsonl").exists()
    assert (out_dir / "eval_suite.csv").exists()
    assert (out_dir / "index.md").exists()

    with (out_dir / "eval_suite.jsonl").open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == len(eval_cfg.suite)
    assert all("eval_name" in r for r in rows)
    for row in rows:
        metrics = row.get("metrics", {})
        assert "macro_f1" in metrics
        assert "acc" in metrics
        assert "loss" in metrics
