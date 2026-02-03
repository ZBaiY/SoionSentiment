from __future__ import annotations

from pathlib import Path

from soion_sentiment.analysis.mistakes_io import load_mistakes_jsonl, write_mistakes_jsonl


def _fake_mistakes(n: int) -> list[dict]:
    return [
        {
            "run_id": "run_1",
            "split": "pb66",
            "example_id": f"ex_{i}",
            "y_true": "negative",
            "y_pred": "positive",
            "probs": [0.1, 0.2, 0.7],
            "margin": 0.5,
            "text": f"sample {i}",
        }
        for i in range(n)
    ]


def test_write_all_mistakes(tmp_path: Path) -> None:
    path = tmp_path / "sample_mistake.jsonl"
    mistakes = _fake_mistakes(3)
    write_mistakes_jsonl(mistakes, path, max_n=None, seed=1234, run_id="run_1", split="pb66")
    rows = load_mistakes_jsonl(path)
    assert len(rows) == 3


def test_write_sampled_mistakes_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "sample_mistake.jsonl"
    mistakes = _fake_mistakes(5)
    write_mistakes_jsonl(mistakes, path, max_n=2, seed=42, run_id="run_1", split="pb66")
    rows_a = load_mistakes_jsonl(path)

    path_b = tmp_path / "sample_mistake_b.jsonl"
    write_mistakes_jsonl(mistakes, path_b, max_n=2, seed=42, run_id="run_1", split="pb66")
    rows_b = load_mistakes_jsonl(path_b)

    assert len(rows_a) == 2
    assert len(rows_b) == 2
    assert [r["example_id"] for r in rows_a] == [r["example_id"] for r in rows_b]


def test_schema_keys_present(tmp_path: Path) -> None:
    path = tmp_path / "sample_mistake.jsonl"
    mistakes = _fake_mistakes(1)
    write_mistakes_jsonl(mistakes, path, max_n=None, seed=1234, run_id="run_1", split="pb66")
    row = load_mistakes_jsonl(path)[0]
    for key in ["run_id", "split", "example_id", "y_true", "y_pred", "probs", "margin", "text"]:
        assert key in row
