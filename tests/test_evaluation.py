import json
from pathlib import Path

import pytest

from evaluation import evaluate_retrieval, hit_at_k, load_dataset, match_evidence, recall_at_k, reciprocal_rank

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_metrics_on_ranked_matches():
    matches = [set(), {1}, {0, 1}]

    assert hit_at_k(matches, k=1) == 0.0
    assert hit_at_k(matches, k=2) == 1.0
    assert recall_at_k(matches, num_relevant=2, k=2) == 0.5
    assert recall_at_k(matches, num_relevant=2, k=3) == 1.0
    assert reciprocal_rank(matches) == 0.5
    assert reciprocal_rank([set(), set()]) == 0.0


def test_recall_rejects_empty_ground_truth():
    with pytest.raises(ValueError):
        recall_at_k([{0}], num_relevant=0, k=1)


def test_match_evidence_ignores_case_and_whitespace():
    passages = ["Blank holder\n force   controls WRINKLING.", "Unrelated text."]

    matches = match_evidence(passages, ["blank holder force", "wrinkling"])

    assert matches == [{0, 1}, set()]


def test_load_dataset_joins_questions_with_ground_truth(tmp_path):
    questions = tmp_path / "questions.jsonl"
    truth = tmp_path / "ground_truth.jsonl"
    questions.write_text(json.dumps({"id": "a", "question": "Q?"}) + "\n\n", encoding="utf-8")
    truth.write_text(json.dumps({"id": "a", "answer": "A.", "evidence": ["phrase"]}) + "\n", encoding="utf-8")

    examples = load_dataset(questions, truth)

    assert len(examples) == 1
    assert examples[0].question == "Q?"
    assert examples[0].evidence == ["phrase"]


def test_load_dataset_fails_on_missing_ground_truth(tmp_path):
    questions = tmp_path / "questions.jsonl"
    truth = tmp_path / "ground_truth.jsonl"
    questions.write_text(json.dumps({"id": "a", "question": "Q?"}) + "\n", encoding="utf-8")
    truth.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_dataset(questions, truth)


def test_evaluate_retrieval_aggregates_per_question_scores(tmp_path):
    questions = tmp_path / "questions.jsonl"
    truth = tmp_path / "ground_truth.jsonl"
    questions.write_text(
        "\n".join(json.dumps({"id": i, "question": q}) for i, q in [("a", "alpha?"), ("b", "beta?")]),
        encoding="utf-8",
    )
    truth.write_text(
        "\n".join(json.dumps({"id": i, "evidence": [e]}) for i, e in [("a", "alpha fact"), ("b", "beta fact")]),
        encoding="utf-8",
    )
    corpus = {"alpha?": ["noise", "the alpha fact"], "beta?": ["noise", "more noise"]}

    report = evaluate_retrieval(load_dataset(questions, truth), lambda q, k: corpus[q], k=2)

    assert report.hit_rate == 0.5
    assert report.mrr == 0.25
    assert [r.id for r in report.misses] == ["b"]


def test_bundled_ground_truth_evidence_exists_in_sample_corpus():
    corpus = " ".join(path.read_text(encoding="utf-8") for path in sorted((REPO_ROOT / "data" / "raw").glob("*.txt")))
    examples = load_dataset(REPO_ROOT / "evaluation" / "questions.jsonl", REPO_ROOT / "evaluation" / "ground_truth.jsonl")

    for example in examples:
        assert match_evidence([corpus], example.evidence)[0] == set(range(len(example.evidence))), example.id
