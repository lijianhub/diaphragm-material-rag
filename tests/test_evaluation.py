import json
from pathlib import Path

import pytest

from evaluation import benchmark_corpus, evaluate_retrieval, hit_at_k, load_dataset, match_evidence, recall_at_k, reciprocal_rank
from ingestion.loader import load_documents

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
    # Check against parsed text (HTML tags removed, CSV rows labelled), which is what gets indexed.
    documents = load_documents(benchmark_corpus(REPO_ROOT / "data" / "raw"))
    corpus = " ".join(section.text for document in documents for section in document.sections)
    examples = load_dataset(REPO_ROOT / "evaluation" / "questions.jsonl", REPO_ROOT / "evaluation" / "ground_truth.jsonl")

    for example in examples:
        assert match_evidence([corpus], example.evidence)[0] == set(range(len(example.evidence))), example.id


def test_report_breaks_metrics_down_by_category(tmp_path):
    questions = tmp_path / "questions.jsonl"
    truth = tmp_path / "ground_truth.jsonl"
    rows = [("a", "keyword", "alpha?"), ("b", "keyword", "beta?"), ("c", "paraphrase", "gamma?")]
    questions.write_text("\n".join(json.dumps({"id": i, "category": c, "question": q}) for i, c, q in rows), encoding="utf-8")
    truth.write_text(
        "\n".join(json.dumps({"id": i, "evidence": [f"{i} fact"]}) for i, _, _ in rows),
        encoding="utf-8",
    )
    corpus = {"alpha?": ["a fact"], "beta?": ["noise"], "gamma?": ["c fact"]}

    report = evaluate_retrieval(load_dataset(questions, truth), lambda q, k: corpus[q], k=1)
    by_category = report.by_category()

    assert set(by_category) == {"keyword", "paraphrase"}
    assert by_category["keyword"].count == 2
    assert by_category["keyword"].hit_rate == 0.5
    assert by_category["paraphrase"].hit_rate == 1.0


def test_category_defaults_to_general(tmp_path):
    questions = tmp_path / "questions.jsonl"
    truth = tmp_path / "ground_truth.jsonl"
    questions.write_text(json.dumps({"id": "a", "question": "Q?"}), encoding="utf-8")
    truth.write_text(json.dumps({"id": "a", "evidence": ["x"]}), encoding="utf-8")

    assert load_dataset(questions, truth)[0].category == "general"


def test_benchmark_corpus_uses_only_tracked_sample_formats(tmp_path):
    for name in ["notes.txt", "guide.md", "faq.html", "rows.csv", "private.pdf", "spec.docx"]:
        (tmp_path / name).write_text("x", encoding="utf-8")

    assert [p.name for p in benchmark_corpus(tmp_path)] == ["faq.html", "guide.md", "notes.txt", "rows.csv"]
