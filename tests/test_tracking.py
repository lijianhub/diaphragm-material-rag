import os
import subprocess

import pytest

from tracking import (
    JsonlTracker,
    MlflowTracker,
    NullTracker,
    RunRecord,
    create_tracker,
    fingerprint,
    flatten,
    git_state,
    log_safely,
    metric_name,
    version_tags,
)
from tests.fakes import FakeMlflow


def _record(tmp_path=None):
    artifacts = []
    if tmp_path is not None:
        path = tmp_path / "results.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        artifacts = [str(path), str(tmp_path / "missing.jsonl")]
    return RunRecord(
        name="retrieval simple-hash",
        params={"alpha": "0.5", "embedder": "simple-hash-512"},
        metrics={"hit@3": 0.956, "paraphrase.hit@3": 0.833},
        tags={"git_commit": "abc"},
        artifacts=artifacts,
    )


def test_metric_names_are_made_safe_for_mlflow():
    # MLflow rejects '@' in metric names; 'hit@3' must not fail a run.
    assert metric_name("hit@3") == "hit_at_3"
    assert metric_name("paraphrase.recall@3") == "paraphrase.recall_at_3"
    assert metric_name("gate:refusal") == "gate_refusal"


def test_flatten_joins_nested_keys():
    assert flatten("gen", {"a": 1, "b": {"c": 2}}) == {"gen.a": 1, "gen.b.c": 2}


def test_fingerprint_changes_with_content_name_and_membership_but_not_order(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("alpha", encoding="utf-8")
    b.write_text("beta", encoding="utf-8")
    base = fingerprint([a, b])

    assert fingerprint([b, a]) == base
    b.write_text("beta edited", encoding="utf-8")
    assert fingerprint([a, b]) != base
    assert fingerprint([a]) != fingerprint([a, b])
    c = tmp_path / "c.txt"
    c.write_text("alpha", encoding="utf-8")  # same content as a.txt, different name
    assert fingerprint([c]) != fingerprint([a])


def test_git_state_reports_commit_and_uncommitted_changes(tmp_path):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path, check=True, capture_output=True)

    assert git_state(tmp_path)["git_commit"] == "unknown"  # not a repository: no crash
    git("init", "-q")
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "init")

    clean = git_state(tmp_path)
    (tmp_path / "f.txt").write_text("changed", encoding="utf-8")
    dirty = git_state(tmp_path)

    assert len(clean["git_commit"]) == 40 and clean["git_dirty"] == "false"
    assert dirty["git_dirty"] == "true" and dirty["git_commit"] == clean["git_commit"]


def test_version_tags_identify_data_and_corpus(tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_text('{"id": "q01"}\n', encoding="utf-8")
    doc = tmp_path / "notes.txt"
    doc.write_text("text", encoding="utf-8")

    tags = version_tags(tmp_path, {"questions": questions}, [doc])
    questions.write_text('{"id": "q01"}\n{"id": "q02"}\n', encoding="utf-8")
    changed = version_tags(tmp_path, {"questions": questions}, [doc])

    assert set(tags) >= {"git_commit", "git_dirty", "data.questions.sha256", "data.corpus.fingerprint", "data.corpus.files"}
    assert tags["data.corpus.files"] == "1"
    assert changed["data.questions.sha256"] != tags["data.questions.sha256"]
    assert changed["data.corpus.fingerprint"] == tags["data.corpus.fingerprint"]


def test_jsonl_tracker_appends_one_line_per_run(tmp_path):
    tracker = JsonlTracker(tmp_path / "runs" / "runs.jsonl")

    first = tracker.log_run(_record())
    tracker.log_run(RunRecord(name="second", metrics={"hit@1": 0.9}))
    runs = tracker.runs()

    assert [r["name"] for r in runs] == ["retrieval simple-hash", "second"]
    assert runs[0]["run_id"] == first and runs[0]["metrics"]["hit@3"] == 0.956
    assert "time" in runs[0]


def test_mlflow_tracker_logs_params_metrics_tags_and_existing_artifacts(tmp_path):
    fake = FakeMlflow()
    tracker = MlflowTracker(experiment="exp", tracking_uri="sqlite:///x.db", artifact_location="file:///art", client=fake)

    run_id = tracker.log_run(_record(tmp_path))
    run = fake.runs[0]

    assert run_id == "run-1"
    assert fake.tracking_uri == "sqlite:///x.db"
    assert fake.experiments["exp"]["artifact_location"] == "file:///art" and run["experiment"] == "exp"
    assert run["params"] == {"alpha": "0.5", "embedder": "simple-hash-512"}
    assert run["metrics"] == {"hit_at_3": 0.956, "paraphrase.hit_at_3": 0.833}
    assert run["tags"] == {"git_commit": "abc"}
    assert run["artifacts"] == [str(tmp_path / "results.jsonl")]  # the missing file is skipped


def test_mlflow_tracker_keeps_an_existing_experiments_artifact_location():
    fake = FakeMlflow(existing_experiments=["exp"])

    MlflowTracker(experiment="exp", artifact_location="file:///new", client=fake)

    assert fake.experiments["exp"]["artifact_location"] is None


def test_long_param_values_are_cut_instead_of_failing_the_run():
    fake = FakeMlflow()

    MlflowTracker(client=fake).log_run(RunRecord(name="r", params={"served_models": "x" * 2000}))

    assert len(fake.runs[0]["params"]["served_models"]) == 500


def test_create_tracker_puts_files_under_the_repository_root(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    fake = FakeMlflow()

    assert isinstance(create_tracker("none", tmp_path), NullTracker)
    assert create_tracker("jsonl", tmp_path).path == tmp_path.resolve() / "runs" / "runs.jsonl"
    create_tracker("mlflow", tmp_path, client=fake)
    assert fake.tracking_uri == f"sqlite:///{(tmp_path.resolve() / 'mlflow.db').as_posix()}"
    assert fake.experiments["diaphragm-material-rag"]["artifact_location"] == (tmp_path.resolve() / "mlartifacts").as_uri()

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example:5000")
    create_tracker("mlflow", tmp_path, client=fake)
    assert fake.tracking_uri == "http://tracking.example:5000"

    with pytest.raises(ValueError):
        create_tracker("wandb", tmp_path)


def test_a_failing_tracker_does_not_fail_the_evaluation(capsys):
    class Broken:
        def log_run(self, record):
            raise ConnectionError("tracking server down")

    assert log_safely(Broken(), _record()) is None
    assert "tracking failed (ConnectionError" in capsys.readouterr().out


@pytest.mark.skipif(not os.environ.get("RUN_MLFLOW_TESTS"), reason="slow (~15 s of database migrations); set RUN_MLFLOW_TESTS=1")
def test_real_mlflow_round_trip(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    tracker = create_tracker("mlflow", tmp_path, tracking_uri=f"sqlite:///{(tmp_path / 'm.db').as_posix()}")

    run_id = tracker.log_run(_record(tmp_path))
    runs = mlflow.search_runs(experiment_names=["diaphragm-material-rag"], output_format="list")

    assert [r.info.run_id for r in runs] == [run_id]
    assert runs[0].data.metrics["hit_at_3"] == 0.956
    assert runs[0].data.params["embedder"] == "simple-hash-512"
