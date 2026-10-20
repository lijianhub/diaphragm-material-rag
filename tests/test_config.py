import os

from config import load_env_file


def test_env_file_sets_missing_variables_and_reports_only_names(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_TEST_KEY", raising=False)
    monkeypatch.delenv("RAG_TEST_MODEL", raising=False)
    env = tmp_path / ".env"
    env.write_text('# comment\nRAG_TEST_KEY="secret value"\nexport RAG_TEST_MODEL=openai:m\n\nBROKEN LINE\n', encoding="utf-8")

    loaded = load_env_file(env)

    assert loaded == ["RAG_TEST_KEY", "RAG_TEST_MODEL"]  # names only, never values
    assert os.environ["RAG_TEST_KEY"] == "secret value"
    assert os.environ["RAG_TEST_MODEL"] == "openai:m"


def test_env_file_never_overrides_the_real_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_TEST_KEY", "from-shell")
    env = tmp_path / ".env"
    env.write_text("RAG_TEST_KEY=from-file\n", encoding="utf-8")

    assert load_env_file(env) == []
    assert os.environ["RAG_TEST_KEY"] == "from-shell"


def test_missing_env_file_is_not_an_error(tmp_path):
    assert load_env_file(tmp_path / "nope.env") == []


def test_empty_values_are_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_TEST_EMPTY", raising=False)
    env = tmp_path / ".env"
    env.write_text("RAG_TEST_EMPTY=\n", encoding="utf-8")

    assert load_env_file(env) == []
    assert "RAG_TEST_EMPTY" not in os.environ
