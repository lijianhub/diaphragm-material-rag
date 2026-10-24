---
paths:
  - "tests/**"
---

# Testing

- pytest finds modules through `pythonpath = ["src"]` in `pyproject.toml`, so tests import `from chunking import ...` rather than `from src.chunking import ...`. No `conftest.py` or `sys.path` changes are needed.
- Tests must run offline and deterministically, without network access, API keys or model downloads. Use `SimpleEmbedder` or a stub instead of a real embedding or LLM backend.
- Build fixtures inside `tmp_path`. Never depend on git-ignored files in `data/raw/` (such as the material PDF), because they are missing in a fresh clone. Read only the tracked sample corpus returned by `evaluation.benchmark_corpus()` (txt, md, html, csv), and only in data-integrity tests such as the evaluation-set check in `tests/test_evaluation.py`.
- Do not commit binary fixtures. `tests/file_builders.py` generates PDF (multi-page, blank pages), DOCX (headings, tables) and XLSX (several sheets) files in code; use it, and extend it for new formats. Import it as `from tests.file_builders import ...`.
- Agent tests (`tests/test_agent.py`) use `FakePlanner` (scripted `ToolTurn`s) and never call a model; provider tool-calling tests use `FakeToolClient`. Give every agent its own `tmp_path` run store and NCR file.
- Tests are plain `assert`-style functions with no test classes, one `tests/test_<stage>.py` per pipeline stage.
