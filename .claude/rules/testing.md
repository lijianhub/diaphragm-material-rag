---
paths:
  - "tests/**"
---

# Testing

- pytest finds modules through `pythonpath = ["src"]` in `pyproject.toml`, so tests import `from chunking import ...` rather than `from src.chunking import ...`. No `conftest.py` or `sys.path` changes are needed.
- Tests must run offline and deterministically, without network access, API keys or model downloads. Use `SimpleEmbedder` or a stub instead of a real embedding or LLM backend.
- Build fixtures inside `tmp_path`. Never depend on git-ignored files in `data/raw/` (such as the material PDF), because they are missing in a fresh clone. Read only the tracked sample `*.txt` corpus, and only in data-integrity tests such as the evaluation-set check in `tests/test_evaluation.py`.
- Do not commit binary fixtures. `tests/file_builders.py` generates PDF (multi-page, blank pages), DOCX (headings, tables) and XLSX (several sheets) files in code; use it, and extend it for new formats. Import it as `from tests.file_builders import ...`.
- `tests/test_agent.py` is an empty placeholder, as is `src/agent/`. Fill it in when you implement the agent.
- Tests are plain `assert`-style functions with no test classes, one `tests/test_<stage>.py` per pipeline stage.
