---
paths:
  - "tests/**"
---

# Testing

- pytest finds modules through `pythonpath = ["src"]` in `pyproject.toml`, so tests import `from chunking import ...` rather than `from src.chunking import ...`. No `conftest.py` or `sys.path` changes are needed.
- Tests must run offline and deterministically, without network access, API keys or model downloads. Use `SimpleEmbedder` or a stub instead of a real embedding or LLM backend.
- Build fixtures inside `tmp_path`. Do not read from `data/raw/`, because the real PDF there is git-ignored and missing in a fresh clone.
- Do not commit binary fixtures. `tests/test_ingestion.py::_write_pdf` generates a minimal PDF in code; reuse it for any PDF test.
- `tests/test_agent.py` and `tests/test_evaluation.py` are empty placeholders, as are the modules they would cover. Fill them in when you implement `src/agent/` or `src/evaluation/`.
- Tests are plain `assert`-style functions with no test classes, one `tests/test_<stage>.py` per pipeline stage.
