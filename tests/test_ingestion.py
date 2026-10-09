from pathlib import Path

import pytest
from pypdf import PdfReader

from ingestion.loader import Document, load_documents
from tests.file_builders import write_pdf


def test_load_documents_reads_text_files(tmp_path):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.md"
    file_a.write_text("Alpha beta gamma", encoding="utf-8")
    file_b.write_text("Gamma delta", encoding="utf-8")

    docs = load_documents([file_a, file_b])

    assert len(docs) == 2
    assert all(isinstance(doc, Document) for doc in docs)
    assert docs[0].content.startswith("Alpha")
    assert docs[1].source.endswith("b.md")


def test_load_documents_from_directory(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "paper.txt").write_text("hello world", encoding="utf-8")

    docs = load_documents(docs_dir)

    assert len(docs) == 1
    assert docs[0].content == "hello world"


def test_load_documents_reads_pdf_files(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    write_pdf(pdf_path, "Alpha beta gamma")

    docs = load_documents(pdf_path)

    assert len(docs) == 1
    assert docs[0].source.endswith("sample.pdf")
    assert "Alpha beta gamma" in docs[0].content

    reader = PdfReader(str(pdf_path))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Alpha beta gamma" in extracted


def test_ingest_reports_failures_and_keeps_going(tmp_path):
    from ingestion.loader import ingest

    (tmp_path / "good.txt").write_text("Springback notes", encoding="utf-8")
    (tmp_path / "broken.docx").write_bytes(b"not a zip")
    (tmp_path / "~$good.docx").write_bytes(b"office lock file")
    (tmp_path / ".hidden.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "drawing.dwg").write_bytes(b"\x00")

    result = ingest(tmp_path)

    assert [Path(d.source).name for d in result.documents] == ["good.txt"]
    assert [Path(f.source).name for f in result.failures] == ["broken.docx"]
    assert sorted(Path(s).name for s in result.skipped) == [".hidden.txt", "drawing.dwg", "~$good.docx"]


def test_ingest_flags_identical_content_under_different_names(tmp_path):
    from ingestion.loader import ingest

    (tmp_path / "a.txt").write_text("Same text", encoding="utf-8")
    (tmp_path / "copy of a.txt").write_text("Same text", encoding="utf-8")

    result = ingest(tmp_path)

    assert len(result.documents) == 2
    assert [(Path(dup).name, Path(orig).name) for dup, orig in result.duplicates] == [("copy of a.txt", "a.txt")]


def test_documents_carry_type_title_and_sections(tmp_path):
    path = tmp_path / "procedure.md"
    path.write_text("# Aging\nVacuum only.\n\n## Cooling\nAir cool.", encoding="utf-8")

    document = load_documents(path)[0]

    assert document.metadata["doc_type"] == "md"
    assert document.metadata["title"] == "Aging"
    assert len(document.sections) == 2
    assert "Vacuum only." in document.content and "Air cool." in document.content


def test_load_documents_raises_on_failures_in_strict_mode(tmp_path):
    (tmp_path / "broken.docx").write_bytes(b"not a zip")

    with pytest.raises(ValueError, match="broken.docx"):
        load_documents(tmp_path)
