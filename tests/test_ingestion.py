from pathlib import Path

from pypdf import PdfReader

from ingestion.loader import Document, load_documents


def _write_pdf(path: Path, text: str) -> None:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 16 Tf 50 100 Td ({escaped}) Tj ET".encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf_parts = [b"%PDF-1.4\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(b"".join(pdf_parts)))
        pdf_parts.append(f"{index} 0 obj\n".encode("latin-1"))
        pdf_parts.append(obj)
        pdf_parts.append(b"\nendobj\n")

    xref_offset = len(b"".join(pdf_parts))
    pdf_parts.append(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf_parts.append(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf_parts.append(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf_parts.append(f"trailer\n<< /Root 1 0 R /Size {len(objects) + 1} >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1"))
    path.write_bytes(b"".join(pdf_parts))


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
    _write_pdf(pdf_path, "Alpha beta gamma")

    docs = load_documents(pdf_path)

    assert len(docs) == 1
    assert docs[0].source.endswith("sample.pdf")
    assert "Alpha beta gamma" in docs[0].content

    reader = PdfReader(str(pdf_path))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Alpha beta gamma" in extracted
