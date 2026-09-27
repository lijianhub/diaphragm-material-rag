from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None


@dataclass
class Document:
    source: str
    content: str


def _read_pdf_file(path: Path) -> str:
    if PdfReader is None:
        raise ImportError("PDF support requires the 'pypdf' package. Install it with: pip install pypdf")

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_documents(paths: Union[str, Path, Iterable[Union[str, Path]]]) -> List[Document]:
    """Load text documents from a file, directory, or iterable of file paths."""
    if isinstance(paths, (str, Path)):
        file_paths = [Path(paths)]
    else:
        file_paths = [Path(p) for p in paths]

    documents: List[Document] = []
    for path in file_paths:
        if path.is_dir():
            candidates = sorted(path.rglob("*"))
            for candidate in candidates:
                if candidate.is_file() and candidate.suffix.lower() in {".txt", ".md", ".json", ".csv", ".pdf"}:
                    if candidate.suffix.lower() == ".pdf":
                        content = _read_pdf_file(candidate)
                    else:
                        content = candidate.read_text(encoding="utf-8")
                    documents.append(Document(source=str(candidate), content=content))
        elif path.is_file():
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                content = _read_pdf_file(path)
            else:
                content = path.read_text(encoding="utf-8")
            documents.append(Document(source=str(path), content=content))

    return documents
