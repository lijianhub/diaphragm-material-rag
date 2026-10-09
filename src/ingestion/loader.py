from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

from .parser import ParseError, Section, parse_file, supported_suffixes

PathInput = Union[str, Path, Iterable[Union[str, Path]]]


@dataclass
class Document:
    source: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Citation units (pages, sheets, heading sections). Empty means "one section: content".
    sections: List[Section] = field(default_factory=list)


@dataclass
class IngestFailure:
    source: str
    reason: str


@dataclass
class IngestResult:
    documents: List[Document] = field(default_factory=list)
    failures: List[IngestFailure] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    duplicates: List[Tuple[str, str]] = field(default_factory=list)  # (duplicate, first seen)

    def summary(self) -> str:
        return (
            f"loaded={len(self.documents)} failed={len(self.failures)} "
            f"skipped={len(self.skipped)} duplicates={len(self.duplicates)}"
        )


def _is_junk(path: Path) -> bool:
    # Hidden files and Office lock files ("~$report.docx") are never documents.
    return path.name.startswith((".", "~$"))


def _expand(paths: PathInput) -> Tuple[List[Path], List[str]]:
    """Explicit files are always attempted; directories contribute supported, non-junk files."""
    roots = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    suffixes = set(supported_suffixes())
    files: List[Path] = []
    skipped: List[str] = []
    for root in roots:
        if root.is_dir():
            for candidate in sorted(root.rglob("*")):
                if not candidate.is_file():
                    continue
                if _is_junk(candidate) or candidate.suffix.lower() not in suffixes:
                    skipped.append(str(candidate))
                else:
                    files.append(candidate)
        elif root.is_file():
            files.append(root)
    return files, skipped


def load_file(path: Path) -> Document:
    parsed = parse_file(path)
    content = "\n\n".join(section.text for section in parsed.sections)
    metadata = {"doc_type": path.suffix.lstrip(".").lower() or "text", **parsed.metadata}
    return Document(source=str(path), content=content, metadata=metadata, sections=parsed.sections)


def ingest(paths: PathInput) -> IngestResult:
    """Load every supported file, collecting failures instead of stopping at the first one.

    A corrupted file, a scanned PDF without text or an unsupported explicit file
    becomes an ``IngestFailure``; the rest of the batch still loads.
    """
    files, skipped = _expand(paths)
    result = IngestResult(skipped=skipped)
    seen: Dict[str, str] = {}
    for path in files:
        try:
            document = load_file(path)
        except (ParseError, OSError) as exc:
            result.failures.append(IngestFailure(source=str(path), reason=str(exc)))
            continue
        digest = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
        if digest in seen:
            result.duplicates.append((document.source, seen[digest]))
        else:
            seen[digest] = document.source
        result.documents.append(document)
    return result


def load_documents(paths: PathInput) -> List[Document]:
    """Strict loading: raise if any file fails. Use ``ingest`` to keep going and get a report."""
    result = ingest(paths)
    if result.failures:
        details = "; ".join(f"{Path(f.source).name}: {f.reason}" for f in result.failures)
        raise ValueError(f"{len(result.failures)} file(s) could not be loaded: {details}")
    return result.documents
