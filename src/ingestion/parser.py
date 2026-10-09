"""Turn files of different formats into plain-text sections with metadata.

Each parser returns ``Parsed(sections, metadata)``. A section is a unit with its
own citation metadata: a PDF page, a spreadsheet sheet, or the text under a
heading in DOCX, HTML or Markdown. Chunking happens per section, so every chunk
can be cited as "manual.pdf, page 3" or "lots.xlsx, sheet Q3".

Office formats are read with the standard library (they are zip archives of
XML), so no extra dependencies are needed. Register more formats with
``register_parser``.
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from xml.etree import ElementTree

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None


class ParseError(Exception):
    """A file could not be turned into text; the message says why."""


@dataclass
class Section:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Parsed:
    sections: List[Section]
    metadata: Dict[str, Any] = field(default_factory=dict)


ParserFn = Callable[[Path], Parsed]
_PARSERS: Dict[str, ParserFn] = {}


def register_parser(suffixes: Sequence[str], parser: ParserFn) -> None:
    for suffix in suffixes:
        _PARSERS[suffix.lower()] = parser


def supported_suffixes() -> List[str]:
    return sorted(_PARSERS)


def parse_file(path: Path) -> Parsed:
    path = Path(path)
    parser = _PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ParseError(f"unsupported file type {path.suffix or '(none)'!r}")
    try:
        parsed = parser(path)
    except ParseError:
        raise
    except (zipfile.BadZipFile, ElementTree.ParseError, KeyError) as exc:
        raise ParseError(f"corrupted or invalid {path.suffix} file: {exc}") from exc
    except Exception as exc:  # any library failure becomes a per-file error, not a crash
        raise ParseError(f"{type(exc).__name__}: {exc}") from exc

    parsed.sections = [s for s in parsed.sections if s.text.strip()]
    if not parsed.sections:
        raise ParseError("no extractable text")
    return parsed


# ---- plain text, Markdown, JSON, CSV ----------------------------------------


def read_text(path: Path) -> str:
    """UTF-8 (with or without BOM), falling back to Windows-1252, which most legacy
    Office exports use, and finally Latin-1, which never fails."""
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _parse_text(path: Path) -> Parsed:
    return Parsed([Section(read_text(path))])


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _parse_markdown(path: Path) -> Parsed:
    sections: List[Section] = []
    heading: Optional[str] = None
    lines: List[str] = []
    title: Optional[str] = None
    in_code = False

    def flush() -> None:
        if "".join(lines).strip():
            sections.append(Section("\n".join(lines).strip(), {"heading": heading} if heading else {}))

    for line in read_text(path).splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _MD_HEADING.match(line)
        if match:
            flush()
            heading, lines = match.group(2), [line]
            if title is None and len(match.group(1)) == 1:
                title = heading
        else:
            lines.append(line)
    flush()
    return Parsed(sections, {"title": title} if title else {})


def _labelled_rows(rows: List[List[str]]) -> str:
    """Render table rows as "Header: value; ..." lines, so every chunk keeps the column
    meaning even when the header row lands in another chunk."""
    if not rows:
        return ""
    header = [h.strip() or f"column {i + 1}" for i, h in enumerate(rows[0])]
    lines = []
    for row in rows[1:]:
        pairs = [f"{header[i] if i < len(header) else f'column {i + 1}'}: {v.strip()}" for i, v in enumerate(row) if v.strip()]
        if pairs:
            lines.append("; ".join(pairs))
    return "\n".join(lines)


def _parse_csv(path: Path) -> Parsed:
    text = read_text(path)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = [row for row in csv.reader(io.StringIO(text), dialect) if any(cell.strip() for cell in row)]
    return Parsed([Section(_labelled_rows(rows))], {"rows": max(len(rows) - 1, 0)})


# ---- HTML ---------------------------------------------------------------------


class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "noscript", "template", "nav", "footer", "svg"}
    BLOCK = {"p", "div", "li", "tr", "br", "section", "article", "table", "ul", "ol", "pre", "blockquote"}
    HEADINGS = {"h1", "h2", "h3"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: List[Section] = []
        self.title = ""
        self._heading: Optional[str] = None
        self._buffer: List[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._heading_text: Optional[List[str]] = None

    def _flush(self) -> None:
        text = re.sub(r"[ \t\xa0]+", " ", "".join(self._buffer))
        text = re.sub(r"\s*\n\s*", "\n", text).strip()
        if text:
            self.sections.append(Section(text, {"heading": self._heading} if self._heading else {}))
        self._buffer = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.HEADINGS and not self._skip_depth:
            self._flush()
            self._heading_text = []
        elif tag in self.BLOCK:
            self._buffer.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in self.HEADINGS and self._heading_text is not None:
            self._heading = re.sub(r"\s+", " ", "".join(self._heading_text)).strip()
            self._buffer.append(self._heading + "\n")
            self._heading_text = None
        elif tag in self.BLOCK:
            self._buffer.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._skip_depth:
            return
        elif self._heading_text is not None:
            self._heading_text.append(data)
        else:
            self._buffer.append(data)


def _parse_html(path: Path) -> Parsed:
    parser = _HTMLText()
    parser.feed(read_text(path))
    parser.close()
    parser._flush()
    title = re.sub(r"\s+", " ", parser.title).strip()
    return Parsed(parser.sections, {"title": title} if title else {})


# ---- PDF ----------------------------------------------------------------------


def _parse_pdf(path: Path) -> Parsed:
    if PdfReader is None:
        raise ParseError("PDF support requires the 'pypdf' package: pip install pypdf")
    reader = PdfReader(str(path))
    sections = [
        Section(text, {"page": number})
        for number, page in enumerate(reader.pages, start=1)
        if (text := (page.extract_text() or "")).strip()
    ]
    if not sections:
        raise ParseError("no extractable text; the PDF is probably scanned images and needs OCR")
    return Parsed(sections, {"pages": len(reader.pages)})


# ---- DOCX -----------------------------------------------------------------------

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_text(element) -> str:
    parts = []
    for node in element.iter():
        if node.tag == f"{_W}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{_W}tab":
            parts.append("\t")
        elif node.tag in (f"{_W}br", f"{_W}cr"):
            parts.append("\n")
    return "".join(parts)


def _parse_docx(path: Path) -> Parsed:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    body = root.find(f"{_W}body")
    sections: List[Section] = []
    heading: Optional[str] = None
    lines: List[str] = []
    title: Optional[str] = None

    def flush() -> None:
        if "".join(lines).strip():
            sections.append(Section("\n".join(lines).strip(), {"heading": heading} if heading else {}))

    for block in list(body) if body is not None else []:
        if block.tag == f"{_W}p":
            style = block.find(f"{_W}pPr/{_W}pStyle")
            style_name = style.get(f"{_W}val", "") if style is not None else ""
            text = _docx_text(block).strip()
            if style_name.lower().startswith(("heading", "title")) and text:
                flush()
                heading, lines = text, [text]
                if title is None:
                    title = text
            elif text:
                lines.append(text)
        elif block.tag == f"{_W}tbl":
            for row in block.iter(f"{_W}tr"):
                cells = [_docx_text(cell).strip() for cell in row.findall(f"{_W}tc")]
                if any(cells):
                    lines.append(" | ".join(cells))
    flush()
    return Parsed(sections, {"title": title} if title else {})


# ---- XLSX -----------------------------------------------------------------------

_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _column_index(ref: str) -> int:
    index = 0
    for ch in ref:
        if not ch.isalpha():
            break
        index = index * 26 + (ord(ch.upper()) - 64)
    return index - 1


def _parse_xlsx(path: Path) -> Parsed:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        shared: List[str] = []
        if "xl/sharedStrings.xml" in names:
            for item in ElementTree.fromstring(archive.read("xl/sharedStrings.xml")).iter(f"{_S}si"):
                shared.append("".join(t.text or "" for t in item.iter(f"{_S}t")))

        rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {rel.get("Id"): rel.get("Target") for rel in rels.iter(f"{_PKG}Relationship")}
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))

        sections: List[Section] = []
        for sheet in workbook.iter(f"{_S}sheet"):
            target = targets[sheet.get(f"{_R}id")].lstrip("/")
            target = target if target.startswith("xl/") else f"xl/{target}"
            rows: List[List[str]] = []
            for row in ElementTree.fromstring(archive.read(target)).iter(f"{_S}row"):
                values: Dict[int, str] = {}
                for cell in row.iter(f"{_S}c"):
                    kind = cell.get("t")
                    if kind == "inlineStr":
                        text = "".join(t.text or "" for t in cell.iter(f"{_S}t"))
                    else:
                        raw = cell.findtext(f"{_S}v")
                        if raw is None:
                            continue
                        text = shared[int(raw)] if kind == "s" else ("TRUE" if raw == "1" else "FALSE") if kind == "b" else raw
                    values[_column_index(cell.get("r", "A"))] = text
                if values:
                    rows.append([values.get(i, "") for i in range(max(values) + 1)])
            sections.append(Section(_labelled_rows(rows), {"sheet": sheet.get("name")}))
    return Parsed(sections, {"sheets": len(sections)})


register_parser([".txt", ".text", ".log"], _parse_text)
register_parser([".json"], _parse_text)
register_parser([".md", ".markdown"], _parse_markdown)
register_parser([".csv", ".tsv"], _parse_csv)
register_parser([".html", ".htm"], _parse_html)
register_parser([".pdf"], _parse_pdf)
register_parser([".docx"], _parse_docx)
register_parser([".xlsx"], _parse_xlsx)
