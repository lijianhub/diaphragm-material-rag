"""Build minimal PDF, DOCX and XLSX files in code, so tests need no binary fixtures."""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union
from xml.sax.saxutils import escape


def write_pdf(path: Path, pages: Union[str, Sequence[str]]) -> None:
    """One line of Helvetica text per page; an empty string makes a page with no text."""
    if isinstance(pages, str):
        pages = [pages]
    n = len(pages)
    # Object numbers: 1 catalog, 2 pages, 3 font, then (page, content) pairs.
    page_ids = [4 + 2 * i for i in range(n)]
    objects: List[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {n} >>".encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 40 100 Td ({escaped}) Tj ET".encode("latin-1", errors="replace") if text else b""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Contents {page_ids[i] + 1} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>".encode("latin-1")
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream")

    parts = [b"%PDF-1.4\n"]
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(b"".join(parts)))
        parts.append(f"{number} 0 obj\n".encode("latin-1") + obj + b"\nendobj\n")
    xref = len(b"".join(parts))
    parts.append(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1"))
    parts.extend(f"{offset:010d} 00000 n \n".encode("latin-1") for offset in offsets)
    parts.append(f"trailer\n<< /Root 1 0 R /Size {len(objects) + 1} >>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1"))
    path.write_bytes(b"".join(parts))


_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def write_docx(path: Path, blocks: Sequence[Tuple[str, object]]) -> None:
    """Blocks are ("p", text), ("h1", text), ("h2", text) or ("table", [[cell, ...], ...])."""
    body = []
    for kind, value in blocks:
        if kind == "table":
            rows = "".join(
                "<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:t>{escape(c)}</w:t></w:r></w:p></w:tc>" for c in row) + "</w:tr>"
                for row in value
            )
            body.append(f"<w:tbl>{rows}</w:tbl>")
        else:
            style = {"h1": "Heading1", "h2": "Heading2"}.get(kind)
            ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
            body.append(f"<w:p>{ppr}<w:r><w:t xml:space=\"preserve\">{escape(value)}</w:t></w:r></w:p>")
    document = f'<?xml version="1.0" encoding="UTF-8"?><w:document {_W}><w:body>{"".join(body)}</w:body></w:document>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document)


def _column(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def write_xlsx(path: Path, sheets: Dict[str, Sequence[Sequence[object]]]) -> None:
    """Strings go to the shared-string table, numbers are stored as numbers."""
    shared: List[str] = []
    sheet_xml = []
    for rows in sheets.values():
        xml_rows = []
        for r, row in enumerate(rows, start=1):
            cells = []
            for c, value in enumerate(row):
                ref = f"{_column(c)}{r}"
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    cells.append(f'<c r="{ref}"><v>{value}</v></c>')
                else:
                    if value not in shared:
                        shared.append(value)
                    cells.append(f'<c r="{ref}" t="s"><v>{shared.index(value)}</v></c>')
            xml_rows.append(f'<row r="{r}">{"".join(cells)}</row>')
        sheet_xml.append(
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
        )
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rel_ns = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    workbook = f"<workbook {ns} {rel_ns}><sheets>" + "".join(
        f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i, name in enumerate(sheets, start=1)
    ) + "</sheets></workbook>"
    rels = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(
        f'<Relationship Id="rId{i}" Type="worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets) + 1)
    ) + "</Relationships>"
    strings = f"<sst {ns}>" + "".join(f"<si><t>{escape(s)}</t></si>" for s in shared) + "</sst>"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        archive.writestr("xl/sharedStrings.xml", strings)
        for i, xml in enumerate(sheet_xml, start=1):
            archive.writestr(f"xl/worksheets/sheet{i}.xml", xml)
