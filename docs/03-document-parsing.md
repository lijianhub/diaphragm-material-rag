# Document Parsing

[02-document-ingestion.md](02-document-ingestion.md) covers *what* is supported and how batches are loaded. This document covers *how* each format is parsed in `src/ingestion/parser.py`, and how to add a new one.

## The parser contract

```python
parse_file(path) -> Parsed(sections: List[Section], metadata: Dict)
Section(text: str, metadata: Dict)          # e.g. {"page": 2}, {"sheet": "Q3 lots"}, {"heading": "Cooling"}
```

`parse_file` picks the parser by file suffix and then applies the same rules to every format:

1. Any library exception becomes a `ParseError` with a readable reason: corrupted zip archives, malformed XML and missing archive members are reported as "corrupted or invalid .docx file"; anything else as `"<ExceptionType>: <message>"`.
2. Sections whose text is only whitespace are dropped.
3. If no section is left, the file fails with "no extractable text".

`ingest()` turns each `ParseError` into an entry in the failure report, so one bad file never stops a batch.

## Format by format

### Plain text and JSON (`.txt .text .log .json`)
One section with the decoded text. JSON is indexed as text; its keys and values stay searchable.

**Decoding** (`read_text`): try UTF-8 with BOM handling (`utf-8-sig`), then Windows-1252, then Latin-1. Windows-1252 covers most legacy Office and Excel exports (for example `°` as byte `0xB0`). Latin-1 accepts any byte, so decoding itself never fails.

### Markdown (`.md .markdown`)
- A new section starts at every ATX heading (`#` to `######`); the heading line stays in the section text, and the heading is stored as `heading`.
- Lines inside fenced code blocks (```` ``` ````) are never treated as headings, so `# comment` in a shell snippet does not split the document.
- The first level-1 heading becomes the document `title`.
- Setext headings (underlined with `===`) are not recognised; they stay as ordinary text.

### HTML (`.html .htm`)
Uses the standard-library `HTMLParser`.

| Element | Treatment |
|---|---|
| `script`, `style`, `noscript`, `template`, `nav`, `footer`, `svg` | Dropped with all their content (boilerplate and code) |
| `title` | Becomes the document `title`, not body text |
| `h1`, `h2`, `h3` | Start a new section; the heading text is kept and stored as `heading` |
| `p`, `div`, `li`, `tr`, `br`, `table`, `section`, `article`, … | Line breaks, so words from adjacent blocks do not run together |
| Character references (`&nbsp;`, `&amp;`) | Decoded; non-breaking spaces become normal spaces |

Whitespace is collapsed within lines. Text before the first heading forms a section without a `heading`.

### PDF (`.pdf`)
- `pypdf` extracts text page by page; each page with text is a section with `page` (1-based).
- Blank pages are dropped **without renumbering**, so a citation always names the real page.
- The document metadata records the total `pages`.
- If no page has text, the file is a scan and fails with "no extractable text; the PDF is probably scanned images and needs OCR".

PDF text extraction follows the content stream, not the visual layout: multi-column pages can interleave columns, and tables lose their grid. Check the extracted text of important PDFs.

### Word (`.docx`)
A `.docx` file is a zip archive; the body is `word/document.xml` in the WordprocessingML namespace. The parser walks the body's children in order:

```xml
<w:p>                                   paragraph
  <w:pPr><w:pStyle w:val="Heading1"/>   style: Heading* or Title starts a new section
  <w:r><w:t>text</w:t></w:r>           runs of text; <w:tab/> → tab, <w:br/> → newline
<w:tbl><w:tr><w:tc>…                    table rows → "cell | cell | cell"
```

- The first heading becomes the document `title`.
- Headers, footers, footnotes and comments live in other archive members and are not read.
- Text boxes and content inside unusual containers may be missed.

### Excel (`.xlsx`)
Also a zip archive (SpreadsheetML):

| Archive member | Used for |
|---|---|
| `xl/workbook.xml` | Sheet names and their relationship ids, in workbook order |
| `xl/_rels/workbook.xml.rels` | Relationship id → worksheet file |
| `xl/sharedStrings.xml` | Text cells store an index into this table |
| `xl/worksheets/sheetN.xml` | Rows and cells |

Cell values: `t="s"` → shared string; `t="inlineStr"` → inline text; `t="b"` → `TRUE`/`FALSE`; otherwise the stored value, which for formulas is the **cached result**. Columns are aligned by the cell reference (`C7` → column 3), so empty cells don't shift values into the wrong column.

Each sheet is one section with `sheet`. Rows become labelled lines using the first row as headers:

```
Alloy: ELGILOY; Base: Cobalt; Hardness HV: 560
```

Empty cells are omitted from a line; columns beyond the header row are named `column N`.

### CSV / TSV (`.csv .tsv`)
The delimiter is detected from the first 4 KB (comma, semicolon or tab). Blank rows are skipped, and rows become labelled lines exactly as for Excel. The metadata records the number of data `rows`.

## Adding a format

```python
from pathlib import Path
from ingestion import register_parser
from ingestion.parser import Parsed, Section

def parse_pptx(path: Path) -> Parsed:
    sections = [Section(text, {"slide": n}) for n, text in enumerate(read_slides(path), start=1)]
    return Parsed(sections, {"slides": len(sections)})

register_parser([".pptx"], parse_pptx)
```

Guidelines:

- Return **sections that match how people cite the format** (page, slide, sheet, heading). Chunking runs per section, and section metadata flows into every chunk.
- Raise `ParseError` with a reason a person can act on ("needs OCR", "password protected"), not a stack trace.
- Prefer the standard library or a light dependency; heavy libraries go behind an optional extra and a lazy import.
- Add a builder for the format to `tests/file_builders.py` and generate test files in code; never commit binary fixtures.

## Limitations

| Not handled | Workaround |
|---|---|
| Scanned PDFs and images | OCR (for example Amazon Textract) before ingestion |
| Legacy binary `.doc`, `.xls`, `.ppt` | Convert to `.docx`, `.xlsx`, `.pptx` |
| Password-protected files | Reported as failures |
| Charts, embedded images, merged-cell layout | Ignored; only text is indexed |
| Right-to-left and complex PDF layouts | Verify the extracted text |
