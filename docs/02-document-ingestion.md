# Document Ingestion

Ingestion turns files of different formats into `Document`s: plain text plus metadata, split into **sections** that become citation units.

```bash
python scripts/index.py --source path/to/folder     # ingest, report failures, sync the index
```

## Supported formats

| Format | Suffixes | Sections | Metadata |
|---|---|---|---|
| Plain text, logs, JSON | `.txt .text .log .json` | one | — |
| Markdown | `.md .markdown` | one per heading | `title` (first H1), `heading` |
| HTML | `.html .htm` | one per `h1`–`h3` | `title` (`<title>`), `heading` |
| PDF | `.pdf` | one per page with text | `pages`, `page` |
| Word | `.docx` | one per Heading/Title paragraph | `title`, `heading`; tables as `a \| b` rows |
| Excel | `.xlsx` | one per sheet | `sheets`, `sheet`; rows as `Header: value; …` |
| CSV / TSV | `.csv .tsv` | one | `rows`; rows as `Header: value; …` |

Office formats are zip archives of XML and are read with the standard library, so no extra dependency is needed. PDF uses `pypdf`. Register another format with `ingestion.register_parser([".ext"], parse_fn)`.

## Why sections

Chunking runs **per section**, so a chunk never spans two pages or two sheets, and every chunk carries its section's metadata into the vector store:

```
lot_results.xlsx, sheet Q3 lots
forming_spec.docx, section "Acceptance"
Elgiloy_Havar_Material_Requirements_for_Diaphragm.pdf, page 2
```

`scripts/query.py` prints sources in this form. Chunk offsets (`start`, `end`) are relative to the section.

## Tables become labelled rows

A spreadsheet row `L-302 | HAVAR | reject | tearing at the rim` is stored as

```
Lot: L-302; Alloy: HAVAR; Result: reject; Reason: tearing at the rim
```

so a chunk keeps the meaning of every value even when the header row ends up in another chunk.

## Failure handling

`ingest(paths)` never stops at the first bad file. It returns an `IngestResult`:

| Field | Contents |
|---|---|
| `documents` | Everything that loaded |
| `failures` | `(source, reason)`: corrupted Office files, unsupported explicit files, PDFs without a text layer ("needs OCR") |
| `skipped` | Hidden files, Office lock files (`~$report.docx`) and unsupported types found while scanning folders |
| `duplicates` | `(duplicate, original)` pairs with identical content under different names |

`scripts/index.py` passes failed files to `SearchIndex.sync(..., keep=...)`, so a file that fails to parse this time **keeps its previously indexed version** instead of being deleted from the index.

`load_documents(paths)` is the strict variant: it raises if any file fails, and is used where a silent gap would be wrong (for example the evaluation corpus).

Text files are decoded as UTF-8 (with or without BOM), falling back to Windows-1252 and then Latin-1, so legacy exports load instead of crashing.

## Known limits

- **Scanned PDFs** have no text layer and are reported as failures; OCR (for example Amazon Textract) is needed.
- **Legacy `.doc` / `.xls`** (pre-2007 binary formats) are not supported; convert them to `.docx` / `.xlsx`.
- XLSX formulas are read as their cached values; merged cells and charts are ignored.
- Images inside documents are ignored.
