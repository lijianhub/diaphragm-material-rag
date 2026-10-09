import pytest

from ingestion.parser import ParseError, parse_file, supported_suffixes
from tests.file_builders import write_docx, write_pdf, write_xlsx


def test_supported_formats_cover_common_enterprise_documents():
    assert {".txt", ".md", ".html", ".htm", ".pdf", ".docx", ".xlsx", ".csv", ".json"} <= set(supported_suffixes())


def test_text_handles_utf8_bom_and_legacy_windows_encoding(tmp_path):
    bom = tmp_path / "bom.txt"
    bom.write_bytes("﻿HAVAR foil".encode("utf-8"))
    legacy = tmp_path / "legacy.txt"
    legacy.write_bytes("Temperature 500 \xb0C".encode("cp1252"))

    assert parse_file(bom).sections[0].text == "HAVAR foil"
    assert parse_file(legacy).sections[0].text == "Temperature 500 °C"


def test_markdown_splits_on_headings_and_takes_title_from_h1(tmp_path):
    path = tmp_path / "procedure.md"
    path.write_text("# Aging Procedure\nIntro line.\n\n## Furnace\nUse vacuum.\n\n## Cooling\nAir cool.\n", encoding="utf-8")

    parsed = parse_file(path)

    assert parsed.metadata["title"] == "Aging Procedure"
    assert [s.metadata.get("heading") for s in parsed.sections] == ["Aging Procedure", "Furnace", "Cooling"]
    assert "Use vacuum." in parsed.sections[1].text


def test_html_drops_scripts_and_styles_and_splits_on_headings(tmp_path):
    path = tmp_path / "faq.html"
    path.write_text(
        "<html><head><title>Supplier FAQ</title><style>p {color: red}</style></head><body>"
        "<nav>Home | About</nav><h1>Lead times</h1><p>Strip ships in 6&nbsp;weeks &amp; wire in 4.</p>"
        "<script>track()</script><h2>Certificates</h2><p>Every lot has a certificate.</p></body></html>",
        encoding="utf-8",
    )

    parsed = parse_file(path)
    full = "\n".join(s.text for s in parsed.sections)

    assert parsed.metadata["title"] == "Supplier FAQ"
    assert "track()" not in full and "color: red" not in full and "Home | About" not in full
    assert "6 weeks & wire in 4." in full
    assert [s.metadata.get("heading") for s in parsed.sections] == ["Lead times", "Certificates"]


def test_docx_reads_paragraphs_headings_and_tables(tmp_path):
    path = tmp_path / "spec.docx"
    write_docx(
        path,
        [
            ("h1", "Material Specification"),
            ("p", "Strip must be cold rolled."),
            ("h2", "Tolerances"),
            ("table", [["Property", "Limit"], ["Thickness", "0.05 mm"]]),
        ],
    )

    parsed = parse_file(path)

    assert parsed.metadata["title"] == "Material Specification"
    assert [s.metadata["heading"] for s in parsed.sections] == ["Material Specification", "Tolerances"]
    assert "Strip must be cold rolled." in parsed.sections[0].text
    assert "Property | Limit" in parsed.sections[1].text
    assert "Thickness | 0.05 mm" in parsed.sections[1].text


def test_xlsx_makes_one_section_per_sheet_with_header_labelled_rows(tmp_path):
    path = tmp_path / "properties.xlsx"
    write_xlsx(
        path,
        {
            "Alloys": [["Alloy", "Base", "Hardness HV"], ["ELGILOY", "Cobalt", 560], ["HAVAR", "Cobalt", None]],
            "Notes": [["Note"], ["Age after forming"]],
        },
    )

    parsed = parse_file(path)

    assert [s.metadata["sheet"] for s in parsed.sections] == ["Alloys", "Notes"]
    assert "Alloy: ELGILOY; Base: Cobalt; Hardness HV: 560" in parsed.sections[0].text
    assert "Alloy: HAVAR; Base: Cobalt" in parsed.sections[0].text
    assert "Note: Age after forming" in parsed.sections[1].text


def test_csv_rows_are_labelled_with_their_headers(tmp_path):
    path = tmp_path / "lots.csv"
    path.write_text("Lot,Alloy,Result\nL-101,ELGILOY,pass\nL-102,HAVAR,fail\n", encoding="utf-8")

    text = parse_file(path).sections[0].text

    assert "Lot: L-101; Alloy: ELGILOY; Result: pass" in text
    assert "Lot: L-102; Alloy: HAVAR; Result: fail" in text


def test_pdf_makes_one_section_per_page_with_page_numbers(tmp_path):
    path = tmp_path / "manual.pdf"
    write_pdf(path, ["Forming pressure limits", "", "Springback compensation"])

    parsed = parse_file(path)

    assert [s.metadata["page"] for s in parsed.sections] == [1, 3]  # the blank page is dropped
    assert "Springback compensation" in parsed.sections[1].text
    assert parsed.metadata["pages"] == 3


def test_pdf_without_any_text_is_reported_as_needing_ocr(tmp_path):
    path = tmp_path / "scanned.pdf"
    write_pdf(path, ["", ""])

    with pytest.raises(ParseError, match="OCR"):
        parse_file(path)


def test_corrupted_office_file_raises_parse_error(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"this is not a zip archive")

    with pytest.raises(ParseError):
        parse_file(path)


def test_unsupported_format_raises_parse_error(tmp_path):
    path = tmp_path / "drawing.dwg"
    path.write_bytes(b"\x00\x01")

    with pytest.raises(ParseError, match="unsupported"):
        parse_file(path)
