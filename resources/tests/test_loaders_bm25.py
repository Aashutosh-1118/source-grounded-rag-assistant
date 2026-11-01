import pytest
from langchain_core.documents import Document

import loaders
from bm25 import BM25Index, tokenize


def test_url_ids_ignore_trailing_slash_and_spaces():
    assert loaders.document_id_for_url("https://a.com/x/") == loaders.document_id_for_url(" https://a.com/x ")
    assert loaders.document_id_for_url("https://a.com/x") != loaders.document_id_for_url("https://a.com/y")


def test_file_ids_ignore_case():
    assert loaders.document_id_for_file("Report.PDF") == loaders.document_id_for_file("report.pdf")


@pytest.mark.parametrize("url", ["http://127.0.0.1/x", "http://10.0.0.5/x", "http://localhost/x", "ftp://example.com/x"])
def test_unsafe_urls_rejected(url):
    with pytest.raises(ValueError):
        loaders.validate_public_url(url)


def test_load_text_file(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# Title\nSome useful text.", encoding="utf-8")
    docs = loaders.load_file(f, "notes.md")
    assert docs[0].metadata == {"source": "notes.md", "title": "notes.md", "source_type": "file"}


def test_empty_and_unsupported_files(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   ")
    with pytest.raises(ValueError):
        loaders.load_file(empty, "empty.txt")
    other = tmp_path / "x.exe"
    other.write_bytes(b"x")
    with pytest.raises(ValueError):
        loaders.load_file(other, "x.exe")


def test_load_pdf_has_page_numbers(tmp_path):
    fpdf = pytest.importorskip("fpdf")
    pdf = fpdf.FPDF()
    for text in ("First page text about mortgages.", "Second page text about rent."):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, text)
    path = tmp_path / "doc.pdf"
    pdf.output(str(path))
    docs = loaders.load_file(path, "doc.pdf")
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert "mortgages" in docs[0].page_content


def test_load_docx(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("Rent rose four percent.")
    path = tmp_path / "doc.docx"
    d.save(str(path))
    docs = loaders.load_file(path, "doc.docx")
    assert "Rent rose" in docs[0].page_content


def test_tokenize_keeps_numbers():
    assert "6.8" in tokenize("The rate was 6.8 percent.")
    assert "1,450" in tokenize("rent of 1,450 dollars")
    assert "the" not in tokenize("the rate")


def test_bm25_ranks_exact_terms_first():
    docs = [
        Document(page_content="General notes about housing markets."),
        Document(page_content="The vault code is 4417."),
        Document(page_content="Rates changed in March."),
    ]
    hits = BM25Index(docs).search("vault code", k=3)
    assert hits[0][0].page_content.startswith("The vault")
    assert all(score > 0 for _, score in hits)


def test_bm25_no_match_returns_nothing():
    docs = [Document(page_content="alpha beta gamma")]
    assert BM25Index(docs).search("zebra", k=3) == []
    assert BM25Index([]).search("anything", k=3) == []
