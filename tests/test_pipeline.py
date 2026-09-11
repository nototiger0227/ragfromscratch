from pathlib import Path

from docx import Document

from chunk import chunk_text
from extract import extract_text
from insights import extract_financial_insights
from query import bm25_rank


def test_extract_text_handles_txt(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("This is a plain text file.\nIt contains multiple lines.")

    result = extract_text(str(path))

    assert "plain text file" in result
    assert "multiple lines" in result


def test_extract_text_handles_docx(tmp_path):
    path = tmp_path / "sample.docx"
    doc = Document()
    doc.add_paragraph("Quarterly revenue was $20M.")
    doc.add_paragraph("The company expanded into Europe.")
    doc.save(path)

    result = extract_text(str(path))

    assert "Quarterly revenue was $20M." in result
    assert "expanded into Europe" in result


def test_chunk_text_keeps_headings_and_respects_overlap():
    text = "# Revenue\nThe company generated significant revenue in 2024.\n\n# Risk\nSupply chain interruptions remain a key concern."

    chunks = chunk_text(text, chunk_size=80, overlap=20)

    assert len(chunks) > 1
    assert any("Revenue" in chunk for chunk in chunks)
    assert any("Risk" in chunk for chunk in chunks)
    assert all(len(chunk) <= 80 for chunk in chunks)


def test_bm25_rank_prefers_exact_keyword_match():
    docs = [
        "The company reported revenue growth of 12 percent.",
        "Supply chain risk remains a key concern for operations.",
    ]

    ranked = bm25_rank("revenue growth", docs)

    assert ranked[0]["text"] == docs[0]
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_extract_financial_insights_groups_key_signals():
    text = (
        "Revenue increased to $20M. Net income was $4M. Total debt was $8M.\n"
        "Key risks include supply chain disruption. The outlook expects steady growth."
    )

    result = extract_financial_insights(text)

    assert any("Revenue increased" in item for item in result["revenue"])
    assert any("Net income" in item for item in result["profit"])
    assert any("Total debt" in item for item in result["debt"])
    assert any("Key risks" in item for item in result["risks"])
    assert any("outlook" in item.lower() for item in result["outlook"])
