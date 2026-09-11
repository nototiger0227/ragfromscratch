"""Document extraction utilities for PDFs, DOCX, and TXT files."""
from pathlib import Path

import pymupdf as fitz
from docx import Document


def extract_pdf_text(pdf_path: str) -> str:
    """Return the full text of a PDF as one big string, page by page."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        pages.append(f"\n\n--- Page {page_num} ---\n\n{text}")
    doc.close()
    return "".join(pages)


def extract_docx_text(docx_path: str) -> str:
    """Return the text content from a DOCX file."""
    document = Document(docx_path)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_txt_text(txt_path: str) -> str:
    """Return the contents of a plain text file."""
    return Path(txt_path).read_text(encoding="utf-8")


def extract_text(file_path: str) -> str:
    """Dispatch to the right parser based on the file extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_text(str(path))
    if suffix == ".docx":
        return extract_docx_text(str(path))
    if suffix in {".txt", ".md", ".csv"}:
        return extract_txt_text(str(path))

    raise ValueError(f"Unsupported file type: {suffix}. Supported: PDF, DOCX, TXT")


if __name__ == "__main__":
    import sys
    text = extract_text(sys.argv[1])
    print(text[:2000])
    print(f"\n\n[...{len(text)} characters total]")
