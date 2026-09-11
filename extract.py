"""
Step 1 of indexing: turn a PDF into plain text.

We use PyMuPDF (imported as `fitz`) because it's fast and handles most
real-world PDFs well. It's NOT great at preserving table structure —
that's a known weakness we'll address in Stage 2 with pymupdf4llm,
which converts to Markdown and keeps tables intact. For now, plain
text is enough to learn the pipeline.
"""
import pymupdf as fitz  # Updated import


def extract_text(pdf_path: str) -> str:
    """Return the full text of a PDF as one big string, page by page."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        # Tagging each page's text with its page number now, while we
        # still know it, means later we can tell the user "this answer
        # came from page 12" instead of just dumping an answer with no
        # source.
        pages.append(f"\n\n--- Page {page_num} ---\n\n{text}")
    doc.close()
    return "".join(pages)


if __name__ == "__main__":
    # Quick manual test: python extract.py data/sample.pdf
    import sys
    text = extract_text(sys.argv[1])
    print(text[:2000])
    print(f"\n\n[...{len(text)} characters total]")
