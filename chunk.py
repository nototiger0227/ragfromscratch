"""
Step 2 of indexing: split text into chunks small enough to embed
meaningfully and retrieve precisely.

WHY CHUNK AT ALL? Two reasons:
1. Embedding models compress a passage into ONE fixed-size vector. Feed
   in an entire 100-page report and the vector becomes a mushy average
   of everything — useless for finding a specific fact. Feed in one
   paragraph and the vector actually represents that paragraph's
   meaning.
2. LLM context windows are limited (and cost more per token) — you want
   to retrieve only the few paragraphs that are actually relevant, not
   the whole document.

WHY OVERLAP CHUNKS? If a sentence explaining "Q3 revenue was $4.2B due
to..." gets cut in half at a chunk boundary, the reason gets separated
from the number. Overlap (repeating the last N characters of chunk i at
the start of chunk i+1) reduces the odds that a key fact gets split
across two chunks with neither having full context.

This version chunks by raw character count, which is simple but naive
— it can cut mid-sentence or mid-table-row. Stage 2 replaces this with
smarter, structure-aware chunking (split on paragraph/section
boundaries first, only falling back to hard cuts for oversized
sections).
"""
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # step forward less than chunk_size -> overlap
    return [c.strip() for c in chunks if c.strip()]


if __name__ == "__main__":
    sample = "A" * 50 + " " + "B" * 50 + " " + "C" * 50
    for i, c in enumerate(chunk_text(sample, chunk_size=60, overlap=20)):
        print(f"chunk {i}: {c[:30]}... (len={len(c)})")
