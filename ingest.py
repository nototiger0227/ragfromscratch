"""
Step 3 of indexing: turn each chunk into a vector (embedding) and store
it in a vector database.

WHAT IS AN EMBEDDING? A list of numbers (e.g. 1536 floats) produced by
a model, such that chunks with SIMILAR MEANING produce vectors that are
close together in that 1536-dimensional space. "Revenue grew 12%" and
"Sales increased twelve percent" end up near each other even though
they share almost no words — this is what lets semantic search beat
plain keyword (Ctrl+F) search.

WHAT IS A VECTOR DATABASE? A database built to answer "which of my
stored vectors are closest to THIS vector?" efficiently, even across
millions of vectors. We use Chroma because it's local, free, and needs
zero setup (it just writes files to disk) — great for learning.
"""
import chromadb
from google import genai

from config import (
    GOOGLE_API_KEY, EMBEDDING_MODEL, CHROMA_DIR, COLLECTION_NAME,
)
from extract import extract_text
from chunk import chunk_text

client = genai.Client(api_key=GOOGLE_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Call the embedding API for a batch of texts.

    Note: the free tier rate-limits requests per minute, not just per
    day. If you hit a 429 here on a big PDF, the fix is to embed in
    smaller batches with a short sleep between them, not a code bug.
    """
    result = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
    return [e.values for e in result.embeddings]


def ingest_pdf(pdf_path: str, doc_id: str):
    """
    doc_id: a short identifier for this document, e.g. "apple_2024".
    Used so we can later filter retrieval to one document, and so
    re-running ingestion on the same doc overwrites cleanly.

    Returns the list of chunks that were stored, so callers (like the
    web UI) can show exactly what got indexed without a second lookup.
    """
    print(f"[1/4] Extracting text from {pdf_path}...")
    text = extract_text(pdf_path)

    print("[2/4] Chunking...")
    chunks = chunk_text(text)
    print(f"      -> {len(chunks)} chunks")

    print("[3/4] Embedding chunks (calls Gemini API)...")
    vectors = embed_texts(chunks)

    print("[4/4] Storing in Chroma...")
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    # If this doc_id was ingested before, clear its old chunks first so
    # re-ingesting doesn't leave stale duplicates sitting in the store.
    existing = collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        embeddings=vectors,
        documents=chunks,
        metadatas=[{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))],
    )
    print(f"Done. '{doc_id}' is now searchable ({len(chunks)} chunks indexed).")

    return [
        {"id": f"{doc_id}_{i}", "chunk_index": i, "text": c, "char_count": len(c)}
        for i, c in enumerate(chunks)
    ]


def list_documents() -> list[dict]:
    """Return every doc_id currently stored, with how many chunks each has.
    This is what 'how are they stored' actually looks like under the hood:
    one flat collection, every chunk tagged with a doc_id in its metadata."""
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    all_rows = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in all_rows["metadatas"]:
        counts[meta["doc_id"]] = counts.get(meta["doc_id"], 0) + 1
    return [{"doc_id": doc_id, "chunk_count": n} for doc_id, n in counts.items()]


def get_document_chunks(doc_id: str) -> list[dict]:
    """Return all stored chunks for one document, in original order —
    the actual rows sitting in the vector DB right now."""
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    rows = collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    items = [
        {"id": id_, "chunk_index": meta["chunk_index"], "text": doc, "char_count": len(doc)}
        for id_, doc, meta in zip(rows["ids"], rows["documents"], rows["metadatas"])
    ]
    return sorted(items, key=lambda c: c["chunk_index"])


if __name__ == "__main__":
    # Usage: python ingest.py data/apple_2024.pdf apple_2024
    import sys
    ingest_pdf(sys.argv[1], sys.argv[2])