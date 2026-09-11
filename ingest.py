"""Indexing pipeline: text extraction, chunking, embedding, and storage."""
from pathlib import Path

import chromadb
from google import genai

from chunk import chunk_text
from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
)
from extract import extract_text

client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


def get_genai_client():
    """Create the Gemini client only when a valid API key exists."""
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to your .env file before running ingest or query."
        )
    if client is None:
        return genai.Client(api_key=GOOGLE_API_KEY)
    return client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the configured Gemini embedding model."""
    if not texts:
        return []
    result = get_genai_client().models.embed_content(model=EMBEDDING_MODEL, contents=texts)
    return [e.values for e in result.embeddings]


def ingest_document(file_path: str, doc_id: str):
    """Extract, chunk, embed, and store a PDF/DOCX/TXT document."""
    path = Path(file_path)
    source_type = path.suffix.lower().lstrip(".") or "unknown"

    print(f"[1/4] Extracting text from {path}...")
    text = extract_text(str(path))

    print("[2/4] Chunking...")
    chunks = chunk_text(text)
    print(f"      -> {len(chunks)} chunks")

    if not chunks:
        return []

    print("[3/4] Embedding chunks (calls Gemini API)...")
    vectors = embed_texts(chunks)

    print("[4/4] Storing in Chroma...")
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    existing = collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        embeddings=vectors,
        documents=chunks,
        metadatas=[{
            "doc_id": doc_id,
            "chunk_index": i,
            "source_type": source_type,
            "char_count": len(chunk),
            "file_name": path.name,
        } for i, chunk in enumerate(chunks)],
    )
    print(f"Done. '{doc_id}' is now searchable ({len(chunks)} chunks indexed).")

    return [
        {"id": f"{doc_id}_{i}", "chunk_index": i, "text": chunk, "char_count": len(chunk)}
        for i, chunk in enumerate(chunks)
    ]


def ingest_pdf(pdf_path: str, doc_id: str):
    """Backward-compatible wrapper for PDF ingestion."""
    return ingest_document(pdf_path, doc_id)


def list_documents() -> list[dict]:
    """Return every doc_id currently stored, with how many chunks each has."""
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    all_rows = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in all_rows["metadatas"]:
        counts[meta["doc_id"]] = counts.get(meta["doc_id"], 0) + 1
    return [{"doc_id": doc_id, "chunk_count": n} for doc_id, n in counts.items()]


def get_document_chunks(doc_id: str) -> list[dict]:
    """Return all stored chunks for one document, in original order."""
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    rows = collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    items = [
        {"id": id_, "chunk_index": meta["chunk_index"], "text": doc, "char_count": len(doc)}
        for id_, doc, meta in zip(rows["ids"], rows["documents"], rows["metadatas"])
    ]
    return sorted(items, key=lambda c: c["chunk_index"])


if __name__ == "__main__":
    import sys
    ingest_document(sys.argv[1], sys.argv[2])