"""
The query-time half of RAG: given a question, retrieve relevant chunks
and ask the LLM to answer using ONLY those chunks.

This is "R-A-G" spelled out:
  Retrieval    -> collection.query() below
  Augmented    -> we stuff retrieved chunks into the prompt
  Generation   -> client.chat.completions.create() below
"""
import chromadb
from google import genai
from google.genai import types

from config import GOOGLE_API_KEY, EMBEDDING_MODEL, CHAT_MODEL, CHROMA_DIR, COLLECTION_NAME, TOP_K
from ingest import embed_texts

client = genai.Client(api_key=GOOGLE_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


def retrieve(question: str, doc_id: str | None = None, top_k: int = TOP_K):
    """Embed the question, then ask Chroma for the top_k closest chunks."""
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    [question_vector] = embed_texts([question])

    where_filter = {"doc_id": doc_id} if doc_id else None  # optional: restrict to one document

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=top_k,
        where=where_filter,
    )
    # Chroma returns parallel lists; zip them into (chunk_text, metadata) pairs
    return list(zip(results["documents"][0], results["metadatas"][0]))


SYSTEM_PROMPT = """You are a financial document analysis assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain the answer, say so explicitly —
do not guess or use outside knowledge. When you use a number or fact,
mention which chunk/page it came from if that's indicated in the context.
"""


def ask(question: str, doc_id: str | None = None) -> dict:
    """Run the full retrieve -> generate pipeline and return BOTH the
    answer and the exact chunks that were used to produce it. Returning
    the evidence alongside the answer (rather than just the text) is
    what makes retrieval debuggable/visible instead of a black box."""
    retrieved = retrieve(question, doc_id=doc_id)

    # Build the "augmented" context block that gets inserted into the prompt
    context_block = "\n\n".join(
        f"[Chunk {meta['chunk_index']} from {meta['doc_id']}]\n{text}"
        for text, meta in retrieved
    )

    user_prompt = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,  # 0 = deterministic, favors sticking to facts over creativity
        ),
    )

    return {
        "answer": response.text,
        "sources": [
            {"chunk_id": f"{meta['doc_id']}_{meta['chunk_index']}",
             "doc_id": meta["doc_id"], "chunk_index": meta["chunk_index"], "text": text}
            for text, meta in retrieved
        ],
    }


def answer(question: str, doc_id: str | None = None) -> str:
    """Thin wrapper for the CLI, which just wants the text."""
    return ask(question, doc_id=doc_id)["answer"]


if __name__ == "__main__":
    print("Ask a question about your ingested document(s). Ctrl+C to quit.\n")
    while True:
        q = input("> ")
        print("\n" + answer(q) + "\n")