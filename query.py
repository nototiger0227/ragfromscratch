"""Query pipeline with hybrid retrieval and evidence-backed answers."""
import re
import time

import chromadb
from google import genai
from google.genai import types

from config import CHAT_MODEL, CHROMA_DIR, COLLECTION_NAME, GOOGLE_API_KEY, TOP_K
from ingest import embed_texts

client = genai.Client(api_key=GOOGLE_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


def lexical_overlap_score(question: str, text: str) -> float:
    """Simple keyword-overlap score used to rerank the dense retrieval candidates."""
    question_tokens = set(re.findall(r"\b\w+\b", question.lower()))
    text_tokens = set(re.findall(r"\b\w+\b", text.lower()))
    if not question_tokens:
        return 0.0
    return len(question_tokens & text_tokens) / len(question_tokens)


def retrieve(question: str, doc_id: str | None = None, top_k: int = TOP_K):
    """Hybrid retrieval: dense semantic search + lightweight lexical reranking."""
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    [question_vector] = embed_texts([question])

    where_filter = {"doc_id": doc_id} if doc_id else None
    candidate_count = max(top_k * 5, top_k)

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=candidate_count,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results.get("distances", [[0.0] * len(docs)])[0]

    reranked = []
    for text, meta, distance in zip(docs, metas, distances):
        lexical_score = lexical_overlap_score(question, text)
        semantic_score = 1.0 / (1.0 + distance) if distance is not None else 0.0
        combined_score = (0.65 * semantic_score) + (0.35 * lexical_score)
        reranked.append((combined_score, text, meta))

    reranked.sort(key=lambda item: item[0], reverse=True)
    return [(text, meta) for _, text, meta in reranked[:top_k]]


SYSTEM_PROMPT = """You are a financial document analysis assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain the answer, say so explicitly —
do not guess or use outside knowledge. When you use a number or fact,
mention which chunk/page it came from if that's indicated in the context.
"""


def ask(question: str, doc_id: str | None = None) -> dict:
    """Retrieve relevant evidence, send it to the LLM, and return answer + sources."""
    retrieved = retrieve(question, doc_id=doc_id)

    context_block = "\n\n".join(
        f"[Chunk {meta['chunk_index']} from {meta['doc_id']} ({meta.get('source_type', 'unknown')})]\n{text}"
        for text, meta in retrieved
    )

    user_prompt = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
        ),
    )

    return {
        "answer": response.text,
        "sources": [
            {
                "chunk_id": f"{meta['doc_id']}_{meta['chunk_index']}",
                "doc_id": meta["doc_id"],
                "chunk_index": meta["chunk_index"],
                "source_type": meta.get("source_type", "unknown"),
                "text": text,
            }
            for text, meta in retrieved
        ],
    }


def benchmark_retrieval(question_bank: list[str], doc_id: str | None = None, top_k: int = TOP_K) -> dict:
    """Measure average latency and result counts for a set of test questions."""
    latencies: list[float] = []
    hits = 0

    for question in question_bank:
        start = time.perf_counter()
        results = retrieve(question, doc_id=doc_id, top_k=top_k)
        latencies.append(time.perf_counter() - start)
        hits += len(results)

    return {
        "question_count": len(question_bank),
        "average_latency_seconds": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        "total_retrieved_chunks": hits,
        "average_chunks_per_query": round(hits / len(question_bank), 2) if question_bank else 0.0,
    }


def answer(question: str, doc_id: str | None = None) -> str:
    """Thin wrapper for the CLI, which just wants the text."""
    return ask(question, doc_id=doc_id)["answer"]


if __name__ == "__main__":
    print("Ask a question about your ingested document(s). Ctrl+C to quit.\n")
    while True:
        q = input("> ")
        print("\n" + answer(q) + "\n")