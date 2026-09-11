"""Query pipeline with hybrid retrieval and evidence-backed answers."""
import re
import time

import chromadb
from google import genai
from google.genai import types

from config import CHAT_MODEL, CHROMA_DIR, COLLECTION_NAME, GOOGLE_API_KEY, TOP_K
from ingest import embed_texts

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


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def bm25_rank(question: str, documents: list[str], top_k: int | None = None) -> list[dict]:
    """Compute a lightweight BM25 score for lexical retrieval candidates."""
    if not documents:
        return []

    q_tokens = tokenize(question)
    if not q_tokens:
        return [{"text": doc, "score": 0.0} for doc in documents[: top_k or len(documents)]]

    doc_tokens = [tokenize(doc) for doc in documents]
    docs_with_terms = [set(tokens) for tokens in doc_tokens]
    n_docs = len(documents)
    avgdl = sum(len(tokens) for tokens in doc_tokens) / n_docs
    k1 = 1.5
    b = 0.75

    def idf(term: str) -> float:
        df = sum(1 for doc_terms in docs_with_terms if term in doc_terms)
        return max(0.0, (n_docs - df + 0.5) / (df + 0.5))

    scores: list[float] = []
    for tokens in doc_tokens:
        score = 0.0
        term_counts = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1
        for token in q_tokens:
            if token not in term_counts:
                continue
            tf = term_counts[token]
            denom = tf + k1 * (1 - b + b * (len(tokens) / avgdl if avgdl else 1.0))
            score += idf(token) * ((tf * (k1 + 1)) / denom)
        scores.append(score)

    ranked = [
        {"text": doc, "score": round(score, 6)}
        for doc, score in zip(documents, scores)
    ]
    ranked.sort(key=lambda item: item["score"], reverse=True)
    if top_k is not None:
        return ranked[:top_k]
    return ranked


def lexical_overlap_score(question: str, text: str) -> float:
    """Simple keyword-overlap score used to rerank the dense retrieval candidates."""
    question_tokens = set(tokenize(question))
    text_tokens = set(tokenize(text))
    if not question_tokens:
        return 0.0
    return len(question_tokens & text_tokens) / len(question_tokens)


def retrieve(question: str, doc_id: str | None = None, top_k: int = TOP_K):
    """Hybrid retrieval: dense semantic search + lexical BM25 reranking."""
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

    dense_candidates = []
    for text, meta, distance in zip(docs, metas, distances):
        lexical_score = lexical_overlap_score(question, text)
        semantic_score = 1.0 / (1.0 + distance) if distance is not None else 0.0
        dense_candidates.append({
            "text": text,
            "meta": meta,
            "dense_score": 0.65 * semantic_score + 0.35 * lexical_score,
        })

    bm25_candidates = bm25_rank(question, [item["text"] for item in dense_candidates], top_k=len(dense_candidates))
    bm25_by_text = {item["text"]: item["score"] for item in bm25_candidates}

    hybrid = []
    for item in dense_candidates:
        text = item["text"]
        bm25_score = bm25_by_text.get(text, 0.0)
        combined_score = item["dense_score"] + (0.5 * bm25_score)
        hybrid.append((combined_score, text, item["meta"]))

    hybrid.sort(key=lambda item: item[0], reverse=True)
    return [(text, meta) for _, text, meta in hybrid[:top_k]]


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

    response = get_genai_client().models.generate_content(
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