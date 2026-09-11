# FinRAG — Detailed Implementation and Technical Documentation

## 1. Project Overview

FinRAG is a lightweight Retrieval-Augmented Generation (RAG) system built for querying financial and document-based content using semantic search and grounded large language model responses. The project is designed as a practical, production-oriented learning build: it combines a document ingestion pipeline, vector storage, hybrid retrieval, and a web API around a simple AI QA workflow.

The system takes unstructured documents, extracts text, splits it into meaningful chunks, embeds those chunks into a vector space, stores them in a local vector database, retrieves context based on a user question, and then asks an LLM to answer using only the retrieved evidence.

This project is intentionally structured so that each stage of the RAG pipeline is isolated and understandable:

- Extraction: Document -> raw text
- Chunking: Text -> semantically coherent chunks
- Embedding: Chunks -> high-dimensional vectors
- Storage: Vectors -> Chroma local database
- Retrieval: Query -> nearest relevant chunks
- Generation: Chunks + prompt -> grounded answer

---

## 2. Why This Project Exists

The purpose of FinRAG is to demonstrate a realistic, end-to-end RAG pipeline without relying on a large external application stack. It focuses on the fundamental concepts behind modern document question answering:

- semantic retrieval
- chunking strategy
- vector search
- evidence-grounded generation
- citation-backed answers
- production-friendly API interfaces

The project is especially relevant for financial documents because such documents often contain dense, fact-heavy, highly structured text where retrieval quality matters more than raw model creativity.

---

## 3. High-Level Architecture

```mermaid
flowchart LR
    A[User Upload / Query] --> B[FastAPI App]
    B --> C[Document Extractor]
    C --> D[Chunking Layer]
    D --> E[Embedding Model]
    E --> F[Chroma Vector DB]
    F --> G[Hybrid Retrieval]
    G --> H[LLM Answer Generation]
    H --> I[Response + Source Evidence]
```

### Core architectural components

1. Document ingestion layer
   - Accepts uploaded files
   - Normalizes them to plain text
   - Splits text into manageable chunks

2. Embedding and indexing layer
   - Converts each chunk to an embedding vector
   - Stores metadata like document id, chunk index, content length, source type

3. Retrieval layer
   - Uses semantic similarity for dense retrieval
   - Uses lexical overlap and BM25-style reranking for keyword-aware matching

4. Generation layer
   - Injects retrieved evidence into an LLM prompt
   - Requests the model to answer only from the provided context

5. API and UI layer
   - FastAPI exposes upload, document listing, chunk retrieval, and Q&A endpoints
   - Frontend displays fetched chunks and evidence highlights

---

## 4. Phase-wise Implementation Journey

## Phase 1 — Minimal End-to-End Pipeline

### What was implemented

The first version of the project established the full RAG flow for a single document type: PDF ingestion.

Files involved:

- config.py
- extract.py
- chunk.py
- ingest.py
- query.py
- app.py

### Purpose

This phase proved the concept end-to-end:

- extract text from a PDF
- split it into chunks
- embed chunks
- save them in Chroma
- retrieve them for a question
- send them to an LLM
- return an answer

### Core idea

The system is designed around the classic retrieval pipeline:

- Document -> chunks
- chunks -> embeddings
- query -> nearest vectors
- nearest chunks -> LLM prompt context
- model -> grounded answer

### Why it matters

This phase established the foundation for any RAG project. Before adding advanced features, it demonstrates the essential flow: retrieval and generation are separate concerns, and retrieval quality is often the most important factor in answer quality.

---

## Phase 2 — Multi-Format Document Extraction

### What changed

The extraction layer was upgraded to support multiple document types:

- PDF
- DOCX
- TXT

### Files changed

- extract.py

### Implementation details

The extractor now dispatches based on file extension:

- .pdf -> PyMuPDF extraction
- .docx -> python-docx paragraph extraction
- .txt -> plain text parsing

### Why this matters

Most real-world document corpora contain mixed file types. Supporting multiple formats increases the practical value of the project and makes it more realistic for portfolio use. It also reduces project brittleness because the indexer no longer depends on a single file format.

### Theoretical concept

Document parsing is the first step of any information retrieval pipeline. Without reliable extraction, the downstream chunking and embedding stages are useless. A clean extractor normalizes different source formats into a single plain-text representation, which keeps the rest of the pipeline consistent.

---

## Phase 3 — Section-Aware Chunking

### What changed

The original project used simple fixed-size character splitting. This was replaced with section-aware chunking, which identifies natural boundaries before applying overlap-based chunk splitting.

### Files changed

- chunk.py

### Implementation details

The new chunker:

- normalizes whitespace
- splits text into logical sections using blank lines and heading-like patterns
- chunks section-by-section rather than treating the full document as one continuous string
- applies overlap to prevent important context from being split across boundaries

### Why it matters

Chunking is one of the most important decisions in RAG performance. If chunks are too small, they lose context. If they are too large, embeddings become blurry and less precise.

### Theoretical concept

The embedding model compresses a chunk into a single high-dimensional vector, so each chunk should represent a compact unit of meaning. Section-aware chunking is more semantically coherent than arbitrary character slicing because it respects natural document boundaries such as headings, paragraphs, and topical blocks.

### Observed effect

This improves retrieval quality because the stored vectors align better with human-recognizable document structures. In financial reports, where facts often appear in paragraphs or subheadings, semantically meaningful chunks outperform arbitrary fixed-length cuts.

---

## Phase 4 — Metadata Enrichment and Document Tracking

### What changed

Each indexed chunk now stores richer metadata and source details.

Example metadata stored for each chunk:

- doc_id
- chunk_index
- source_type
- char_count
- file_name

### Files changed

- ingest.py

### Why this matters

Metadata makes retrieval and debugging much more practical. It allows:

- filtering by a specific document
- identifying chunk provenance
- tracing which source text contributed to an answer
- debugging retrieval failures more quickly

### Theoretical concept

Vector search is powerful, but retrieval becomes much more useful when metadata is attached to each vector. Metadata acts as a control layer: it allows users and systems to constrain retrieval to a document, file type, or business scope without losing the vector-based semantic ranking capability.

---

## Phase 5 — Hybrid Retrieval: Dense + Lexical Ranking

### What changed

The retrieval pipeline moved from a pure semantic search to a hybrid setup:

- vector similarity search for semantic meaning
- lexical scoring for exact keyword overlap
- reranking of candidate chunks using a combined score

### Files changed

- query.py

### Implementation details

The hybrid retrieval process:

1. embed the user question
2. query Chroma for the nearest candidate chunks
3. compute a dense similarity score
4. compute a lexical overlap score from the question terms
5. combine both scores to rerank candidates
6. return the top-k chunks

This is more effective than relying on embeddings alone because some queries are exact-term-driven; others are semantic and conceptual.

### Theoretical concept

Dense retrieval is good at paraphrase and concept matching, while lexical methods are good at exact terminology and rare technical terms. Hybrid retrieval combines the strengths of both approaches.

This is especially useful in financial documents because questions often include:

- exact company names
- tickers
- legal terms
- financial metrics
- unusual product names

Pure vector search may underperform on exact-term recall, while BM25-like lexical ranking helps recover those direct matches.

---

## Phase 6 — Retrieval Benchmarking and Latency Measurement

### What changed

A benchmark utility was added to measure retrieval behavior quantitatively.

### Files changed

- benchmark.py
- query.py

### What metrics are available

Current benchmark output includes:

- number of questions evaluated
- average latency per query
- total retrieved chunks
- average chunks per query

### Why this matters

This is important for a real production-ready system because it allows comparison of retrieval strategies and tuning of chunk size, top-k values, and reranking heuristics. It moves the project from “works once” to “measurable and tunable.”

### Theoretical concept

In information retrieval, performance is not just about answer quality. It also depends on:

- latency
- cost per query
- number of documents retrieved
- ranking quality
- recall and precision tradeoffs

A system that retrieves the right chunk in a lower latency and with less token waste is typically more useful in production settings.

---

## Phase 7 — API Layer and UI Integration

### What was implemented

A FastAPI application exposes the pipeline as a web service and serves the frontend.

### Files changed

- app.py
- frontend/index.html

### API endpoints

#### 1. Upload and ingest a document

```http
POST /api/ingest
Content-Type: multipart/form-data
```

Form fields:

- file: uploaded document
- doc_id: custom short identifier for uploaded document

Example:

```bash
curl -X POST "http://localhost:8000/api/ingest" \
  -F "file=@data/sample.pdf" \
  -F "doc_id=sample_doc"
```

Response:

```json
{
  "doc_id": "sample_doc",
  "chunk_count": 21,
  "chunks": [
    {
      "id": "sample_doc_0",
      "chunk_index": 0,
      "text": "...",
      "char_count": 812
    }
  ]
}
```

#### 2. List indexed documents

```http
GET /api/documents
```

Response:

```json
{
  "documents": [
    {"doc_id": "sample_doc", "chunk_count": 21}
  ]
}
```

#### 3. Get chunks for a specific document

```http
GET /api/documents/{doc_id}/chunks
```

#### 4. Ask a question

```http
POST /api/query
Content-Type: application/json
```

Example request:

```json
{
  "question": "What was the total revenue?",
  "doc_id": "sample_doc"
}
```

Example response:

```json
{
  "answer": "Total revenue was $12.4 billion.",
  "sources": [
    {
      "chunk_id": "sample_doc_7",
      "doc_id": "sample_doc",
      "chunk_index": 7,
      "source_type": "pdf",
      "text": "..."
    }
  ]
}
```

### UI functionality

The frontend includes:

- file upload interface
- document list panel
- chunk catalog browsing
- query input
- evidence chips showing the retrieved chunks used to answer
- clickable chunk highlighting to quickly inspect evidence

This makes the project more demonstrable in interviews and more effective as a portfolio item.

---

## 5. Theoretical Concepts Behind the Implementation

## 5.1 Embeddings

Embeddings are dense vector representations of text. They capture semantics: texts with similar meanings become located near each other in vector space even when the literal words differ.

In this project, each chunk is encoded by a Gemini embedding model. This allows semantic search to retrieve relevant chunks based on concept similarity rather than only exact keyword matches.

## 5.2 Vector Databases

A vector database stores embeddings and supports nearest-neighbor retrieval. Chroma is used here because it is simple, local, and easy to understand.

Its value is in efficiently performing:

- nearest-neighbor lookup
- document filtering
- metadata retrieval
- persistent storage

## 5.3 Retrieval-Augmented Generation

RAG combines two systems:

1. A retrieval system finds relevant context
2. A generative language model produces an answer from that context

This is useful because it grounds model output in documents rather than relying entirely on memory or training data.

## 5.4 Chunking

Chunking is essential because embedding a whole 100-page document into one vector leads to poor representation and diluted semantics. Small, focused chunks help represent the important facts while keeping retrieval precise.

Better chunking usually improves performance because the retrieved evidence is both relevant and compact.

## 5.5 BM25 and Lexical Retrieval

BM25 is a classic ranking function for text retrieval. It rewards token overlap and term importance while penalizing overly common words. In a hybrid retrieval setup, it complements semantic retrieval and helps with exact-phrase and technical-term queries.

## 5.6 Grounding and Citation

Grounded generation means the model is constrained to answer from supplied context. This is critical for reliability in financial or business document QA, where factual accuracy matters and hallucination risk is high.

By returning the exact source chunks used in the answer, the project allows the user to verify the evidence and debug retrieval quality.

---

## 6. What Changed in the Project Over Time

### Before the upgrades

The original project:

- supported only PDF ingestion
- used fixed-size chunking
- relied on semantic retrieval alone
- lacked advanced metadata handling
- had limited explanation of retrieval decisions
- had a basic frontend and minimal evidence tracking

### After the upgrades

The project now:

- accepts PDF, DOCX, and TXT inputs
- uses section-aware chunking with overlap
- maintains richer metadata per chunk
- uses hybrid retrieval prioritizing semantic and lexical relevance
- includes benchmarking utilities for latency and query behavior
- serves a more usable interface with click-to-source evidence
- provides better developer debugging and project storytelling value

---

## 7. Current Project Strengths

This project is now strong enough to serve as a real portfolio item because it demonstrates:

- product-fluent AI engineering
- document ingestion and preprocessing
- retrieval pipeline implementation
- vector database usage
- LLM grounding
- API design
- frontend integration
- evidence-based answer generation
- evaluation awareness and benchmarking mindset

---

## 8. Current Limitations and Future Improvements

Even with these upgrades, the project still has room to evolve:

1. More advanced chunking
   - section-aware chunking is better than naive slicing, but structure-aware splitting on tables and headings could be improved further

2. Better reranking
   - a cross-encoder reranker could re-score a larger candidate pool for stronger ranking quality

3. Hybrid retrieval optimization
   - BM25 and dense retrieval can be tuned with weighting experiments and evaluation sets

4. Document-level metadata filters
   - additional metadata like title, company name, year, document type could improve filtering and routing

5. Production hardening
   - authentication, logging, background jobs, and deployment configuration would be needed for production use

---

## 9. Recommended Local Workflow

### Setup

```bash
cd /workspaces/ragfromscratch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variable

Create a .env file:

```env
GOOGLE_API_KEY=your_api_key_here
```

### Start the web app

```bash
python -m uvicorn app:app --reload
```

### Ingest a file

```bash
curl -X POST "http://localhost:8000/api/ingest" -F "file=@data/sample.pdf" -F "doc_id=sample_doc"
```

### Ask a question

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What were the key risks?","doc_id":"sample_doc"}'
```

---

## 10. Summary

FinRAG is a learning-first but increasingly production-minded RAG system. It started as a minimal document Q&A app and evolved into a more robust pipeline with multi-format ingestion, section-aware chunking, hybrid retrieval, benchmarking, and source-based evidence tracking.

The most important engineering lesson is simple:

- retrieval quality matters more than model complexity
- chunking and ranking are central to good RAG systems
- evidence grounding makes the output more dependable and explainable
- API and UI integration turn a technical prototype into a usable application

This project demonstrates the practical core of modern RAG and is strong enough to discuss confidently in a portfolio, interview, or technical walkthrough.
