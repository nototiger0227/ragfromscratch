# FinRAG

Document Q&A for PDFs, DOCX files, and TXT documents using a grounded RAG pipeline.

FinRAG is a lightweight retrieval-augmented generation system for asking natural-language questions over business and financial documents. It ingests mixed document types, chunks text intelligently, retrieves the most relevant evidence, and answers with source-backed responses grounded in the original content.

This project combines a local vector database, hybrid retrieval, and a FastAPI-backed UI into a practical end-to-end document Q&A workflow that is easy to understand, extend, and explain in interviews.

## Highlights

- Multi-format ingestion: PDF, DOCX, and TXT
- Section-aware chunking with overlap for better retrieval quality
- Hybrid semantic + lexical retrieval
- Document-level metadata and source traceability
- Immediate financial insight extraction for revenue, profit, debt, risks, and outlook
- Batch ingestion for up to five documents at once
- FastAPI backend with a simple browser UI
- Evidence-grounded answer generation using Gemini

## Repo Blurb

Built as a portfolio-ready RAG project, FinRAG demonstrates the full document intelligence pipeline: ingest raw files, normalize text, split content into meaningful chunks, embed and index them in ChromaDB, retrieve the most relevant passages for a user question, and generate a grounded answer with supporting evidence. It is designed to showcase real AI engineering skills in retrieval, vector search, LLM grounding, API design, and user-facing evaluation.

## Architecture

```text
User Upload / Query
      |
      v
  FastAPI App
      |
      +--> Extract text (PDF / DOCX / TXT)
      |
      +--> Chunk text intelligently
      |
      +--> Embed chunks
      |
      +--> Store in ChromaDB
      |
      +--> Hybrid retrieval (semantic + lexical)
      |
      +--> Gemini LLM answers with source evidence
      |
      v
  Response + cited passages
```

## Project Goals

- Build a clean end-to-end retrieval pipeline for unstructured documents
- Separate extraction, chunking, retrieval, and generation into clear modules
- Support mixed-source document ingestion with consistent processing
- Show evidence-grounded question answering instead of raw model memory
- Create a project that is easy to explain in interviews and portfolio reviews

## Core Layers

1. Ingestion
  - extract text from PDF, DOCX, and TXT files
  - normalize content into a consistent text representation
  - chunk it into semantically meaningful units

2. Retrieval
  - embed the user question
  - search ChromaDB for the closest semantic matches
  - rerank using lexical overlap / BM25-style scoring

3. Generation
  - pass relevant chunks into the LLM prompt
  - answer only from retrieved evidence
  - include source metadata for transparency and trust

4. Serving Layer
  - FastAPI endpoints for ingest, query, and document listing
  - browser-based frontend for live interaction

---

## Repository Structure

```text
.
├── app.py                 # FastAPI backend and static frontend hosting
├── benchmark.py           # Retrieval timing and volume benchmark helper
├── chunk.py               # Section-aware chunking logic
├── config.py              # Central configuration and environment settings
├── documentation.md       # Detailed technical documentation
├── extract.py             # PDF / DOCX / TXT extraction logic
├── frontend/
│   └── index.html         # Browser UI for upload, document catalog, and Q&A
├── ingest.py              # Embedding + Chroma storage pipeline
├── query.py               # Hybrid retrieval + answer generation
├── requirements.txt       # Python dependencies
├── tests/
│   ├── conftest.py
│   └── test_pipeline.py   # Regression tests for extraction and chunking
├── data/                  # User-uploaded files and local document storage
├── chroma_db/             # Local ChromaDB persistence directory
├── README.md              # Product-style project documentation
└── .env                   # Local environment variable file (not committed)
```

---

## Tech Stack

- Python 3.10+
- FastAPI
- ChromaDB
- Google Gemini API
- PyMuPDF for PDF parsing
- python-docx for DOCX parsing
- HTML + JavaScript frontend

---

## Setup

### 1. Clone and create a virtual environment

```bash
cd /workspaces/ragfromscratch
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

> The project reads this key from the environment and uses it for both embedding and answer generation.

---

## How the RAG Pipeline Works

### Step 1 — Extraction

The system accepts a document and converts it into raw text. This is handled by [extract.py](extract.py).

Supported file types:

- PDF
- DOCX
- TXT

Each file is normalized to a text representation before it enters the retrieval pipeline.

### Step 2 — Chunking

The text is split into smaller chunks using [chunk.py](chunk.py). The chunking strategy is section-aware and overlap-aware.

This means the system:

- respects headings and paragraphs when possible
- avoids splitting important facts across chunk boundaries
- keeps chunk sizes manageable for embedding and retrieval

### Step 3 — Embedding

Each chunk is converted into a vector using a Gemini embedding model in [ingest.py](ingest.py).

This transforms the text into a numeric representation capturing conceptual similarity.

### Step 4 — Storage

The vectors are saved in ChromaDB with metadata such as:

- document id
- chunk index
- source type
- file name
- character length

This allows metadata-aware retrieval and debugging.

### Step 5 — Retrieval

At query time, [query.py](query.py) embeds the user question and retrieves the closest chunks from ChromaDB.

The retrieval logic is hybrid:

- dense semantic retrieval using embeddings
- lexical reranking using keyword overlap / BM25-inspired scoring

This improves retrieval when the user asks about exact terms like ticker symbols, product names, or financial metrics.

### Step 6 — Generation

The retrieved chunks are inserted into a prompt and sent to the Gemini chat model. The prompt instructs the model to respond only using the supplied context.

The response includes:

- final answer
- source chunks used to support the answer

This makes the model output easier to inspect and verifies the answer is grounded in the document content.

---

## API Documentation

The backend is implemented in [app.py](app.py). It exposes the document pipeline over HTTP.

### 1. Ingest a document

```http
POST /api/ingest
```

#### Request

multipart/form-data

- file: uploaded document
- doc_id: short identifier for the uploaded document

#### Example

```bash
curl -X POST "http://localhost:8000/api/ingest" \
  -F "file=@data/sample.pdf" \
  -F "doc_id=sample_doc"
```

#### Example response

```json
{
  "doc_id": "sample_doc",
  "chunk_count": 18,
  "chunks": [
    {
      "id": "sample_doc_0",
      "chunk_index": 0,
      "text": "...",
      "char_count": 820
    }
  ]
}
```

### 2. Ingest up to five documents

```http
POST /api/ingest-batch
```

The endpoint accepts repeated `files` form fields. Optional repeated `doc_ids` fields can override filename-based document IDs. Each indexed document returns chunks and an immediate local financial snapshot grouped into `revenue`, `profit`, `debt`, `risks`, and `outlook`.

```bash
curl -X POST "http://localhost:8000/api/ingest-batch" \
  -F "files=@data/report_2024.pdf" \
  -F "files=@data/report_2023.pdf"
```

### 3. List all indexed documents

```http
GET /api/documents
```

#### Example response

```json
{
  "documents": [
    {"doc_id": "sample_doc", "chunk_count": 18}
  ]
}
```

### 4. Get chunks for a document

```http
GET /api/documents/{doc_id}/chunks
```

#### Example

```bash
curl "http://localhost:8000/api/documents/sample_doc/chunks"
```

### 5. Ask a question

```http
POST /api/query
```

#### Request body

```json
{
  "question": "What were the main risks mentioned?",
  "doc_id": "sample_doc"
}
```

#### Example

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What were the main risks mentioned?","doc_id":"sample_doc"}'
```

#### Example response

```json
{
  "answer": "The document highlights supply chain disruption and inflation pressure as the primary risks.",
  "sources": [
    {
      "chunk_id": "sample_doc_6",
      "doc_id": "sample_doc",
      "chunk_index": 6,
      "source_type": "pdf",
      "text": "Supply chain disruption and inflation pressure remain significant risks..."
    }
  ]
}
```

---

## Frontend Behavior

The frontend served from [frontend/index.html](frontend/index.html) includes:

- multi-file drag-and-drop upload for up to five documents
- document ID assignment
- document listing
- chunk catalog view
- post-ingestion financial insight cards for revenue, profit, debt, risks, and outlook
- question input
- answer panel
- clickable source evidence chips

This makes the project more compelling in demos because users can visually inspect the exact retrieved evidence used to answer a question.

---

## Retrieval and Ranking Theory

### Dense retrieval

Dense retrieval measures similarity in embedding space. It is excellent for semantic matching when the user asks using different wording than the document.

Example:

- Document says: “Revenue declined in the second quarter.”
- User asks: “How did sales perform in Q2?”

Embedding similarity helps align these concepts even when the exact wording differs.

### Lexical retrieval

Lexical retrieval checks overlap in actual tokens. It works very well for exact names, numbers, metrics, and technical terms.

Example:

- Document mentions “Net income margin was 18%.”
- User asks: “What was the net income margin?”

A lexical method is strong here because the terms overlap directly.

### Hybrid retrieval

Hybrid retrieval combines both signals to improve robustness. This is important because an enterprise document system often contains both conceptual and exact-match queries.

---

## Benchmarking and Evaluation

The project includes a simple benchmark utility in [benchmark.py](benchmark.py). It can be used to measure:

- average retrieval latency
- total retrieved chunks
- query-to-query performance patterns

This is valuable because RAG systems are often evaluated on both output quality and retrieval efficiency.

Example:

```bash
python benchmark.py --questions "What was revenue?" "What are the risks?" "What is the strategy?"
```

---

## Production Considerations

The project is already a strong learning and portfolio build, but a production system would add:

- authentication and authorization
- user session management
- logging and observability
- background ingestion jobs
- rate limiting
- secure environment handling
- better deployment packaging
- monitoring for retriever quality drift
- persistent storage for document metadata

---

## Key Implementation Benefits

This project demonstrates several useful engineering capabilities:

- document ingestion from mixed source types
- modular, testable retrieval architecture
- vector-based semantic search
- hybrid ranking logic
- grounded LLM responses using only retrieved context
- API-driven application deployment
- frontend evidence display and source tracing

These are all highly relevant to modern AI product work and are strong talking points in interviews.

---

## Summary

FinRAG is a practical, understandable, and extendable RAG system that begins with a basic document Q&A pipeline and evolves into a more robust architecture with hybrid retrieval, metadata-aware indexing, and evidence-based generation.

It is a strong example of how AI systems are built in practice: not just by calling an LLM, but by combining structured preprocessing, retrieval, LLM grounding, and a usable service interface.

This project is now positioned as a real portfolio-ready RAG application with a solid technical narrative and a clear explanation of the design decisions behind it.
