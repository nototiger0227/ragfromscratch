# FinRAG — Stage 1: Minimal RAG Pipeline

A bare‑bones but complete RAG pipeline for querying financial PDFs.

**Project Structure**

| File | Job |
|---|---|
| `config.py` | Central configuration for models, chunking, and DB paths |
| `extract.py` | PDF → text extraction |
| `chunk.py` | Text → list of overlapping character chunks |
| `ingest.py` | Chunk → embedding → Chroma vector DB (includes dedup on re‑ingest) |
| `query.py` | Question → retrieve chunks → ask LLM → answer |
| `app.py` | FastAPI wrapper exposing the above over HTTP |
| `frontend/` | Dependency‑free HTML/JS UI (served at `/` by FastAPI) |

## Setup

```bash
# Clone the repository and change into the project directory
cd e:/rag/ragfromscratch

# Create a virtual environment (Windows) and activate it
python -m venv .venv
.venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Copy the example env file and add your Gemini API key
cp .env.example .env   # Windows: copy .env.example .env
# Edit .env and set GOOGLE_API_KEY to a free Gemini API key from https://aistudio.google.com/apikey
```

## Run the original CLI tools

```bash
# 1. Index a PDF (place it in the `data/` folder first)
python ingest.py data/apple_2024.pdf apple_2024

# 2. Ask questions interactively
python query.py
# > What was total revenue?
# > What risks does the company mention around supply chain?
```

## Run the web UI

```bash
# Start the FastAPI server (auto‑reload enabled for development)
python -m uvicorn app:app --reload

# Open your browser at the following address:
http://127.0.0.1:8000
```

The UI provides:
- **Upload & Index**: Choose a PDF, give it a short `doc_id`, and ingest it.
- **Document List**: See all indexed documents and their chunk counts.
- **Chunk Catalog**: Browse each stored chunk as a card (showing index and character count).
- **Ask Panel**: Submit a question; the answer appears together with the exact chunks used.
- **Evidence Highlighting**: Chunks that contributed to the answer light up in amber, making retrieval transparent.

## Development Notes

- The FastAPI backend in `app.py` is a thin wrapper; it **does not duplicate** any ingestion or query logic.
- Re‑ingesting a document with the same `doc_id` automatically clears the old chunks to avoid duplication.
- All data (vector DB, uploaded PDFs) are stored locally under `chroma_db/` and `data/` respectively.
- For a production deployment you would lock down CORS origins and move the static UI to a CDN.

## Next Steps (Stage 2)

When you’re ready to improve the pipeline, consider:
- Structure‑aware chunking (split on headings/paragraphs).
- Hybrid retrieval (combine BM25 keyword search with vector search).
- Cross‑encoder reranking of retrieved chunks.

Feel free to ask me when you want to start building those extensions!
