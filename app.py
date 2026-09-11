"""
Thin web layer over the existing pipeline. This file deliberately does
NOT reimplement extraction/chunking/embedding/retrieval — it just
exposes the functions already in ingest.py and query.py over HTTP so a
browser can call them. Keeping pipeline logic out of the web layer
means you can keep testing ingest.py/query.py directly from the
terminal too; nothing about them changed to accommodate the web UI.
"""
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ingest import ingest_document, list_documents, get_document_chunks
from query import ask

app = FastAPI(title="FinRAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local learning use; lock down before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


@app.post("/api/ingest")
async def api_ingest(file: UploadFile, doc_id: str = Form(...)):
    """Save the uploaded document, run it through the pipeline, and return the created chunks."""
    safe_name = file.filename or "uploaded_file"
    file_path = DATA_DIR / safe_name
    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = ingest_document(str(file_path), doc_id)
    return {"doc_id": doc_id, "chunk_count": len(chunks), "chunks": chunks}


@app.get("/api/documents")
async def api_list_documents():
    return {"documents": list_documents()}


@app.get("/api/documents/{doc_id}/chunks")
async def api_get_chunks(doc_id: str):
    return {"doc_id": doc_id, "chunks": get_document_chunks(doc_id)}


class QuestionRequest(BaseModel):
    question: str
    doc_id: str | None = None


@app.post("/api/query")
async def api_query(req: QuestionRequest):
    return ask(req.question, doc_id=req.doc_id)


# Serve the frontend itself. Because it's served from the same FastAPI
# app (same origin, same port), the frontend's fetch() calls to /api/...
# don't need any special cross-origin handling.
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")