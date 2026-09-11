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

from fastapi import File, FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ingest import ingest_document, list_documents, get_document_chunks, summarize_document
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


def _safe_document_path(file: UploadFile, index: int = 0) -> Path:
    """Create a collision-resistant local filename for an uploaded document."""
    original_name = Path(file.filename or f"uploaded_file_{index}").name
    return DATA_DIR / original_name


def _ingest_upload(file: UploadFile, doc_id: str, index: int = 0) -> dict:
    file_path = _safe_document_path(file, index)
    with file_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    insights = summarize_document(str(file_path))
    chunks = ingest_document(str(file_path), doc_id)
    return {"doc_id": doc_id, "chunk_count": len(chunks), "chunks": chunks, "insights": insights}


@app.post("/api/ingest")
async def api_ingest(file: UploadFile, doc_id: str = Form(...)):
    """Save the uploaded document, run it through the pipeline, and return the created chunks."""
    try:
        return _ingest_upload(file, doc_id)
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Document ingestion failed.",
                "error": str(exc),
            },
        )


@app.post("/api/ingest-batch")
async def api_ingest_batch(files: list[UploadFile] = File(...), doc_ids: list[str] | None = Form(None)):
    """Ingest up to five documents and return an insight summary for each one."""
    if not files or len(files) > 5:
        return JSONResponse(status_code=400, content={"detail": "Upload between one and five documents."})

    results = []
    try:
        supplied_ids = doc_ids or []
        for index, file in enumerate(files):
            doc_id = supplied_ids[index].strip() if index < len(supplied_ids) and supplied_ids[index].strip() else Path(file.filename or f"document_{index + 1}").stem
            results.append(_ingest_upload(file, doc_id, index))
        return {"documents": results, "document_count": len(results), "chunk_count": sum(item["chunk_count"] for item in results)}
    except Exception as exc:
        return JSONResponse(status_code=503, content={"detail": "Batch ingestion failed.", "error": str(exc)})


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
    try:
        return ask(req.question, doc_id=req.doc_id)
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Query failed.",
                "error": str(exc),
            },
        )


# Serve the frontend itself. Because it's served from the same FastAPI
# app (same origin, same port), the frontend's fetch() calls to /api/...
# don't need any special cross-origin handling.
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")