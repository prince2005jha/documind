import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.ingest import (
    ingest_pipeline,
    vectorstore_exists,
    DOCS_PATH,
    VECTORSTORE_PATH,
)
from rag.chain import ask_question

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocuMind API",
    description="RAG-powered developer documentation Q&A backend",
    version="1.0.0",
)

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:5176",  # Vite fallback dev server
    "http://localhost:3000",  # CRA / other
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5176",
]

cors_origins = os.getenv("CORS_ORIGINS", "")
allowed_origins = [
    origin.strip() for origin in cors_origins.split(",") if origin.strip()
] if cors_origins else DEFAULT_CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(DOCS_PATH, exist_ok=True)


# ── Pydantic Models ──────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class AskRequest(BaseModel):
    question: str
    history: Optional[List[Message]] = []


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    model: str
    chunks_used: int


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — returns index status."""
    files = _list_docs()
    return {
        "status": "ok",
        "index_ready": vectorstore_exists(),
        "document_count": len(files),
    }


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload one or more PDF / HTML / Markdown files."""
    allowed = {".pdf", ".html", ".htm", ".md", ".markdown"}
    saved = []
    skipped = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed:
            skipped.append(file.filename)
            continue
        dest = os.path.join(DOCS_PATH, file.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(file.filename)

    if not saved:
        raise HTTPException(
            status_code=400,
            detail=f"No valid files uploaded. Accepted: PDF, HTML, Markdown. Skipped: {skipped}",
        )
    return {"uploaded": saved, "skipped": skipped}


@app.post("/ingest")
def ingest():
    """Build (or rebuild) the FAISS vector index from uploaded docs."""
    doc_files = _list_docs()
    if not doc_files:
        raise HTTPException(
            status_code=400,
            detail="No documents found in data/docs/. Upload files first.",
        )
    try:
        chunks = ingest_pipeline()
        return {
            "status": "success",
            "chunks_created": chunks,
            "files_indexed": len(doc_files),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Ask a question. Returns answer, sources, and metadata."""
    if not vectorstore_exists():
        raise HTTPException(
            status_code=400,
            detail="Vector index not built yet. Upload documents and click 'Build Index' first.",
        )
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        history = [{"role": m.role, "content": m.content} for m in (req.history or [])]
        result = ask_question(req.question, history=history)
        return result
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


@app.get("/documents")
def list_documents():
    """List all uploaded documents and index status."""
    files = _list_docs()
    return {
        "files": files,
        "index_ready": vectorstore_exists(),
        "count": len(files),
    }


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Delete a specific uploaded document."""
    filepath = os.path.join(DOCS_PATH, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    os.remove(filepath)
    return {"deleted": filename, "message": "File removed. Rebuild the index to reflect changes."}


@app.delete("/index")
def clear_index():
    """Delete the FAISS vector index (forces re-ingestion)."""
    removed = []
    if os.path.exists(VECTORSTORE_PATH):
        if os.path.isdir(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)
        else:
            os.remove(VECTORSTORE_PATH)
        removed.append(VECTORSTORE_PATH)

    for ext in [".faiss", ".pkl"]:
        p = VECTORSTORE_PATH + ext
        if os.path.exists(p):
            os.remove(p)
            removed.append(p)
    return {"status": "index_cleared", "removed": removed}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _list_docs() -> List[str]:
    allowed = {".pdf", ".html", ".htm", ".md", ".markdown"}
    if not os.path.exists(DOCS_PATH):
        return []
    return [
        f for f in os.listdir(DOCS_PATH)
        if os.path.splitext(f)[1].lower() in allowed
    ]
