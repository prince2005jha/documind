import os
import re
from pathlib import Path

try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        UnstructuredHTMLLoader,
        UnstructuredMarkdownLoader,
    )
except ImportError:  # pragma: no cover - optional dependency guard
    PyPDFLoader = None
    UnstructuredHTMLLoader = None
    UnstructuredMarkdownLoader = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - optional dependency guard
    RecursiveCharacterTextSplitter = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - optional dependency guard
    HuggingFaceEmbeddings = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency guard
    SentenceTransformer = None

try:
    from langchain_community.vectorstores import FAISS
except ImportError:  # pragma: no cover - optional dependency guard
    FAISS = None

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - optional dependency guard
    Document = None


class SentenceTransformerEmbeddingsWrapper:
    """Simple wrapper around `sentence_transformers.SentenceTransformer` that
    provides the `embed_documents` and `embed_query` methods expected by
    LangChain/FAISS."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        embs = self.model.encode(list(texts), show_progress_bar=False)
        return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]

    def embed_query(self, text):
        emb = self.model.encode([text], show_progress_bar=False)[0]
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def __call__(self, texts):
        """Allow the wrapper to be called like other LangChain embeddings.
        If `texts` is a string, return a single embedding; if it's an iterable,
        return a list of embeddings.
        """
        if isinstance(texts, str):
            return self.embed_query(texts)
        return self.embed_documents(texts)

BASE_DIR = Path(__file__).resolve().parents[1]
VECTORSTORE_PATH = str(BASE_DIR / "vectorstore" / "faiss_index")
DOCS_PATH = str(BASE_DIR / "data" / "docs")


def get_embeddings():
    """Load embeddings using the configured backend."""
    backend = os.getenv("EMBEDDING_BACKEND", "").strip().lower()
    openai_key = os.getenv("OPENAI_API_KEY", "")

    def _openai_embeddings():
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(openai_api_key=openai_key)

    if backend == "openai":
        if not openai_key or openai_key == "your_openai_key_here":
            raise RuntimeError(
                "EMBEDDING_BACKEND=openai requires OPENAI_API_KEY to be set in .env."
            )
        return _openai_embeddings()

    if backend == "local":
        try:
            return SentenceTransformerEmbeddingsWrapper("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"⚠️  Could not load SentenceTransformer embeddings: {e}")
        try:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device_map": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            print(f"⚠️  Could not load HuggingFaceEmbeddings: {e}")
        raise RuntimeError(
            "EMBEDDING_BACKEND=local requires sentence-transformers or "
            "langchain_huggingface to be installed."
        )

    if openai_key and openai_key != "your_openai_key_here":
        try:
            return _openai_embeddings()
        except Exception as e:
            print(f"⚠️  OpenAIEmbeddings import failed: {e}")

    try:
        return SentenceTransformerEmbeddingsWrapper("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"⚠️  Could not load SentenceTransformer embeddings: {e}")

    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device_map": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        print(f"⚠️  Could not load HuggingFaceEmbeddings: {e}")

    raise RuntimeError(
        "No embeddings backend available. Set OPENAI_API_KEY for OpenAI embeddings "
        "or install sentence-transformers / langchain_huggingface for local models."
    )


def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 60):
    """Split a long text block into smaller overlapping chunks."""
    if not text or not text.strip():
        return []

    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= chunk_size:
        return [normalized]

    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return [chunk.page_content if hasattr(chunk, "page_content") else str(chunk) for chunk in splitter.split_text(normalized)]

    parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    parts = [part.strip() for part in parts if part and part.strip()]
    if not parts:
        return []

    chunks = []
    current = ""
    for part in parts:
        if not current:
            current = part
            continue
        if len(current) + 1 + len(part) <= chunk_size:
            current = f"{current} {part}"
        else:
            chunks.append(current)
            if chunk_overlap > 0:
                current = f"{current[-chunk_overlap:]} {part}".strip()
            else:
                current = part

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk.strip()]


def load_documents(folder_path: str = DOCS_PATH):
    """Load PDF, HTML, and Markdown files from the given folder."""
    docs = []
    if not os.path.exists(folder_path):
        return docs
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        try:
            if filename.endswith(".pdf"):
                if PyPDFLoader is None:
                    raise RuntimeError("PyPDFLoader is not available. Install backend requirements first.")
                loader = PyPDFLoader(filepath)
            elif filename.endswith(".html") or filename.endswith(".htm"):
                if UnstructuredHTMLLoader is None:
                    raise RuntimeError("HTML loader is not available. Install backend requirements first.")
                loader = UnstructuredHTMLLoader(filepath)
            elif filename.endswith(".md") or filename.endswith(".markdown"):
                if UnstructuredMarkdownLoader is None:
                    raise RuntimeError("Markdown loader is not available. Install backend requirements first.")
                loader = UnstructuredMarkdownLoader(filepath)
            else:
                continue
            loaded = loader.load()
            # Tag each doc with the filename for citation
            for doc in loaded:
                doc.metadata["source"] = filename
            docs.extend(loaded)
        except Exception as e:
            print(f"⚠️  Could not load {filename}: {e}")
    print(f"✅ Loaded {len(docs)} document pages from {folder_path}")
    return docs


def chunk_documents(docs):
    """Split documents into smaller overlapping chunks for better retrieval."""
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "60"))

    chunks = []
    for doc in docs:
        text = getattr(doc, "page_content", "") or ""
        if not text.strip():
            continue
        for chunk_text in split_text_into_chunks(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            metadata = dict(getattr(doc, "metadata", {}) or {})
            if Document is not None:
                chunk = Document(page_content=chunk_text, metadata=metadata)
            else:
                chunk = type("SimpleDoc", (), {"page_content": chunk_text, "metadata": metadata})()
            chunks.append(chunk)

    print(f"✅ Created {len(chunks)} chunks (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
    return chunks


def build_vectorstore(chunks):
    """Embed all chunks and persist to FAISS on disk."""
    if FAISS is None:
        raise RuntimeError("FAISS is not available. Install backend requirements first.")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"✅ FAISS index saved → {VECTORSTORE_PATH}")
    return vectorstore


def load_vectorstore():
    """Load the persisted FAISS index from disk."""
    if FAISS is None:
        raise RuntimeError("FAISS is not available. Install backend requirements first.")
    embeddings = get_embeddings()
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def vectorstore_exists() -> bool:
    index_dir = Path(VECTORSTORE_PATH)
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


def ingest_pipeline(folder_path: str = DOCS_PATH) -> int:
    """Full ingestion pipeline: load → chunk → embed → store. Returns chunk count."""
    docs = load_documents(folder_path)
    if not docs:
        raise ValueError("No documents found to ingest.")
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    return len(chunks)
