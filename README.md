# 🧠 DocuMind — RAG Documentation Assistant

A full-stack AI app that lets you upload PDF, HTML, or Markdown docs and ask natural language questions about them. Powered by **RAG (Retrieval-Augmented Generation)** using FAISS + HuggingFace Embeddings + GPT-3.5-turbo.

---

## 🏗️ Architecture

```
Frontend (React + Vite)          Backend (FastAPI)
┌────────────────────┐           ┌──────────────────────────┐
│  Sidebar           │  HTTP     │  POST /upload            │
│  • Upload docs     │ ────────► │  POST /ingest            │
│  • Build index     │           │  POST /ask               │
│  • File list       │           │  GET  /documents         │
├────────────────────┤           ├──────────────────────────┤
│  ChatWindow        │           │  RAG Pipeline            │
│  • Message history │ ◄──────── │  1. Load + chunk docs    │
│  • Source badges   │  JSON     │  2. Embed (MiniLM-L6-v2) │
│  • Typing states   │           │  3. Store in FAISS       │
└────────────────────┘           │  4. Retrieve top-k       │
                                 │  5. GPT-3.5 answers      │
                                 └──────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Clone / unzip the project
```bash
cd documind
```

### 2. Backend setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
# Edit .env and replace "your_openai_key_here"
# Get a key at: https://platform.openai.com/api-keys
# To use Groq as the LLM, set:
# LLM_BACKEND=groq
# GROQ_API_KEY=your_groq_api_key_here
# To use Ollama instead, set:
# USE_OLLAMA=true
# OLLAMA_MODEL=mistral
# To use local embeddings instead of OpenAI, set:
# EMBEDDING_BACKEND=local
# To reduce index size / speed up ingest, set:
# CHUNK_SIZE=1000
# CHUNK_OVERLAP=80
# Start the server
uvicorn main:app --reload --port 8000
```

✅ API docs available at: http://localhost:8000/docs

### 3. Frontend setup (new terminal)
```bash
cd frontend
npm install
npm run dev
```

✅ App available at: http://localhost:5173

---

## 🆓 Free (No OpenAI Key) Option

Use **Ollama** to run a local LLM:

```bash
# Install Ollama: https://ollama.ai
ollama pull mistral

# In backend/.env, set:
USE_OLLAMA=true
OLLAMA_MODEL=mistral
```

---

## 📋 API Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/` | Health check + status |
| POST | `/upload` | Upload PDF/HTML/MD files |
| POST | `/ingest` | Build FAISS vector index |
| POST | `/ask` | Ask a question |
| GET | `/documents` | List uploaded files |
| DELETE | `/documents/{filename}` | Delete a file |
| DELETE | `/index` | Clear the vector index |

### Example API call:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does authentication work?"}'
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + Vite |
| Backend | FastAPI + Uvicorn |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (free) |
| Vector DB | FAISS (local, persisted to disk) |
| LLM | Groq llama-3.1-8b-instant / OpenAI GPT-3.5-turbo / Ollama Mistral |
| RAG Framework | LangChain |
| Doc Loaders | pypdf, unstructured |

---

## 📁 Project Structure

```
documind/
├── backend/
│   ├── main.py              ← FastAPI routes
│   ├── requirements.txt
│   ├── .env                 ← API keys (never commit this!)
│   ├── data/docs/           ← Uploaded documents stored here
│   ├── vectorstore/         ← FAISS index (auto-created)
│   └── rag/
│       ├── ingest.py        ← Load + chunk + embed docs
│       ├── retriever.py     ← Semantic search
│       └── chain.py         ← LLM + prompt + history
└── frontend/
    ├── src/
    │   ├── App.jsx          ← Root layout
    │   ├── api.js           ← Axios API calls
    │   ├── index.css        ← Global styles
    │   └── components/
    │       ├── Sidebar.jsx       ← Upload + index panel
    │       ├── ChatWindow.jsx    ← Chat interface
    │       └── MessageBubble.jsx ← Message renderer
    ├── package.json
    └── vite.config.js
```

---

## 💡 Interview Talking Points

- **RAG pipeline**: document ingestion → semantic chunking → FAISS vector search → LLM generation
- **Embedding model**: free HuggingFace `all-MiniLM-L6-v2`, no API cost
- **Source citation**: every answer includes the source document(s) it came from
- **Conversation history**: last 3 turns sent with each request for context-aware answers
- **Persistent index**: FAISS index saved to disk, no re-embedding on restart
- **Multi-format**: supports PDF, HTML, and Markdown ingestion
- **LLM agnostic**: swap GPT-3.5 for Ollama (free, local) with one env variable

---

## 🚀 Enhancement Ideas (for resume)

- [ ] Add streaming responses (token-by-token like ChatGPT)
- [ ] Add re-ranking with Cohere or cross-encoders
- [ ] Deploy backend to Railway/Render, frontend to Vercel
- [ ] Add Docker Compose for one-command startup
- [ ] Add user authentication with JWT
- [ ] Add conversation memory persistence (SQLite)
