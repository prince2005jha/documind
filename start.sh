#!/bin/bash
# ── DocuMind Startup Script ─────────────────────────────────
# Run from the project root: bash start.sh

echo ""
echo "🧠 DocuMind — RAG Documentation Assistant"
echo "────────────────────────────────────────────"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Install it from https://python.org"
    exit 1
fi

# Check Node
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Install it from https://nodejs.org"
    exit 1
fi

# Backend
echo ""
echo "▶ Starting FastAPI backend on http://localhost:8000 ..."
cd backend

if [ ! -d "venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate 2>/dev/null || venv\Scripts\activate 2>/dev/null

if ! python -c "import fastapi" &> /dev/null; then
    echo "  Installing Python dependencies (first run only)..."
    pip install -r requirements.txt -q
fi

uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Frontend
echo "▶ Starting React frontend on http://localhost:5173 ..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "  Installing Node dependencies (first run only)..."
    npm install -q
fi

npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ DocuMind is running!"
echo "   Frontend → http://localhost:5173"
echo "   Backend  → http://localhost:8000"
echo "   API docs → http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers."
echo ""

# Wait and cleanup on exit
trap "echo ''; echo 'Stopping...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
