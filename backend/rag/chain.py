import os
import re
from typing import Dict, List

from dotenv import load_dotenv

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:  # pragma: no cover - compatibility fallback
    from langchain.schema import HumanMessage, SystemMessage

from rag.retriever import get_relevant_chunks_filtered

load_dotenv()

SYSTEM_PROMPT = """You are DocuMind, an expert technical documentation assistant.
Your job is to answer questions using ONLY the provided documentation context.

Rules:
- Answer ONLY from the given context. Do not hallucinate or use outside knowledge.
- If the answer is not in the context, say: "I couldn't find this in the uploaded documents. Try uploading more relevant files."
- Be concise but complete. Use bullet points or numbered steps when helpful.
- Start with a short direct answer and keep the structure clear.
- Always include a final "Sources:" line with the document names.
- If asked about code, format it in markdown code blocks.
"""

USER_PROMPT_TEMPLATE = """--- DOCUMENTATION CONTEXT ---
{context}
--- END CONTEXT ---

Conversation history:
{history}

Question: {question}

Answer (cite the source document at the end):"""


def format_context(results) -> tuple[str, list[str]]:
    """Build a readable context string and list of unique source files."""
    parts, sources = [], []
    for doc, score in results:
        src = os.path.basename(doc.metadata.get("source", "Unknown"))
        sources.append(src)
        parts.append(f"[From: {src}]\n{doc.page_content.strip()}")
    return "\n\n" + ("-" * 40) + "\n\n".join(parts), list(dict.fromkeys(sources))


def format_history(history: List[Dict]) -> str:
    """Format conversation history for the prompt."""
    if not history:
        return "No prior conversation."
    lines = []
    for msg in history[-6:]:  # last 3 turns
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def extract_answer_and_sources(answer: str) -> tuple[str, list[str]]:
    """Parse the raw model output into a clean answer and optional sources."""
    if not answer or not str(answer).strip():
        return (
            "I couldn't find a clear answer in the uploaded documents.",
            [],
        )

    text = str(answer).strip()
    text = re.sub(
        r"(?i)^\s*(?:sure|certainly|of course)[!,.]?\s*",
        "",
        text,
    )
    text = re.sub(
        r"(?i)^\s*(?:here(?:'s)?\s+)?(?:is\s+)?(?:the\s+)?answer\s*[:\-]?\s*",
        "",
        text,
    )
    text = re.sub(r"\r\n?", "\n", text)

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return (
            "I couldn't find a clear answer in the uploaded documents.",
            [],
        )

    parsed_sources = []
    last_line = lines[-1].strip()
    source_match = re.match(r"(?i)^sources?\s*[:\-]\s*(.*)$", last_line)
    if source_match:
        parsed_sources = [s.strip() for s in re.split(r"[;,]", source_match.group(1)) if s.strip()]
        lines = lines[:-1]

    first_line = lines[0]
    answer_prefix = re.match(r"(?i)^answer\s*[:\-]?\s*(.*)$", first_line)
    if answer_prefix:
        lines[0] = answer_prefix.group(1).strip()

    cleaned = "\n".join(lines).strip()
    if not cleaned:
        cleaned = "I couldn't find a clear answer in the uploaded documents."

    return cleaned, parsed_sources


def normalize_answer_text(answer: str) -> str:
    """Normalize LLM output into a readable answer string."""
    cleaned, _ = extract_answer_and_sources(answer)
    return cleaned


def get_llm():
    backend = os.getenv("LLM_BACKEND", "").strip().lower()
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if backend == "ollama" or use_ollama:
        from langchain_community.llms import Ollama
        model = os.getenv("OLLAMA_MODEL", "mistral")
        return Ollama(model=model), "ollama"

    if backend == "groq" or os.getenv("GROQ_API_KEY", ""):
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            raise ValueError(
                "LLM_BACKEND=groq requires GROQ_API_KEY to be set in .env."
            )
        from langchain_groq import ChatGroq
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        return ChatGroq(
            model=model,
            groq_api_key=groq_key,
        ), "groq"

    if backend == "openai" or os.getenv("OPENAI_API_KEY", ""):
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "your_openai_key_here":
            raise ValueError(
                "No API key found. Set OPENAI_API_KEY in .env for OpenAI usage."
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=api_key,
        ), "openai"

    raise ValueError(
        "No LLM backend configured. Set LLM_BACKEND=groq, LLM_BACKEND=openai, "
        "or USE_OLLAMA=true in .env."
    )


def ask_question(question: str, history: List[Dict] = None) -> Dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from FAISS
    2. Format context + history into prompt
    3. Call LLM
    4. Return answer, sources, and model used
    """
    if history is None:
        history = []

    results = get_relevant_chunks_filtered(question, k=5)
    if not results:
        return {
            "answer": "I couldn't find relevant information in the uploaded documents.",
            "sources": [],
            "model": "none",
            "chunks_used": 0,
        }

    context, sources = format_context(results)
    history_str = format_history(history)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        context=context,
        history=history_str,
        question=question,
    )

    llm, model_name = get_llm()

    if model_name == "ollama":
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        response_text = llm.invoke(full_prompt)
    else:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        response_text = getattr(response, "content", response)

    if hasattr(response_text, "content"):
        response_text = response_text.content
    elif isinstance(response_text, list):
        response_text = "\n".join(str(item) for item in response_text)
    else:
        response_text = str(response_text)

    answer_text, model_sources = extract_answer_and_sources(response_text)
    cleaned_answer = normalize_answer_text(answer_text)
    combined_sources = list(dict.fromkeys(sources + model_sources))

    return {
        "answer": cleaned_answer,
        "sources": combined_sources,
        "model": model_name,
        "chunks_used": len(results),
    }
