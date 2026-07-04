from rag.ingest import load_vectorstore


def get_relevant_chunks(query: str, k: int = 5):
    """
    Semantic search: returns top-k (Document, score) tuples.
    Lower score = more similar (L2 distance).
    """
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


def get_relevant_chunks_filtered(query: str, k: int = 5, score_threshold: float = 1.5):
    """
    Like get_relevant_chunks but filters out poor matches above the threshold.
    """
    results = get_relevant_chunks(query, k=k)
    # Keep only results with distance below threshold (closer = more relevant)
    filtered = [(doc, score) for doc, score in results if score <= score_threshold]
    return filtered if filtered else results[:2]  # always return at least 2
