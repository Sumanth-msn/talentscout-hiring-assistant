"""
services/qa_retriever.py — Mini-RAG using sentence-transformers + cosine similarity.
Retrieves reference questions from tech_qa_bank.json to ground LLM question generation.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once at import time (tiny, fast, works offline)
_model = SentenceTransformer("all-MiniLM-L6-v2")


def _load_qa_bank() -> list:
    path = Path(__file__).parent.parent / "data" / "tech_qa_bank.json"
    with open(path) as f:
        return json.load(f)


_QA_BANK = _load_qa_bank()
_QUESTIONS = [q["question"] for q in _QA_BANK]
_EMBEDDINGS = _model.encode(_QUESTIONS)


def retrieve_reference_questions(
    technology: str,
    difficulty: str,
    n: int = 2,
) -> list:
    """
    Return up to n reference questions relevant to the given technology and difficulty.
    The LLM uses these as inspiration only — not verbatim output.
    Falls back to similarity-only matching if no tag/difficulty filters match.
    """
    query = f"{technology} {difficulty} programming interview question"
    query_emb = _model.encode([query])
    scores = cosine_similarity(query_emb, _EMBEDDINGS)[0]

    # Try filtered match first (difficulty + technology tag)
    filtered = [
        (i, scores[i])
        for i, q in enumerate(_QA_BANK)
        if q.get("difficulty", "").lower() == difficulty.lower()
        and technology.lower() in [t.lower() for t in q.get("tags", [])]
    ]

    # Fall back to similarity-only if filter yields nothing
    if not filtered:
        filtered = [(i, scores[i]) for i in range(len(_QA_BANK))]

    filtered.sort(key=lambda x: x[1], reverse=True)
    return [_QA_BANK[i]["question"] for i, _ in filtered[:n]]
