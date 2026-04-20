import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

_model = SentenceTransformer("all-MiniLM-L6-v2")  # tiny, fast, offline-capable


def _load_qa_bank() -> list[dict]:
    path = Path(__file__).parent.parent / "data" / "tech_qa_bank.json"
    with open(path) as f:
        return json.load(f)


_QA_BANK = _load_qa_bank()
_EMBEDDINGS = _model.encode([q["question"] for q in _QA_BANK])


def retrieve_reference_questions(
    technology: str, difficulty: str, n: int = 3
) -> list[str]:
    """
    Find the most relevant reference questions for a technology + difficulty.
    The LLM uses these as inspiration, not verbatim output.
    """
    query = f"{technology} {difficulty} question"
    query_emb = _model.encode([query])
    scores = cosine_similarity(query_emb, _EMBEDDINGS)[0]

    # Filter by difficulty tag, then sort by similarity
    filtered = [
        (i, scores[i])
        for i, q in enumerate(_QA_BANK)
        if q.get("difficulty", "").lower() == difficulty.lower()
        and technology.lower() in q.get("tags", [])
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)

    return [_QA_BANK[i]["question"] for i, _ in filtered[:n]]
