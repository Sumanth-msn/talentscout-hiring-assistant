"""
FAISS-backed RAG engine for tech question retrieval.
Seeded from data/tech_qa_bank.json.
Falls back to LLM generation for unlisted technologies.

Uses st.cache_resource via a factory function so the index
is built ONCE per Streamlit server process (performance optimization).
"""

import json
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)
BANK_PATH = Path("data/tech_qa_bank.json")


def build_rag_engine() -> "TechQuestionRAG":
    """Factory — call this inside st.cache_resource."""
    return TechQuestionRAG()


class TechQuestionRAG:
    def __init__(self):
        with open(BANK_PATH) as f:
            self.bank = json.load(f)

        docs = [
            Document(
                page_content=item["question"],
                metadata={
                    "tech": item["tech"].lower(),
                    "difficulty": item["difficulty"],
                    "concept": item["concept"],
                },
            )
            for item in self.bank
        ]

        self.vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
        self._tech_set = {item["tech"].lower() for item in self.bank}
        logger.info(f"RAG engine ready. {len(docs)} questions indexed.")

    def get_questions(self, tech: str, difficulty: str, k: int = 3) -> list[str]:
        """
        Retrieve k questions for a given tech and difficulty.
        Falls back to semantic search without difficulty filter if insufficient results.
        """
        tech = tech.lower()

        # Attempt 1: exact tech + difficulty match
        results = self.vectorstore.similarity_search(
            query=f"{tech} {difficulty} interview question",
            k=k * 2,  # over-fetch then filter
        )

        filtered = [
            r.page_content
            for r in results
            if r.metadata.get("tech") == tech
            and r.metadata.get("difficulty") == difficulty
        ]

        if len(filtered) >= k:
            return filtered[:k]

        # Attempt 2: same tech any difficulty
        fallback = [r.page_content for r in results if r.metadata.get("tech") == tech]

        if fallback:
            return (filtered + fallback)[:k]

        # Attempt 3: semantic search (handles niche/unlisted stacks)
        semantic = self.vectorstore.similarity_search(
            query=f"{tech} programming interview",
            k=k,
        )
        return [r.page_content for r in semantic]

    def is_known_tech(self, tech: str) -> bool:
        return tech.lower() in self._tech_set
