"""
services/evaluator.py — LLM-as-Judge scoring with Dynamic Difficulty Adjustment.
"""

import os
import json

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


def get_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def score_answer(question: str, answer: str, technology: str) -> dict:
    """
    Evaluate a candidate's answer using the LLM as judge.
    Returns a dict with score (0-10), feedback, and next_difficulty.
    Falls back to safe defaults if JSON parsing fails.
    """
    llm = get_llm()

    prompt = (
        f"You are a strict technical interviewer evaluating a candidate's answer.\n\n"
        f"Technology: {technology}\n"
        f"Question: {question}\n"
        f"Candidate's Answer: {answer}\n\n"
        f"Respond ONLY with valid JSON (no markdown, no backticks):\n"
        f"{{\n"
        f'  "score": <integer 0-10>,\n'
        f'  "feedback": "<one sentence: what they got right or wrong>",\n'
        f'  "next_difficulty": "<Easy|Medium|Hard>"\n'
        f"}}\n\n"
        f"Scoring guide:\n"
        f"- 8-10: Complete and accurate → next: Hard\n"
        f"- 5-7:  Partially correct     → next: Medium\n"
        f"- 0-4:  Incorrect/incomplete  → next: Easy"
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        raw = response.content.strip()
        # Strip accidental markdown fences
        raw = raw.strip("`").replace("json\n", "").replace("json", "").strip()
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {
            "score": 5,
            "feedback": "Answer received.",
            "next_difficulty": "Medium",
        }
