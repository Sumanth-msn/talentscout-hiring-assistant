"""
graph/state.py — CandidateState definition.
NOTE: messages is a plain list here because we call nodes directly in app.py,
not via graph.invoke(). The merge() function in app.py handles appending.
"""

from typing import TypedDict, List, Optional


class TechScore(TypedDict):
    technology: str
    score: float  # 0.0 – 10.0
    difficulty_reached: str  # "Easy" | "Medium" | "Hard"


class CandidateState(TypedDict):
    # Conversation history
    messages: list

    # Candidate info — all Optional so we can detect "not yet filled"
    full_name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    years_experience: Optional[int]
    desired_position: Optional[str]
    current_location: Optional[str]
    tech_stack: Optional[List[str]]  # None until filled, then a list

    # Interview progress
    current_tech_index: int
    current_question_index: int
    current_difficulty: str  # "Easy" | "Medium" | "Hard"
    questions_asked: List[str]
    answers_given: List[str]
    tech_scores: List[TechScore]

    # Sentiment tracking
    sentiment_history: List[str]  # "positive" | "neutral" | "negative"

    # Control flow
    phase: str  # "greeting"|"info"|"tech_questions"|"closing"|"ended"
    guardrail_triggered: bool
    session_id: str
