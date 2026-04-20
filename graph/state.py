from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages


class TechScore(TypedDict):
    technology: str
    score: float  # 0.0 - 10.0
    difficulty_reached: str  # "Easy" | "Medium" | "Hard"


class CandidateState(TypedDict):
    # Conversation
    messages: Annotated[list, add_messages]

    # Candidate Info (collected sequentially)
    full_name: Optional[str]
    email: Optional[str]  # stored redacted in LLM context
    phone: Optional[str]  # stored redacted in LLM context
    years_experience: Optional[int]
    desired_position: Optional[str]
    current_location: Optional[str]
    tech_stack: List[str]  # ["Python", "Django", "PostgreSQL"]

    # Interview State
    current_tech_index: int  # which tech we're questioning
    current_question_index: int  # 0-4 questions per tech
    current_difficulty: str  # "Easy" | "Medium" | "Hard"
    questions_asked: List[str]
    answers_given: List[str]
    tech_scores: List[TechScore]

    # Sentiment
    sentiment_history: List[str]  # ["positive", "neutral", "negative"]

    # Control Flow
    phase: str  # "greeting"|"info"|"tech_questions"|"closing"|"ended"
    guardrail_triggered: bool
    session_id: str
