"""
CandidateState: The single source of truth for the entire interview session.
All nodes read from and write to this TypedDict.
"""

from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class TechStackEntry(TypedDict):
    """Structured tech stack item with category classification."""

    name: str
    category: str  # "language" | "framework" | "database" | "tool" | "platform"


class ScoreEntry(TypedDict):
    """One evaluated answer with its score and rationale."""

    tech: str
    question: str
    answer: str
    score: int  # 1–5
    rationale: str
    difficulty: str  # easy | medium | hard
    sentiment: str  # positive | neutral | negative


class CandidateState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ── Candidate profile (filled progressively) ──────────────────────────────
    full_name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    current_location: Optional[str]
    years_of_experience: Optional[str]
    desired_positions: Optional[str]
    tech_stack: Optional[list]  # list of TechStackEntry dicts

    # ── Flow control ──────────────────────────────────────────────────────────
    current_phase: str  # greeting | info_gathering | tech_interview | ended
    exit_requested: bool  # set True when exit keyword detected

    # ── Tech interview tracking ───────────────────────────────────────────────
    questions_asked: list  # list of question strings
    candidate_answers: list  # list of answer strings
    answer_scores: list  # list of ScoreEntry dicts
    current_difficulty: str  # easy | medium | hard
    current_tech_index: int  # which tech in stack we're currently on
    questions_this_tech: int  # count of questions asked for current tech (target: 3–5)

    # ── Clarification handling ────────────────────────────────────────────────
    needs_clarification: bool  # True if candidate asked to rephrase

    # ── Session metadata ──────────────────────────────────────────────────────
    session_id: str
    language: str  # e.g. "English", "Hindi", "Tamil"
