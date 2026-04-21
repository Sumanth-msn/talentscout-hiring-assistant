"""
graph/edges.py — Graph definition for TalentScout.
NOTE: The graph is only used for the initial greeting invocation.
All subsequent user-input processing is handled directly in app.py
using the merge() helper to avoid LangGraph list-overwrite issues.
"""

from langgraph.graph import StateGraph, END
from graph.state import CandidateState
from graph.nodes import (
    guardrail_node,
    greeting_node,
    info_gather_node,
    tech_question_node,
    evaluator_node,
    closing_node,
)


def _route_guardrail(state: dict) -> str:
    if state.get("guardrail_triggered"):
        return "closing" if state.get("phase") == "closing" else END
    phase = state.get("phase", "greeting")
    if phase == "greeting":
        return "greeting"
    if phase == "info":
        return "info_gather"
    if phase == "tech_questions":
        return "tech_question"
    if phase == "closing":
        return "closing"
    return END


def _route_info(state: dict) -> str:
    return "tech_question" if state.get("phase") == "tech_questions" else END


def _route_eval(state: dict) -> str:
    return "closing" if state.get("phase") == "closing" else "tech_question"


def build_graph():
    """Build and compile the LangGraph state machine."""
    g = StateGraph(CandidateState)

    g.add_node("guardrail", guardrail_node)
    g.add_node("greeting", greeting_node)
    g.add_node("info_gather", info_gather_node)
    g.add_node("tech_question", tech_question_node)
    g.add_node("evaluator", evaluator_node)
    g.add_node("closing", closing_node)

    g.set_entry_point("guardrail")

    g.add_conditional_edges(
        "guardrail",
        _route_guardrail,
        {
            "greeting": "greeting",
            "info_gather": "info_gather",
            "tech_question": "tech_question",
            "closing": "closing",
            END: END,
        },
    )

    g.add_edge("greeting", END)

    g.add_conditional_edges(
        "info_gather",
        _route_info,
        {
            "tech_question": "tech_question",
            END: END,
        },
    )

    g.add_edge("tech_question", END)

    g.add_conditional_edges(
        "evaluator",
        _route_eval,
        {
            "tech_question": "tech_question",
            "closing": "closing",
        },
    )

    g.add_edge("closing", END)

    return g.compile()
