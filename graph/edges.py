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


def route_after_guardrail(state: dict) -> str:
    if state.get("guardrail_triggered"):
        return "closing" if state.get("phase") == "closing" else "end_guardrail"
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


def route_after_info(state: dict) -> str:
    if state.get("phase") == "tech_questions":
        return "tech_question"
    return "info_gather"  # still collecting info


def route_after_evaluator(state: dict) -> str:
    if state.get("phase") == "closing":
        return "closing"
    return "tech_question"  # more questions to ask


def build_graph():
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
        route_after_guardrail,
        {
            "greeting": "greeting",
            "info_gather": "info_gather",
            "tech_question": "tech_question",
            "closing": "closing",
            "end_guardrail": END,
        },
    )

    g.add_edge("greeting", END)  # waits for user input

    g.add_conditional_edges(
        "info_gather",
        route_after_info,
        {
            "tech_question": "tech_question",
            "info_gather": END,  # waits for next user input
        },
    )

    g.add_edge("tech_question", END)  # waits for candidate's answer

    g.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {
            "tech_question": "tech_question",
            "closing": "closing",
        },
    )

    g.add_edge("closing", END)

    return g.compile()
