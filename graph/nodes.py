"""
graph/nodes.py — All LangGraph nodes for TalentScout.
Each node receives the full state dict and returns a partial delta dict.
"""

import os
import uuid
import json

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from services.pii_redactor import redact_pii, redact_state_for_llm
from services.evaluator import score_answer
from services.qa_retriever import retrieve_reference_questions
from services.sentiment import analyze_sentiment


def get_llm(temperature: float = 0.7) -> ChatGroq:
    """Return a configured Groq LLM instance."""
    return ChatGroq(
        model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ── Node 1: Guardrail ──────────────────────────────────────────────────────────

EXIT_KEYWORDS = {"bye", "exit", "quit", "stop", "goodbye"}
TOXIC_KEYWORDS = {"hate", "kill", "abuse"}


def guardrail_node(state: dict) -> dict:
    """
    Safety gate — runs before every user-input processing step.
    Detects exit intent and toxic/off-topic messages.
    Returns guardrail_triggered=True to halt normal flow if needed.
    """
    last_user_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content.lower()
            break

    if any(kw in last_user_msg for kw in EXIT_KEYWORDS):
        return {"guardrail_triggered": True, "phase": "closing"}

    if any(kw in last_user_msg for kw in TOXIC_KEYWORDS):
        return {
            "guardrail_triggered": True,
            "messages": [
                AIMessage(
                    content="I'm here to assist with the hiring process only. "
                    "Let's keep our conversation professional."
                )
            ],
        }

    return {"guardrail_triggered": False}


# ── Node 2: Greeting ───────────────────────────────────────────────────────────


def greeting_node(state: dict) -> dict:
    """
    Sends the welcome message and transitions phase to 'info'.
    Called once at session start.
    """
    msg = (
        "Welcome to **TalentScout** — your AI-powered hiring assistant!\n\n"
        "I'll guide you through a brief screening process:\n"
        "1. I'll collect some basic information about you\n"
        "2. You'll share your tech stack\n"
        "3. I'll ask technical questions to assess your proficiency\n\n"
        "Type **'bye'** at any time to end the session.\n\n"
        "Let's begin — what is your **full name**?"
    )
    return {
        "phase": "info",
        "session_id": str(uuid.uuid4()),
        "messages": [AIMessage(content=msg)],
    }


# ── Node 3: Info Gathering ─────────────────────────────────────────────────────

# Ordered list of fields to collect
INFO_FIELDS = [
    "full_name",
    "email",
    "phone",
    "years_experience",
    "desired_position",
    "current_location",
    "tech_stack",
]

# Prompt to show AFTER each field is collected (None = last field)
INFO_PROMPTS = {
    "full_name": "What is your **email address**?",
    "email": "What is your **phone number**?",
    "phone": "How many **years of experience** do you have in tech?",
    "years_experience": "What **position(s)** are you applying for?",
    "desired_position": "What is your **current location** (city/country)?",
    "current_location": (
        "Almost there! Please list your **tech stack** — programming languages, "
        "frameworks, databases, and tools you are proficient in.\n"
        "(e.g. Python, Django, PostgreSQL, Docker)"
    ),
    "tech_stack": None,  # last field — triggers transition to tech questions
}


def info_gather_node(state: dict) -> dict:
    """
    Collects candidate info one field at a time.
    Uses the LLM to parse natural-language answers into structured values.
    When all fields are filled, transitions phase to 'tech_questions'.
    """
    llm = get_llm(temperature=0.1)

    # Get the last user message
    last_user_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    # Redact PII before sending to LLM
    safe_msg = redact_pii(last_user_msg)

    # Find which field still needs to be filled
    # IMPORTANT: tech_stack starts as None (not []), so this check is reliable
    current_field = None
    for field in INFO_FIELDS:
        if state.get(field) is None:
            current_field = field
            break

    # All fields filled — transition
    if current_field is None:
        return {"phase": "tech_questions"}

    # Ask LLM to extract the value from the user's message
    parse_prompt = (
        f"Extract the value for field '{current_field}' from this message.\n"
        f'Message: "{safe_msg}"\n\n'
        f"Rules:\n"
        f'- For \'tech_stack\': return a JSON array like ["Python", "Django"]\n'
        f"- For 'years_experience': return only an integer\n"
        f"- For everything else: return the cleaned string value\n"
        f"Respond ONLY with the raw value. No explanation, no markdown."
    )
    parsed = llm.invoke([HumanMessage(content=parse_prompt)]).content.strip()

    # Parse and store the value
    update: dict = {}

    if current_field == "tech_stack":
        try:
            tech_list = json.loads(parsed)
            update["tech_stack"] = (
                tech_list if isinstance(tech_list, list) else [parsed]
            )
        except (json.JSONDecodeError, ValueError):
            update["tech_stack"] = [t.strip() for t in parsed.split(",") if t.strip()]

    elif current_field == "years_experience":
        digits = "".join(filter(str.isdigit, parsed))
        update["years_experience"] = int(digits) if digits else 0

    elif current_field in ("email", "phone"):
        # Store original value — PII is redacted only when sent to LLM
        update[current_field] = last_user_msg.strip()

    else:
        update[current_field] = parsed

    # Send the next prompt or transition
    next_prompt = INFO_PROMPTS.get(current_field)
    if next_prompt:
        update["messages"] = [AIMessage(content=next_prompt)]
    else:
        # tech_stack just collected → switch phase
        # tech_question_node will generate the first question immediately
        update["phase"] = "tech_questions"
        update["current_tech_index"] = 0
        update["current_question_index"] = 0
        update["current_difficulty"] = "Easy"
        # Do NOT reset questions_asked, answers_given, sentiment_history, tech_scores here
        # They start empty from session init and must accumulate during the interview

    return update


# ── Node 4: Technical Question ─────────────────────────────────────────────────


def tech_question_node(state: dict) -> dict:
    """
    Generates one technical question for the current technology and difficulty.
    Uses mini-RAG (qa_retriever) to ground the question in real examples.
    """
    llm = get_llm(temperature=0.8)

    tech_stack = state.get("tech_stack") or []
    idx = state.get("current_tech_index", 0)
    difficulty = state.get("current_difficulty", "Easy")

    # All technologies exhausted — close the session
    if idx >= len(tech_stack):
        return {"phase": "closing"}

    technology = tech_stack[idx]

    # Retrieve reference questions from the QA bank (mini-RAG)
    references = retrieve_reference_questions(technology, difficulty, n=2)
    ref_text = (
        "\n".join(f"- {q}" for q in references)
        if references
        else "No references available."
    )

    safe_state = redact_state_for_llm(state)

    prompt = (
        f"You are a technical interviewer at TalentScout.\n"
        f"Generate ONE {difficulty}-level interview question about {technology}.\n\n"
        f"Candidate profile:\n"
        f"- Experience: {safe_state.get('years_experience')} years\n"
        f"- Desired role: {safe_state.get('desired_position')}\n\n"
        f"Reference questions (for inspiration only — do NOT copy verbatim):\n"
        f"{ref_text}\n\n"
        f"Requirements:\n"
        f"- One clear question only\n"
        f"- Appropriate for {difficulty} level\n"
        f"- Tests practical {technology} knowledge\n"
        f"- Output only the question, no preamble"
    )

    question = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    existing_questions = state.get("questions_asked") or []
    return {
        "messages": [AIMessage(content=f"**[{difficulty}]** {question}")],
        "questions_asked": existing_questions + [question],
    }


# ── Node 5: Evaluator ──────────────────────────────────────────────────────────

MAX_QUESTIONS_PER_TECH = 3


def evaluator_node(state: dict) -> dict:
    """
    Scores the candidate's last answer using LLM-as-Judge.
    Updates difficulty (Dynamic Difficulty Adjustment).
    Accumulates per-question scores and averages them when a tech is complete.
    """
    tech_stack = state.get("tech_stack") or []
    idx = state.get("current_tech_index", 0)
    q_idx = state.get("current_question_index", 0)
    technology = tech_stack[idx] if idx < len(tech_stack) else "General"

    # Get the candidate's last answer
    last_answer = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_answer = m.content
            break

    # Sentiment analysis
    sentiment = analyze_sentiment(last_answer)
    sentiment_history = (state.get("sentiment_history") or []) + [sentiment["label"]]

    # Get the last question that was asked
    questions_asked = state.get("questions_asked") or []
    last_question = questions_asked[-1] if questions_asked else ""

    # LLM-as-Judge evaluation
    evaluation = score_answer(last_question, last_answer, technology)
    score = float(evaluation.get("score", 5))
    next_difficulty = evaluation.get("next_difficulty", "Medium")
    feedback = evaluation.get("feedback", "Answer received.")

    # Accumulate per-question scores in a running list stored in state
    # Key: "current_tech_scores" — list of floats for the current technology
    current_tech_scores = list(state.get("current_tech_scores") or []) + [score]

    # Append this answer to answers_given
    answers_given = (state.get("answers_given") or []) + [last_answer]
    new_q_idx = q_idx + 1

    # Sentiment badge for feedback display
    sentiment_badge = sentiment["emoji"]
    feedback_msg = f"{sentiment_badge} *{feedback}*"

    updates: dict = {
        "answers_given": answers_given,
        "sentiment_history": sentiment_history,
        "current_difficulty": next_difficulty,
        "current_question_index": new_q_idx,
        "current_tech_scores": current_tech_scores,
    }

    if new_q_idx >= MAX_QUESTIONS_PER_TECH:
        # All questions done for this technology — compute average score
        avg_score = round(sum(current_tech_scores) / len(current_tech_scores), 1)
        tech_scores = list(state.get("tech_scores") or [])
        tech_scores.append(
            {
                "technology": technology,
                "score": avg_score,
                "difficulty_reached": state.get("current_difficulty", "Easy"),
            }
        )

        next_idx = idx + 1
        updates["tech_scores"] = tech_scores
        updates["current_tech_index"] = next_idx
        updates["current_question_index"] = 0
        updates["current_difficulty"] = "Easy"
        updates["current_tech_scores"] = []  # reset for next technology

        if next_idx < len(tech_stack):
            next_tech = tech_stack[next_idx]
            updates["messages"] = [
                AIMessage(
                    content=f"{feedback_msg}\n\nGreat work! Moving on to **{next_tech}**."
                )
            ]
        else:
            # All technologies done
            updates["phase"] = "closing"
            updates["messages"] = [
                AIMessage(
                    content=f"{feedback_msg}\n\nThat concludes the technical assessment! "
                    "Generating your report now..."
                )
            ]
    else:
        updates["messages"] = [AIMessage(content=f"{feedback_msg}\n\nNext question:")]

    return updates


# ── Node 6: Closing ────────────────────────────────────────────────────────────


def closing_node(state: dict) -> dict:
    """
    Sends the farewell message and sets phase to 'ended'.
    The PDF report download is handled by app.py after this.
    """
    name = state.get("full_name") or "there"
    msg = (
        f"Thank you, **{name}**, for completing the TalentScout screening!\n\n"
        "Our team will review your responses and reach out within **3–5 business days**.\n\n"
        "Your interview report is ready to download below.\n\n"
        "Best of luck!"
    )
    return {
        "phase": "ended",
        "messages": [AIMessage(content=msg)],
    }
