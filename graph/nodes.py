import os, uuid
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from services.pii_redactor import redact_pii, redact_state_for_llm
from services.evaluator import score_answer
from services.qa_retriever import retrieve_reference_questions
from services.sentiment import analyze_sentiment
from services.report_generator import generate_report


def get_llm():
    return ChatGroq(
        model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ── Node 1: Guardrail ──────────────────────────────────────────────────────────

EXIT_KEYWORDS = {"bye", "exit", "quit", "stop", "end", "goodbye"}
TOXIC_KEYWORDS = {"hate", "kill", "abuse"}  # expand as needed


def guardrail_node(state: dict) -> dict:
    """
    First node. Checks for exit intent and toxic/irrelevant input.
    Sets guardrail_triggered=True to route to closing if needed.
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
                    content="I'm here to assist with the hiring process only. Let's keep our conversation professional. 😊"
                )
            ],
        }

    return {"guardrail_triggered": False}


# ── Node 2: Greeting ───────────────────────────────────────────────────────────


def greeting_node(state: dict) -> dict:
    """Send initial greeting and set phase to info-gathering."""
    msg = (
        "👋 Welcome to **TalentScout** — your AI-powered hiring assistant!\n\n"
        "I'll be guiding you through a brief screening process today. Here's what to expect:\n"
        "1. I'll collect some basic information about you\n"
        "2. You'll tell me about your tech stack\n"
        "3. I'll ask a few technical questions to assess your proficiency\n\n"
        "At any time, type **'bye'** to end the session.\n\n"
        "Let's begin! Could you please tell me your **full name**?"
    )
    return {
        "phase": "info",
        "session_id": str(uuid.uuid4()),
        "messages": [AIMessage(content=msg)],
    }


# ── Node 3: Info Gathering ─────────────────────────────────────────────────────

INFO_FIELDS = [
    "full_name",
    "email",
    "phone",
    "years_experience",
    "desired_position",
    "current_location",
    "tech_stack",
]

INFO_PROMPTS = {
    "full_name": "Great! What's your **email address**?",
    "email": "Thanks! And your **phone number**?",
    "phone": "Got it. How many **years of experience** do you have in tech?",
    "years_experience": "Excellent! What **position(s)** are you applying for?",
    "desired_position": "Perfect. What's your **current location** (city/country)?",
    "current_location": "Almost there! Please list your **tech stack** — include programming languages, frameworks, databases, and tools you're proficient in. (e.g., Python, Django, PostgreSQL, Docker)",
    "tech_stack": None,  # triggers transition to tech questions
}


def info_gather_node(state: dict) -> dict:
    """
    Extracts info from the last user message and advances to the next field.
    Uses LLM to parse natural language answers into structured data.
    """
    llm = get_llm()
    last_user_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    # Redact PII before sending to LLM
    safe_msg = redact_pii(last_user_msg)

    # Figure out which field we're currently filling
    current_field = None
    for field in INFO_FIELDS:
        if state.get(field) is None:
            current_field = field
            break

    if current_field is None:
        return {"phase": "tech_questions"}

    # Parse the user's answer into structured data
    parse_prompt = f"""Extract the value for field '{current_field}' from this user message.
Message: "{safe_msg}"

Rules:
- For 'tech_stack': return a JSON array of strings like ["Python", "Django"]
- For 'years_experience': return just an integer
- For everything else: return the cleaned string value
- Respond ONLY with the raw value, no explanation, no markdown.
"""
    parsed = llm.invoke([HumanMessage(content=parse_prompt)]).content.strip()

    # Store the value (raw for non-PII, redacted for PII fields)
    update = {}
    if current_field == "tech_stack":
        try:
            import json

            tech_list = json.loads(parsed)
            update["tech_stack"] = (
                tech_list if isinstance(tech_list, list) else [parsed]
            )
        except:
            update["tech_stack"] = [t.strip() for t in parsed.split(",")]
    elif current_field == "years_experience":
        try:
            update["years_experience"] = int("".join(filter(str.isdigit, parsed)))
        except:
            update["years_experience"] = 0
    elif current_field in ("email", "phone"):
        # Store original (for report), but only the redacted version goes to LLM
        update[current_field] = last_user_msg.strip()
    else:
        update[current_field] = parsed

    # Generate next prompt
    next_prompt = INFO_PROMPTS.get(current_field)
    if next_prompt:
        update["messages"] = [AIMessage(content=next_prompt)]
    else:
        # tech_stack just filled → move to tech questions
        update["phase"] = "tech_questions"
        update["current_tech_index"] = 0
        update["current_question_index"] = 0
        update["current_difficulty"] = "Easy"
        update["tech_scores"] = []
        tech_list = update.get("tech_stack", state.get("tech_stack", []))
        update["messages"] = [
            AIMessage(
                content=f"Great profile! I'll now ask you some technical questions. Let's start with **{tech_list[0]}** 🚀\n\n"
                f"Starting with an Easy question to warm up..."
            )
        ]

    return update


# ── Node 4: Technical Question ─────────────────────────────────────────────────


def tech_question_node(state: dict) -> dict:
    """
    Generates a technical question for the current technology + difficulty.
    Uses retrieved reference questions as LLM context (mini-RAG).
    """
    llm = get_llm()
    tech_stack = state.get("tech_stack", [])
    idx = state.get("current_tech_index", 0)
    difficulty = state.get("current_difficulty", "Easy")

    if idx >= len(tech_stack):
        return {"phase": "closing"}

    technology = tech_stack[idx]

    # Retrieve reference questions (mini-RAG)
    references = retrieve_reference_questions(technology, difficulty, n=2)
    ref_text = (
        "\n".join(f"- {q}" for q in references)
        if references
        else "No references found."
    )

    safe_state = redact_state_for_llm(state)

    prompt = f"""You are a technical interviewer for TalentScout.
Generate ONE {difficulty}-level interview question about {technology}.

Candidate profile:
- Experience: {safe_state.get("years_experience")} years
- Desired role: {safe_state.get("desired_position")}

Reference questions (use as inspiration, DO NOT copy verbatim):
{ref_text}

Requirements:
- Single, clear question only
- Appropriate for {difficulty} level
- Directly tests practical {technology} knowledge
- No preamble, just the question
"""

    question = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    return {
        "messages": [AIMessage(content=f"**[{difficulty}]** {question}")],
        "questions_asked": state.get("questions_asked", []) + [question],
    }


# ── Node 5: Evaluator ──────────────────────────────────────────────────────────


def evaluator_node(state: dict) -> dict:
    """
    Scores the candidate's answer, updates difficulty (DDA),
    and decides whether to ask more questions or move to next tech.
    """
    tech_stack = state.get("tech_stack", [])
    idx = state.get("current_tech_index", 0)
    q_idx = state.get("current_question_index", 0)
    technology = tech_stack[idx] if idx < len(tech_stack) else "Unknown"

    # Get last user answer
    last_answer = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_answer = m.content
            break

    # Sentiment analysis
    sentiment = analyze_sentiment(last_answer)
    sentiment_history = state.get("sentiment_history", []) + [sentiment["label"]]

    # Get last question
    last_question = ""
    if state.get("questions_asked"):
        last_question = state["questions_asked"][-1]

    # LLM-as-Judge scoring
    evaluation = score_answer(last_question, last_answer, technology)
    score = evaluation.get("score", 5)
    next_difficulty = evaluation.get("next_difficulty", "Medium")
    feedback = evaluation.get("feedback", "")

    answers_given = state.get("answers_given", []) + [last_answer]

    # Update tech scores
    tech_scores = list(state.get("tech_scores", []))

    # Max 3 questions per technology
    MAX_QUESTIONS_PER_TECH = 3
    new_q_idx = q_idx + 1

    updates = {
        "answers_given": answers_given,
        "sentiment_history": sentiment_history,
        "current_difficulty": next_difficulty,
        "current_question_index": new_q_idx,
    }

    # Show feedback to candidate
    sentiment_badge = sentiment["emoji"]
    feedback_msg = f"{sentiment_badge} *{feedback}*"

    if new_q_idx >= MAX_QUESTIONS_PER_TECH:
        # Done with this technology — compute average score
        tech_scores.append(
            {
                "technology": technology,
                "score": float(score),
                "difficulty_reached": state.get("current_difficulty", "Easy"),
            }
        )
        updates["tech_scores"] = tech_scores
        updates["current_tech_index"] = idx + 1
        updates["current_question_index"] = 0
        updates["current_difficulty"] = "Easy"

        next_idx = idx + 1
        if next_idx < len(tech_stack):
            next_tech = tech_stack[next_idx]
            updates["messages"] = [
                AIMessage(
                    content=f"{feedback_msg}\n\nGreat! Let's move to **{next_tech}** now. 🔄"
                )
            ]
        else:
            updates["phase"] = "closing"
            updates["messages"] = [
                AIMessage(
                    content=f"{feedback_msg}\n\nThat wraps up the technical section! Generating your report..."
                )
            ]
    else:
        updates["messages"] = [AIMessage(content=f"{feedback_msg}\n\nNext question:")]

    return updates


# ── Node 6: Closing ────────────────────────────────────────────────────────────


def closing_node(state: dict) -> dict:
    """Generate farewell message and trigger PDF report."""
    name = state.get("full_name", "there")
    msg = (
        f"Thank you, **{name}**, for completing the TalentScout screening! 🎉\n\n"
        "Our team will review your responses and reach out within **3–5 business days**.\n\n"
        "📄 Your interview report has been generated — you can download it below.\n\n"
        "Best of luck! 🚀"
    )
    return {"phase": "ended", "messages": [AIMessage(content=msg)]}
