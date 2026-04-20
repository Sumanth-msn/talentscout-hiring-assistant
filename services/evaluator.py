from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os, json


def get_llm():
    return ChatGroq(
        model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def score_answer(question: str, answer: str, technology: str) -> dict:
    """
    LLM-as-Judge: scores the candidate's answer 0-10.
    Returns score + brief feedback + suggested next difficulty.
    """
    llm = get_llm()
    prompt = f"""You are a strict technical interviewer evaluating a candidate's answer.

Technology: {technology}
Question: {question}
Candidate's Answer: {answer}

Evaluate and respond ONLY with valid JSON (no markdown):
{{
  "score": <integer 0-10>,
  "feedback": "<one sentence: what they got right or wrong>",
  "next_difficulty": "<Easy|Medium|Hard based on performance>"
}}

Scoring guide:
- 8-10: Complete, accurate, shows deep understanding → next: Hard
- 5-7: Partially correct, missing key details → next: Medium  
- 0-4: Incorrect or very incomplete → next: Easy
"""
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        # Strip any accidental markdown fences
        raw = response.content.strip().strip("`").replace("json\n", "")
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "score": 5,
            "feedback": "Could not parse evaluation.",
            "next_difficulty": "Medium",
        }
