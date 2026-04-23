# TalentScout — AI Hiring Assistant

> An intelligent hiring assistant that screens candidates through adaptive technical interviews, evaluates responses in real time, and generates a PDF talent report for recruiters.

Built for the **PGAGI AI/ML Internship Assignment**.

---

## Project Overview

TalentScout conducts end-to-end candidate screening through a structured conversation:

- Collects candidate profile (name, contact, experience, role, location, tech stack)
- Generates tailored technical questions per technology in the declared stack
- Evaluates answers using **LLM-as-Judge** and adjusts difficulty dynamically
- Produces a **PDF Talent Report** with scores and behavioural signals at session end

The bot is built on a **LangGraph state machine** — six single-responsibility nodes route the conversation deterministically. There is no monolithic prompt loop; every phase (greeting → info gathering → technical interview → closing) is a separate, testable function with typed state.

**Key differentiators over a basic chatbot:**
- Dynamic Difficulty Adjustment (Easy → Medium → Hard based on performance)
- PII Redaction Layer — email and phone are never sent to the LLM
- Mini-RAG grounding — questions are inspired by a curated `tech_qa_bank.json`, not generated cold
- Sentiment tracking per answer with confidence badge displayed inline

---

## Installation

**Prerequisites:** Python 3.12+, [uv](https://github.com/astral-sh/uv), a [Groq API key](https://console.groq.com)

```bash
git clone https://github.com/your-username/talentscout-hiring-assistant.git
cd talentscout-hiring-assistant

uv venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

cp .env.example .env                   # Add your GROQ_API_KEY here

python -m textblob.download_corpora    # One-time setup for sentiment

streamlit run app.py
```

**.env.example**
```
GROQ_API_KEY=your_key_here
MODEL_NAME=llama-3.3-70b-versatile
```

**Optional — silence transformer path warnings:**
```toml
# .streamlit/config.toml
[server]
fileWatcherType = "none"
```

---

## Usage Guide

1. App opening the app, it greets automatically
2. Answer each prompt naturally — the LLM extracts structured data from free text
3. After the tech stack is submitted, technical questions begin immediately
4. Each answer receives feedback + a difficulty adjustment for the next question
5. Type `bye` / `exit` / `quit` at any point to end the session gracefully
6. On completion, download the **PDF Talent Report** from the end screen

The **sidebar** shows live session state: collected fields, tech stack pills, per-tech score bars, and current difficulty level.

---

## Technical Details

### Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph — typed state machine, 6 nodes |
| LLM | Groq · `llama-3.3-70b-versatile` via `langchain-groq` |
| Frontend | Streamlit — fully custom CSS dark theme |
| Retrieval | `sentence-transformers` (MiniLM) + scikit-learn cosine similarity |
| PII Redaction | Regex — email, phone, Aadhaar patterns stripped before LLM call |
| Sentiment | TextBlob — polarity score → Confident / Neutral / Uncertain badge |
| PDF Report | FPDF2 — scores, bar chart, behavioural signals table |
| Runtime | Python 3.12, uv package manager |

### Architecture

```
User message
    │
    ▼
GUARDRAIL NODE          ← exit keywords, toxic input, off-topic detection
    │
    ├── GREETING NODE   ← one-time welcome, sets phase = "info"
    │
    ├── INFO GATHER NODE ← sequential field extraction via LLM parsing
    │       │
    │       └── (tech stack collected) ──►
    │
    ├── TECH QUESTION NODE  ← mini-RAG grounded question generation
    │
    ├── EVALUATOR NODE  ← LLM-as-Judge scoring + DDA + sentiment
    │       │
    │       └── (all techs done) ──►
    │
    └── CLOSING NODE    ← farewell message, phase = "ended", PDF download
```

All state flows through a single `CandidateState` TypedDict. Nodes are called **directly** in `app.py` (not via `graph.invoke`) using a custom `merge()` function that appends to `messages` and correctly replaces other list fields — avoiding LangGraph's default overwrite behaviour.

### Project Structure

```
talentscout-hiring-assistant/
├── app.py                    # Streamlit UI + conversation orchestration
├── graph/
│   ├── state.py              # CandidateState TypedDict
│   ├── nodes.py              # 6 LangGraph nodes
│   └── edges.py              # Graph definition
├── services/
│   ├── pii_redactor.py       # Regex PII layer
│   ├── qa_retriever.py       # Mini-RAG retrieval
│   ├── evaluator.py          # LLM-as-Judge
│   ├── sentiment.py          # TextBlob sentiment
│   └── report_generator.py   # PDF generation
└── data/
    └── tech_qa_bank.json     # 60+ curated Q&A reference pairs
```

---

## Prompt Design

### Information Extraction

Each of the 7 candidate fields uses a **dedicated single-field extraction prompt** rather than one large form prompt. This prevents field confusion and handles natural language variations reliably.

```
"Extract the value for field 'tech_stack' from this message.
Return a JSON array like ["Python", "Django"].
Respond ONLY with the raw value. No explanation, no markdown."
```

### Technical Question Generation

Questions are grounded with 2 reference entries retrieved from `tech_qa_bank.json` via cosine similarity. The LLM uses them as inspiration — not verbatim output — ensuring variety while staying on-topic.

```
"Generate ONE {difficulty}-level interview question about {technology}.
Reference questions (inspiration only — do NOT copy):
{retrieved_references}
Output only the question, no preamble."
```

### LLM-as-Judge Evaluation

A strict JSON-only prompt with explicit scoring rubric drives the evaluator:

```
"Respond ONLY with valid JSON:
{ "score": 0-10, "feedback": "one sentence", "next_difficulty": "Easy|Medium|Hard" }

8-10 → Hard next   |   5-7 → Medium next   |   0-4 → Easy next"
```

The `next_difficulty` value directly sets the difficulty for the next question — creating the adaptive interview loop without any additional routing logic.

### Guardrail

Exit and toxicity detection uses **deterministic keyword matching**, not LLM inference — making it instant, reliable, and impossible to confuse.

---

## Challenges & Solutions

**1. Designing a stateful multi-turn interview without losing context**
The core challenge was maintaining coherent conversation state across multiple LLM calls in different phases. A single-prompt chatbot loses track of where it is. The solution was a typed `CandidateState` dict that every node reads from and writes to — giving the system a single source of truth for the entire session.

**2. Making LLM field extraction reliable across free-text answers**
Candidates answer questions differently ("I have 5 years exp", "5", "around 5 years"). Rather than one large extraction prompt, each field gets its own targeted prompt with explicit output rules. This made extraction accurate enough to not require retry logic in practice.

**3. Building an adaptive difficulty system without extra infrastructure**
Dynamic Difficulty Adjustment could have required a separate scoring service. Instead, the evaluator prompt returns `next_difficulty` as a JSON field — the LLM judges and routes in a single call. The state machine picks it up and passes it to the next question node automatically.

**4. Handling list state correctly across node boundaries**
LangGraph's state merge replaces list fields entirely when a node returns a partial delta. This caused `questions_asked` and `answers_given` to reset on every call. The fix was bypassing `graph.invoke()` for the main loop and writing a `merge()` function with explicit rules: append-only for `messages`, full-replace for other lists (since nodes always return the complete updated list).

**5. Keeping PII out of the LLM without breaking conversation flow**
Candidates provide email and phone naturally mid-conversation. The redactor runs on every user message before it reaches any LLM call — replacing sensitive patterns with `[REDACTED_EMAIL]` / `[REDACTED_PHONE]` tokens. The original values are stored in session state for the PDF report but never transmitted to Groq.

---

## Code Quality

- **Modular structure** — each service (`pii_redactor`, `evaluator`, `qa_retriever`, `sentiment`, `report_generator`) is independently importable and testable
- **Typed state** — `CandidateState` TypedDict makes every field explicit; no implicit dict key access
- **Docstrings on every function** — nodes, services, and helpers all document their inputs, outputs, and side effects
- **No bare `except`** — all exception handling catches specific types (`json.JSONDecodeError`, `ValueError`)
- **Git history** — commits follow the format `feat:`, `fix:`, `docs:`, `chore:` for clean history

---

## Demo

> **Live App:** _[https://talentscout-hiring-assistant-by-sumanth-msn.streamlit.app]_

---

*Built with [LangGraph](https://github.com/langchain-ai/langgraph) · [Groq](https://groq.com) · [Streamlit](https://streamlit.io)*
