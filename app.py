"""
TalentScout — AI Hiring Assistant
app.py: Streamlit UI + orchestration logic.

Key design decision: all node calls after the initial greeting are made
DIRECTLY (not via graph.invoke) so we can merge list fields correctly
without LangGraph overwriting them.
"""

import re
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from graph.edges import build_graph
from services.report_generator import generate_report

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TalentScout",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:         #080808;
    --bg2:        #0f0f0f;
    --bg3:        #141414;
    --bg4:        #1a1a1a;
    --red:        #ff2d2d;
    --red-dim:    #c0162a;
    --red-glow:   rgba(255,45,45,0.10);
    --red-border: rgba(255,45,45,0.28);
    --white:      #f0f0f0;
    --grey:       #777;
    --grey-dim:   #3a3a3a;
    --border:     rgba(255,255,255,0.06);
    --border2:    rgba(255,255,255,0.11);
    --font-head:  'Bebas Neue', sans-serif;
    --font-body:  'Inter', sans-serif;
    --font-mono:  'JetBrains Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: var(--bg) !important;
    font-family: var(--font-body) !important;
    color: var(--white) !important;
}

#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"] { display: none !important; }

[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
    overflow-x: hidden !important;
}

[data-testid="stMainBlockContainer"] {
    padding: 0 2.5rem 5rem !important;
    max-width: 820px !important;
    margin: 0 auto !important;
}

/* ── Hero ── */
.ts-hero { text-align:center; padding:2.8rem 0 1.8rem; }
.ts-hero-eyebrow {
    font-family:var(--font-mono); font-size:0.62rem;
    letter-spacing:0.24em; color:var(--red);
    text-transform:uppercase; margin-bottom:0.5rem;
}
.ts-hero-title {
    font-family:var(--font-head);
    font-size:clamp(3.8rem,9vw,7rem);
    line-height:0.92; letter-spacing:0.05em;
    color:var(--white); margin:0;
}
.ts-hero-title .red-word {
    background:linear-gradient(130deg,#ff2d2d 0%,#ff7070 55%,#c0162a 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
}
.ts-hero-sub {
    font-family:var(--font-mono); font-size:0.65rem;
    letter-spacing:0.16em; color:var(--grey);
    margin-top:0.9rem; text-transform:uppercase;
}
.ts-hero-rule {
    width:48px; height:1px;
    background:linear-gradient(90deg,transparent,var(--red),transparent);
    margin:1.4rem auto 0;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background:transparent !important; border:none !important;
    padding:0.45rem 0 !important; gap:0.7rem !important;
    align-items:flex-start !important;
}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"],
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
    width:30px !important; height:30px !important; min-width:30px !important;
    border-radius:6px !important; border:1px solid var(--border2) !important;
    background:var(--bg3) !important; font-family:var(--font-mono) !important;
    font-size:0.5rem !important; color:var(--grey) !important;
    flex-shrink:0 !important; display:flex !important;
    align-items:center !important; justify-content:center !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    background:var(--bg3) !important; border:1px solid var(--border) !important;
    border-left:2px solid var(--red-dim) !important;
    border-radius:2px 12px 12px 12px !important;
    padding:1rem 1.2rem !important; font-size:0.875rem !important;
    line-height:1.8 !important; color:#ddd !important; max-width:88% !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction:row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background:var(--bg4) !important; border:1px solid var(--border2) !important;
    border-radius:12px 2px 12px 12px !important;
    padding:0.8rem 1.1rem !important; font-size:0.875rem !important;
    line-height:1.7 !important; color:var(--white) !important;
    max-width:78% !important; margin-left:auto !important;
}
[data-testid="stChatMessage"] .stMarkdown strong { color:var(--white) !important; font-weight:600 !important; }
[data-testid="stChatMessage"] .stMarkdown em     { color:var(--grey) !important; }
[data-testid="stChatMessage"] .stMarkdown code {
    font-family:var(--font-mono) !important; font-size:0.78rem !important;
    background:rgba(255,45,45,0.07) !important; border:1px solid var(--red-border) !important;
    border-radius:4px !important; padding:0.1em 0.4em !important; color:#ff8888 !important;
}

/* ── Chat input ── */
[data-testid="stChatInputContainer"] {
    background:var(--bg2) !important; border:1px solid var(--border2) !important;
    border-radius:12px !important; padding:4px 8px !important;
    transition:border-color 0.2s,box-shadow 0.2s !important;
}
[data-testid="stChatInputContainer"]:focus-within {
    border-color:rgba(255,45,45,0.45) !important;
    box-shadow:0 0 0 3px var(--red-glow) !important;
}
[data-testid="stChatInputContainer"] textarea {
    background:transparent !important; color:var(--white) !important;
    font-family:var(--font-body) !important; font-size:0.875rem !important;
    caret-color:var(--red) !important;
}
[data-testid="stChatInputContainer"] textarea::placeholder { color:var(--grey-dim) !important; }
[data-testid="stChatInputContainer"] button {
    background:var(--red) !important; border:none !important;
    border-radius:8px !important; transition:background 0.15s,transform 0.1s !important;
}
[data-testid="stChatInputContainer"] button:hover { background:#ff4444 !important; transform:scale(1.04) !important; }
[data-testid="stChatInputContainer"] button svg { fill:white !important; }

/* ── Spinner ── */
div[data-testid="stSpinner"] p {
    color:var(--grey) !important; font-family:var(--font-mono) !important;
    font-size:0.68rem !important; letter-spacing:0.12em !important; text-transform:uppercase !important;
}
[data-testid="stSpinner"] > div { border-color:var(--red) transparent transparent transparent !important; }

/* ── Sidebar internals ── */
.ts-sb-brand { padding:1.4rem 1.2rem 1.1rem; border-bottom:1px solid var(--border); }
.ts-sb-name  { font-family:var(--font-head); font-size:1.35rem; letter-spacing:0.07em; color:var(--white); line-height:1; }
.ts-sb-tagline { font-family:var(--font-mono); font-size:0.55rem; letter-spacing:0.14em; color:var(--grey-dim); margin-top:4px; text-transform:uppercase; }
.ts-sb-status {
    display:inline-flex; align-items:center; gap:6px;
    margin:1rem 1.2rem 0; padding:0.26rem 0.65rem;
    border:1px solid var(--red-border); border-radius:3px; background:var(--red-glow);
}
.ts-sb-status-dot { width:5px; height:5px; border-radius:50%; background:var(--red); animation:blink 1.8s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
.ts-sb-status-text { font-family:var(--font-mono); font-size:0.6rem; letter-spacing:0.1em; color:var(--red); text-transform:uppercase; }
.ts-sb-divider { height:1px; background:var(--border); margin:0.9rem 1.2rem; }
.ts-sb-label { font-family:var(--font-mono); font-size:0.55rem; letter-spacing:0.16em; color:var(--grey-dim); text-transform:uppercase; padding:0 1.2rem; margin-bottom:0.3rem; }
.ts-sb-field { display:flex; align-items:center; justify-content:space-between; padding:0.35rem 1.2rem; gap:8px; }
.ts-sb-field-label { font-size:0.76rem; color:var(--grey); font-family:var(--font-body); flex-shrink:0; }
.ts-sb-field-value { font-size:0.72rem; color:var(--white); font-family:var(--font-mono); text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:130px; }
.ts-sb-dot-on  { width:4px; height:4px; border-radius:50%; background:var(--red); flex-shrink:0; }
.ts-sb-dot-off { width:4px; height:4px; border-radius:50%; background:var(--grey-dim); flex-shrink:0; }
.ts-sb-pills { display:flex; flex-wrap:wrap; gap:5px; padding:0.35rem 1.2rem 0.7rem; }
.ts-sb-pill { font-family:var(--font-mono); font-size:0.6rem; padding:0.16rem 0.48rem; border:1px solid var(--red-border); border-radius:3px; color:#ff8888; background:var(--red-glow); letter-spacing:0.04em; white-space:nowrap; }
.ts-sb-score { padding:0.2rem 1.2rem; }
.ts-sb-score-row { display:flex; justify-content:space-between; margin-bottom:3px; }
.ts-sb-score-tech { font-size:0.7rem; color:var(--grey); font-family:var(--font-body); }
.ts-sb-score-num  { font-size:0.68rem; color:var(--white); font-family:var(--font-mono); }
.ts-sb-track { height:2px; background:var(--border2); border-radius:2px; overflow:hidden; margin-bottom:0.5rem; }
.ts-sb-fill  { height:100%; background:linear-gradient(90deg,var(--red-dim),var(--red)); border-radius:2px; }
.ts-diff-wrap { padding:0 1.2rem 0.8rem; }
.ts-diff-tag { display:inline-block; font-family:var(--font-mono); font-size:0.58rem; letter-spacing:0.1em; padding:0.14rem 0.52rem; border-radius:3px; text-transform:uppercase; }
.ts-diff-easy   { color:#4ade80; background:rgba(74,222,128,0.07); border:1px solid rgba(74,222,128,0.18); }
.ts-diff-medium { color:#fbbf24; background:rgba(251,191,36,0.07); border:1px solid rgba(251,191,36,0.18); }
.ts-diff-hard   { color:var(--red); background:var(--red-glow); border:1px solid var(--red-border); }
.ts-sb-footer { padding:0.6rem 1.2rem 1.4rem; font-family:var(--font-mono); font-size:0.54rem; color:var(--grey-dim); letter-spacing:0.1em; line-height:1.9; text-transform:uppercase; }

/* ── Sentiment badges ── */
.ts-sent { display:inline-block; font-family:var(--font-mono); font-size:0.58rem; letter-spacing:0.08em; padding:0.09rem 0.42rem; border-radius:3px; text-transform:uppercase; }
.ts-sent-pos { color:#4ade80; background:rgba(74,222,128,0.07); border:1px solid rgba(74,222,128,0.18); }
.ts-sent-neu { color:#94a3b8; background:rgba(148,163,184,0.05); border:1px solid rgba(148,163,184,0.16); }
.ts-sent-neg { color:var(--red); background:var(--red-glow); border:1px solid var(--red-border); }

/* ── Session end ── */
.ts-end { margin:2.5rem auto; max-width:460px; text-align:center; border:1px solid var(--border2); border-radius:16px; padding:2.8rem 2rem; background:var(--bg2); }
.ts-end-title { font-family:var(--font-head); font-size:2.4rem; letter-spacing:0.07em; color:var(--white); margin-bottom:0.4rem; }
.ts-end-sub { font-family:var(--font-mono); font-size:0.62rem; letter-spacing:0.16em; color:var(--grey); margin-bottom:1.8rem; text-transform:uppercase; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background:transparent !important; border:1px solid var(--red-border) !important;
    border-radius:8px !important; color:var(--red) !important;
    font-family:var(--font-mono) !important; font-size:0.72rem !important;
    letter-spacing:0.1em !important; text-transform:uppercase !important;
    padding:0.7rem 1.5rem !important; width:100% !important;
    transition:background 0.2s,border-color 0.2s !important;
}
[data-testid="stDownloadButton"] > button:hover { background:var(--red-glow) !important; border-color:var(--red) !important; }

::-webkit-scrollbar { width:3px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--grey-dim); border-radius:3px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────


def merge(state: dict, delta: dict) -> dict:
    """
    Safely merge a node's output delta into the full state.

    Rules:
    - 'messages': always APPEND delta items to existing list
    - 'questions_asked', 'answers_given', 'sentiment_history':
        node returns the FULL list (existing + new), so just replace
    - 'tech_scores': node returns the FULL list, so just replace
    - Everything else: overwrite directly
    """
    APPEND_KEYS = {"messages"}
    REPLACE_KEYS = {
        "questions_asked",
        "answers_given",
        "sentiment_history",
        "tech_scores",
    }

    for k, v in delta.items():
        if k in APPEND_KEYS and isinstance(v, list):
            state[k] = state.get(k, []) + v
        else:
            state[k] = v

    return state


def format_message(content: str) -> str:
    """Convert LLM output to styled HTML for chat display."""

    def _diff(m):
        lvl = m.group(1)
        return f'<span class="ts-diff-tag ts-diff-{lvl.lower()}">{lvl}</span>'

    content = re.sub(r"\*{0,2}\[?(Easy|Medium|Hard)\]?\*{0,2}", _diff, content)
    content = content.replace(
        "🟢", '<span class="ts-sent ts-sent-pos">Confident</span>'
    )
    content = content.replace("🟡", '<span class="ts-sent ts-sent-neu">Neutral</span>')
    content = content.replace(
        "🔴", '<span class="ts-sent ts-sent-neg">Uncertain</span>'
    )
    return content


def render_sidebar(state: dict) -> None:
    """Render the sidebar. Always called before st.stop() so it persists."""
    with st.sidebar:
        phase = state.get("phase", "greeting")
        phase_labels = {
            "greeting": "Initialising",
            "info": "Info Gathering",
            "tech_questions": "Technical Screen",
            "closing": "Closing",
            "ended": "Complete",
        }
        phase_display = phase_labels.get(phase, phase.title())

        st.markdown(
            f"""
        <div class="ts-sb-brand">
            <div class="ts-sb-name">TalentScout</div>
            <div class="ts-sb-tagline">AI Hiring Assistant · v1.0</div>
        </div>
        <div class="ts-sb-status">
            <span class="ts-sb-status-dot"></span>
            <span class="ts-sb-status-text">{phase_display}</span>
        </div>
        <div class="ts-sb-divider"></div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="ts-sb-label">Candidate</div>', unsafe_allow_html=True)
        fields = [
            ("Name", state.get("full_name")),
            ("Email", "Collected" if state.get("email") else None),
            ("Phone", "Collected" if state.get("phone") else None),
            (
                "Exp",
                f"{state.get('years_experience')} yrs"
                if state.get("years_experience")
                else None,
            ),
            ("Role", state.get("desired_position")),
            ("Location", state.get("current_location")),
        ]
        rows = ""
        for lbl, val in fields:
            dot = (
                '<span class="ts-sb-dot-on"></span>'
                if val
                else '<span class="ts-sb-dot-off"></span>'
            )
            vhtml = (
                f'<span class="ts-sb-field-value">{val}</span>'
                if val
                else '<span style="font-family:var(--font-mono);font-size:0.68rem;color:var(--grey-dim);">—</span>'
            )
            rows += f'<div class="ts-sb-field">{dot}<span class="ts-sb-field-label">{lbl}</span>{vhtml}</div>'
        st.markdown(rows, unsafe_allow_html=True)

        tech_stack = state.get("tech_stack")
        if tech_stack:
            st.markdown(
                '<div class="ts-sb-divider"></div><div class="ts-sb-label">Stack</div>',
                unsafe_allow_html=True,
            )
            pills = "".join(f'<span class="ts-sb-pill">{t}</span>' for t in tech_stack)
            st.markdown(
                f'<div class="ts-sb-pills">{pills}</div>', unsafe_allow_html=True
            )

        tech_scores = state.get("tech_scores")
        if tech_scores:
            st.markdown(
                '<div class="ts-sb-divider"></div><div class="ts-sb-label">Assessment</div>',
                unsafe_allow_html=True,
            )
            bars = ""
            for ts in tech_scores:
                pct = min(100, int(ts["score"] * 10))
                bars += f"""
                <div class="ts-sb-score">
                    <div class="ts-sb-score-row">
                        <span class="ts-sb-score-tech">{ts["technology"]}</span>
                        <span class="ts-sb-score-num">{ts["score"]:.1f}</span>
                    </div>
                    <div class="ts-sb-track"><div class="ts-sb-fill" style="width:{pct}%"></div></div>
                </div>"""
            st.markdown(bars, unsafe_allow_html=True)

        if phase == "tech_questions":
            diff = state.get("current_difficulty", "Easy")
            st.markdown(
                f"""
            <div class="ts-sb-divider"></div>
            <div class="ts-sb-label">Difficulty</div>
            <div class="ts-diff-wrap">
                <span class="ts-diff-tag ts-diff-{diff.lower()}">{diff}</span>
            </div>""",
                unsafe_allow_html=True,
            )

        q_count = len(state.get("questions_asked") or [])
        st.markdown(
            f"""
        <div class="ts-sb-divider"></div>
        <div class="ts-sb-footer">
            GDPR compliant · PII redacted<br>
            Questions asked: {q_count}
        </div>""",
            unsafe_allow_html=True,
        )


# ── Session State Init ─────────────────────────────────────────────────────────

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "cs" not in st.session_state:
    # All list fields start empty; all optional fields start as None
    # IMPORTANT: tech_stack starts as None so info_gather_node can detect it
    initial: dict = {
        "messages": [],
        "full_name": None,
        "email": None,
        "phone": None,
        "years_experience": None,
        "desired_position": None,
        "current_location": None,
        "tech_stack": None,  # ← None, not []
        "current_tech_index": 0,
        "current_question_index": 0,
        "current_difficulty": "Easy",
        "questions_asked": [],
        "answers_given": [],
        "tech_scores": [],
        "sentiment_history": [],
        "phase": "greeting",
        "guardrail_triggered": False,
        "session_id": "",
    }
    # Fire greeting immediately — use graph.invoke only for this first call
    from graph.nodes import greeting_node, guardrail_node

    g_delta = guardrail_node(initial)
    initial = merge(initial, g_delta)
    gr_delta = greeting_node(initial)
    initial = merge(initial, gr_delta)
    st.session_state.cs = initial

cs = st.session_state.cs

# ── Always render sidebar first (before any st.stop) ──────────────────────────
render_sidebar(cs)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="ts-hero">
    <div class="ts-hero-eyebrow">AI-Powered Candidate Screening</div>
    <h1 class="ts-hero-title">TALENT<span class="red-word">SCOUT</span></h1>
    <p class="ts-hero-sub">Precision hiring &nbsp;&middot;&nbsp; Adaptive assessment &nbsp;&middot;&nbsp; Instant insights</p>
    <div class="ts-hero-rule"></div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Session end screen ─────────────────────────────────────────────────────────
if cs.get("phase") == "ended":
    st.markdown(
        """
    <div class="ts-end">
        <div class="ts-end-title">SCREENING COMPLETE</div>
        <div class="ts-end-sub">Your report is ready for download</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if cs.get("tech_scores"):
        pdf_bytes = generate_report(cs)
        name = (cs.get("full_name") or "candidate").replace(" ", "_")
        st.download_button(
            label="DOWNLOAD INTERVIEW REPORT  —  PDF",
            data=pdf_bytes,
            file_name=f"talentscout_{name}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown(
        """
    <p style="text-align:center;margin-top:1.4rem;
       font-family:var(--font-mono);font-size:0.6rem;
       color:var(--grey-dim);letter-spacing:0.14em;">
        REFRESH PAGE TO START A NEW SESSION
    </p>
    """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in cs.get("messages", []):
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="assistant"):
            st.markdown(format_message(msg.content), unsafe_allow_html=True)
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="user"):
            st.markdown(msg.content)

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Type your response...")

if user_input:
    from graph.nodes import (
        guardrail_node,
        info_gather_node,
        tech_question_node,
        evaluator_node,
        closing_node,
    )

    cs = st.session_state.cs
    cs["messages"].append(HumanMessage(content=user_input))
    phase = cs.get("phase", "greeting")

    with st.spinner("Analysing..."):
        # ── Step 1: Always run guardrail first ──
        g_delta = guardrail_node(cs)
        cs = merge(cs, g_delta)

        if cs.get("guardrail_triggered"):
            # Exit or toxic input detected
            if cs.get("phase") == "closing":
                close_delta = closing_node(cs)
                cs = merge(cs, close_delta)

        else:
            # ── Step 2: Route based on phase ──

            if phase == "info":
                info_delta = info_gather_node(cs)
                cs = merge(cs, info_delta)

                # If tech_stack was just collected → ask first question immediately
                if cs.get("phase") == "tech_questions":
                    q_delta = tech_question_node(cs)
                    cs = merge(cs, q_delta)

            elif phase == "tech_questions":
                questions = cs.get("questions_asked") or []
                answers = cs.get("answers_given") or []

                if len(questions) > len(answers):
                    # Candidate just answered a question → evaluate
                    eval_delta = evaluator_node(cs)
                    cs = merge(cs, eval_delta)

                    if cs.get("phase") == "closing":
                        close_delta = closing_node(cs)
                        cs = merge(cs, close_delta)
                    else:
                        # Ask next question immediately
                        q_delta = tech_question_node(cs)
                        cs = merge(cs, q_delta)
                else:
                    # No unanswered question — shouldn't normally happen,
                    # but generate next question as fallback
                    q_delta = tech_question_node(cs)
                    cs = merge(cs, q_delta)

            elif phase == "greeting":
                # User typed during greeting — treat as name answer
                from graph.nodes import greeting_node as _gn

                info_delta = info_gather_node(cs)
                cs = merge(cs, info_delta)

    st.session_state.cs = cs
    st.rerun()
