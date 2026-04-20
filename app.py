import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from graph.edges import build_graph
from graph.state import CandidateState
from services.report_generator import generate_report

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TalentScout",
    page_icon="assets/favicon.png",  # optional: drop a 32x32 PNG here
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ──────────────────────────────────────────────────────────────
# Fonts loaded from Google Fonts; fallback stack included.
# Theme: deep obsidian + arterial crimson.
CUSTOM_CSS = """
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Root ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --obsidian:      #0A0A0C;
    --obsidian-2:    #111116;
    --obsidian-3:    #18181F;
    --obsidian-4:    #21212B;
    --surface:       #1C1C24;
    --surface-hover: #23232E;

    --crimson:       #C0162A;
    --crimson-light: #E8213E;
    --crimson-dim:   #8B0F1E;
    --crimson-glow:  rgba(192, 22, 42, 0.18);

    --text-primary:   #F0EEE9;
    --text-secondary: #9A9898;
    --text-muted:     #5C5C66;
    --text-accent:    #E8213E;

    --border:        rgba(255,255,255,0.06);
    --border-accent: rgba(192,22,42,0.35);
    --border-strong: rgba(255,255,255,0.12);

    --radius-sm:  6px;
    --radius-md:  10px;
    --radius-lg:  16px;
    --radius-xl:  24px;

    --font-display: 'DM Serif Display', Georgia, serif;
    --font-body:    'DM Sans', system-ui, sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
}

/* ── Global App Shell ── */
.stApp {
    background: var(--obsidian) !important;
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}

/* Subtle grid texture on the main background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(192,22,42,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(192,22,42,0.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--obsidian-2) !important;
    border-right: 1px solid var(--border) !important;
    padding: 0 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}

/* ── Main content area padding ── */
[data-testid="stMainBlockContainer"] {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 860px !important;
    margin: 0 auto !important;
    position: relative;
    z-index: 1;
}

/* ── Streamlit default overrides ── */
h1, h2, h3, h4 {
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
    font-weight: 400 !important;
}
p, li, label, span, div {
    font-family: var(--font-body) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Wordmark / Brand Header ── */
.ts-wordmark {
    display: flex;
    align-items: baseline;
    gap: 10px;
    padding: 2rem 1.5rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0;
}
.ts-wordmark-logo {
    font-family: var(--font-display);
    font-size: 1.35rem;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.ts-wordmark-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--crimson);
    display: inline-block;
    position: relative;
    top: -2px;
}

/* ── Sidebar Section Labels ── */
.ts-sidebar-section {
    font-family: var(--font-mono) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.12em !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    padding: 1.25rem 1.5rem 0.5rem !important;
}

/* ── Progress Field Row ── */
.ts-field-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.45rem 1.5rem;
    gap: 8px;
}
.ts-field-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    font-family: var(--font-body);
    flex-shrink: 0;
}
.ts-field-value {
    font-size: 0.8rem;
    color: var(--text-primary);
    font-family: var(--font-mono);
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 140px;
}
.ts-field-pending {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
}
.ts-field-dot-filled {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--crimson);
    flex-shrink: 0;
}
.ts-field-dot-empty {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--border-strong);
    flex-shrink: 0;
}

/* ── Phase Badge ── */
.ts-phase-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin: 1rem 1.5rem 0;
    padding: 0.3rem 0.65rem;
    border: 1px solid var(--border-accent);
    border-radius: 4px;
    background: var(--crimson-glow);
}
.ts-phase-badge-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--crimson-light);
    animation: pulse 2s ease-in-out infinite;
}
.ts-phase-badge-text {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--crimson-light);
    letter-spacing: 0.08em;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Tech Stack Pills ── */
.ts-tech-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 0.5rem 1.5rem 1rem;
}
.ts-tech-pill {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    padding: 0.2rem 0.55rem;
    border: 1px solid var(--border-accent);
    border-radius: 4px;
    color: var(--crimson-light);
    background: var(--crimson-glow);
    letter-spacing: 0.04em;
}

/* ── Score Bars ── */
.ts-score-row {
    padding: 0.3rem 1.5rem;
}
.ts-score-label-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
}
.ts-score-tech {
    font-size: 0.75rem;
    color: var(--text-secondary);
    font-family: var(--font-body);
}
.ts-score-num {
    font-size: 0.75rem;
    color: var(--text-primary);
    font-family: var(--font-mono);
}
.ts-score-track {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}
.ts-score-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--crimson-dim), var(--crimson-light));
    border-radius: 2px;
    transition: width 0.6s ease;
}

/* ── Sidebar Divider ── */
.ts-divider {
    height: 1px;
    background: var(--border);
    margin: 1rem 1.5rem;
}

/* ── Main Page Header ── */
.ts-page-header {
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.ts-page-title {
    font-family: var(--font-display) !important;
    font-size: 2.2rem !important;
    color: var(--text-primary) !important;
    line-height: 1.15 !important;
    margin-bottom: 0.4rem !important;
}
.ts-page-title span {
    color: var(--crimson-light);
}
.ts-page-subtitle {
    font-size: 0.82rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
    letter-spacing: 0.06em;
}

/* ── Chat Container ── */
.ts-chat-container {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    padding-bottom: 1rem;
}

/* ── Chat Messages (overriding Streamlit's) ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    gap: 0 !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 var(--radius-lg) var(--radius-lg) var(--radius-lg) !important;
    padding: 1rem 1.25rem !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
    color: var(--text-primary) !important;
    max-width: 82% !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background: var(--obsidian-4) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-lg) 0 var(--radius-lg) var(--radius-lg) !important;
    padding: 1rem 1.25rem !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
    color: var(--text-primary) !important;
    max-width: 82% !important;
    margin-left: auto !important;
}

/* Avatar styling */
[data-testid="chatAvatarIcon-assistant"],
[data-testid="chatAvatarIcon-user"] {
    background: var(--obsidian-3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 50% !important;
    color: var(--crimson-light) !important;
    font-size: 0.7rem !important;
}

/* Markdown inside chat */
[data-testid="stChatMessage"] .stMarkdown strong {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}
[data-testid="stChatMessage"] .stMarkdown em {
    color: var(--text-secondary) !important;
    font-style: italic !important;
}
[data-testid="stChatMessage"] .stMarkdown code {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    background: var(--obsidian-4) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.1em 0.35em !important;
    color: var(--crimson-light) !important;
}

/* ── Chat Input ── */
[data-testid="stChatInputContainer"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.25rem 0.75rem !important;
    margin-top: 1.5rem !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stChatInputContainer"]:focus-within {
    border-color: var(--border-accent) !important;
    box-shadow: 0 0 0 3px var(--crimson-glow) !important;
}
[data-testid="stChatInputContainer"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
    border: none !important;
    outline: none !important;
    caret-color: var(--crimson-light) !important;
}
[data-testid="stChatInputContainer"] textarea::placeholder {
    color: var(--text-muted) !important;
}
[data-testid="stChatInputContainer"] button {
    background: var(--crimson) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: white !important;
    transition: background 0.2s ease !important;
}
[data-testid="stChatInputContainer"] button:hover {
    background: var(--crimson-light) !important;
}

/* ── Download Button ── */
[data-testid="stDownloadButton"] button {
    background: transparent !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: var(--radius-md) !important;
    color: var(--crimson-light) !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.2rem !important;
    width: 100% !important;
    transition: background 0.2s ease, border-color 0.2s ease !important;
    letter-spacing: 0.02em !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: var(--crimson-glow) !important;
    border-color: var(--crimson-light) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: var(--crimson-light) !important;
}
[data-testid="stSpinner"] > div {
    border-color: var(--crimson) transparent transparent transparent !important;
}

/* ── Info / Alert boxes ── */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Session end screen ── */
.ts-session-end {
    text-align: center;
    padding: 3rem 2rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    background: var(--surface);
    margin-top: 2rem;
}
.ts-session-end-title {
    font-family: var(--font-display);
    font-size: 1.6rem;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}
.ts-session-end-sub {
    font-size: 0.82rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
    letter-spacing: 0.06em;
    margin-bottom: 2rem;
}

/* ── Difficulty Tag ── */
.ts-diff-tag {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    margin-right: 4px;
}
.ts-diff-easy   { color: #4ADE80; background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.25); }
.ts-diff-medium { color: #FBBF24; background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.25); }
.ts-diff-hard   { color: var(--crimson-light); background: var(--crimson-glow); border: 1px solid var(--border-accent); }

/* ── Sentiment Badge ── */
.ts-sentiment {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.06em;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
}
.ts-sent-pos { color: #4ADE80; background: rgba(74,222,128,0.08); border: 1px solid rgba(74,222,128,0.2); }
.ts-sent-neu { color: #94A3B8; background: rgba(148,163,184,0.08); border: 1px solid rgba(148,163,184,0.2); }
.ts-sent-neg { color: var(--crimson-light); background: var(--crimson-glow); border: 1px solid var(--border-accent); }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Helper: render sidebar ─────────────────────────────────────────────────────
def render_sidebar(state: dict):
    with st.sidebar:
        # Wordmark
        st.markdown(
            """
        <div class="ts-wordmark">
            <span class="ts-wordmark-logo">TalentScout</span>
            <span class="ts-wordmark-dot"></span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Phase badge
        phase = state.get("phase", "greeting")
        phase_labels = {
            "greeting": "INITIALISING",
            "info": "INFO GATHERING",
            "tech_questions": "TECHNICAL SCREEN",
            "closing": "CLOSING",
            "ended": "SESSION COMPLETE",
        }
        st.markdown(
            f"""
        <div class="ts-phase-badge">
            <span class="ts-phase-badge-dot"></span>
            <span class="ts-phase-badge-text">{phase_labels.get(phase, phase.upper())}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="ts-divider"></div>', unsafe_allow_html=True)

        # Candidate info fields
        st.markdown(
            '<div class="ts-sidebar-section">Candidate</div>', unsafe_allow_html=True
        )

        fields = [
            ("Name", state.get("full_name")),
            ("Email", "collected" if state.get("email") else None),
            ("Phone", "collected" if state.get("phone") else None),
            (
                "Experience",
                f"{state.get('years_experience')}y"
                if state.get("years_experience")
                else None,
            ),
            ("Role", state.get("desired_position")),
            ("Location", state.get("current_location")),
        ]
        for label, val in fields:
            dot = (
                '<span class="ts-field-dot-filled"></span>'
                if val
                else '<span class="ts-field-dot-empty"></span>'
            )
            value_html = (
                f'<span class="ts-field-value">{val}</span>'
                if val
                else '<span class="ts-field-pending">—</span>'
            )
            st.markdown(
                f"""
            <div class="ts-field-row">
                {dot}
                <span class="ts-field-label">{label}</span>
                {value_html}
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Tech stack
        if state.get("tech_stack"):
            st.markdown('<div class="ts-divider"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="ts-sidebar-section">Stack</div>', unsafe_allow_html=True
            )
            pills = "".join(
                f'<span class="ts-tech-pill">{t}</span>' for t in state["tech_stack"]
            )
            st.markdown(
                f'<div class="ts-tech-wrap">{pills}</div>', unsafe_allow_html=True
            )

        # Scores
        if state.get("tech_scores"):
            st.markdown('<div class="ts-divider"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="ts-sidebar-section">Assessment</div>',
                unsafe_allow_html=True,
            )
            for ts in state["tech_scores"]:
                pct = int(ts["score"] * 10)
                st.markdown(
                    f"""
                <div class="ts-score-row">
                    <div class="ts-score-label-row">
                        <span class="ts-score-tech">{ts["technology"]}</span>
                        <span class="ts-score-num">{ts["score"]:.1f}</span>
                    </div>
                    <div class="ts-score-track">
                        <div class="ts-score-fill" style="width:{pct}%"></div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Current difficulty
        if phase == "tech_questions":
            diff = state.get("current_difficulty", "Easy")
            cls = f"ts-diff-{diff.lower()}"
            st.markdown('<div class="ts-divider"></div>', unsafe_allow_html=True)
            st.markdown(
                f"""
            <div style="padding: 0 1.5rem 1rem;">
                <span class="ts-sidebar-section" style="display:block;padding-left:0;padding-bottom:0.4rem;">Difficulty</span>
                <span class="ts-diff-tag {cls}">{diff.upper()}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Footer
        st.markdown('<div class="ts-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            """
        <div style="padding: 0.5rem 1.5rem 1.5rem;">
            <span style="font-family: var(--font-mono); font-size: 0.6rem; color: var(--text-muted); letter-spacing:0.08em;">
                GDPR · PII REDACTED · SESSION-SCOPED
            </span>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ── Helper: format assistant message content ───────────────────────────────────
def format_assistant_content(content: str) -> str:
    """
    Replace [Easy/Medium/Hard] difficulty tags with styled HTML spans,
    and sentiment indicators with styled badges.
    """
    import re

    # Replace difficulty tags like **[Easy]**, **[Medium]**, **[Hard]**
    def replace_diff(m):
        level = m.group(1)
        cls = f"ts-diff-{level.lower()}"
        return f'<span class="ts-diff-tag {cls}">{level.upper()}</span>'

    content = re.sub(r"\*?\*?\[?(Easy|Medium|Hard)\]?\*?\*?", replace_diff, content)

    # Replace sentiment emojis with monochrome badges
    sentiment_map = {
        "🟢": '<span class="ts-sentiment ts-sent-pos">CONFIDENT</span>',
        "🟡": '<span class="ts-sentiment ts-sent-neu">NEUTRAL</span>',
        "🔴": '<span class="ts-sentiment ts-sent-neg">UNCERTAIN</span>',
    }
    for emoji, badge in sentiment_map.items():
        content = content.replace(emoji, badge)

    return content


# ── Session State Init ─────────────────────────────────────────────────────────
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "candidate_state" not in st.session_state:
    st.session_state.candidate_state: CandidateState = {
        "messages": [],
        "full_name": None,
        "email": None,
        "phone": None,
        "years_experience": None,
        "desired_position": None,
        "current_location": None,
        "tech_stack": [],
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
    result = st.session_state.graph.invoke(st.session_state.candidate_state)
    st.session_state.candidate_state.update(result)

state = st.session_state.candidate_state

# ── Render Sidebar ─────────────────────────────────────────────────────────────
render_sidebar(state)

# ── Main Column ───────────────────────────────────────────────────────────────
# Page header
st.markdown(
    """
<div class="ts-page-header">
    <h1 class="ts-page-title">Hiring <span>Screen</span></h1>
    <p class="ts-page-subtitle">TALENTSCOUT · AI-POWERED CANDIDATE SCREENING</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Session End Screen ─────────────────────────────────────────────────────────
if state.get("phase") == "ended":
    st.markdown(
        """
    <div class="ts-session-end">
        <div class="ts-session-end-title">Screening complete</div>
        <div class="ts-session-end-sub">REPORT READY FOR DOWNLOAD</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if state.get("tech_scores"):
        pdf_bytes = generate_report(state)
        candidate_name = (state.get("full_name") or "candidate").replace(" ", "_")
        st.download_button(
            label="Download Interview Report — PDF",
            data=pdf_bytes,
            file_name=f"talentscout_{candidate_name}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown(
        """
    <p style="text-align:center; margin-top:1.5rem; font-family:var(--font-mono);
       font-size:0.72rem; color:var(--text-muted); letter-spacing:0.06em;">
        Refresh the page to start a new session
    </p>
    """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Chat Messages ──────────────────────────────────────────────────────────────
for msg in state.get("messages", []):
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="T"):
            formatted = format_assistant_content(msg.content)
            st.markdown(formatted, unsafe_allow_html=True)
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="C"):
            st.markdown(msg.content)

# ── Chat Input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Your response...")

if user_input:
    current_state = st.session_state.candidate_state
    current_state["messages"].append(HumanMessage(content=user_input))

    with st.spinner(""):
        result = st.session_state.graph.invoke(current_state)
        current_state.update(result)

    st.session_state.candidate_state = current_state
    st.rerun()
