"""
services/report_generator.py — Generates a PDF talent report for the recruiter.
"""

import os
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt
from fpdf import FPDF


def generate_report(state: dict) -> bytes:
    """
    Build a PDF report summarising the candidate's screening session.
    Returns raw PDF bytes for Streamlit's st.download_button.
    """
    tech_scores = state.get("tech_scores", [])
    tech_names = [ts["technology"] for ts in tech_scores]
    scores = [ts["score"] for ts in tech_scores]

    # ── Bar chart ──────────────────────────────────────────────────────────────
    chart_path = None
    if tech_names:
        fig, ax = plt.subplots(figsize=(6, max(2, len(tech_names) * 0.7)))
        bars = ax.barh(tech_names, scores, color="#C0162A")
        ax.set_xlim(0, 10)
        ax.set_xlabel("Score (0-10)")
        ax.set_title("Technical Proficiency by Technology")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, score in zip(bars, scores):
            ax.text(
                score + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}",
                va="center",
                fontsize=9,
            )
        plt.tight_layout()
        chart_path = tempfile.mktemp(suffix=".png")
        plt.savefig(chart_path, dpi=120, bbox_inches="tight")
        plt.close()

    # ── PDF ────────────────────────────────────────────────────────────────────
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 14, "TalentScout - Candidate Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        6,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ln=True,
        align="C",
    )
    pdf.ln(8)

    # Candidate info
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 9, "Candidate Information", ln=True)
    pdf.set_font("Helvetica", "", 11)

    info_fields = [
        ("Name", state.get("full_name", "N/A")),
        ("Position", state.get("desired_position", "N/A")),
        ("Experience", f"{state.get('years_experience', 'N/A')} years"),
        ("Location", state.get("current_location", "N/A")),
        ("Tech Stack", ", ".join(state.get("tech_stack") or [])),
    ]
    # Email / phone intentionally omitted (PII)
    for label, value in info_fields:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(48, 7, f"{label}:", border=0)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, str(value), ln=True)

    pdf.ln(6)

    # Bar chart
    if chart_path:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 9, "Technical Assessment", ln=True)
        pdf.image(chart_path, w=170)
        pdf.ln(4)

        # Per-tech breakdown
        pdf.set_font("Helvetica", "B", 11)
        for ts in tech_scores:
            pdf.cell(
                0,
                7,
                f"{ts['technology']}  -  {ts['score']:.1f}/10  "
                f"(highest difficulty reached: {ts['difficulty_reached']})",
                ln=True,
            )

        os.unlink(chart_path)

    pdf.ln(6)

    # Sentiment summary
    sentiments = state.get("sentiment_history") or []
    if sentiments:
        pos = sentiments.count("positive")
        neg = sentiments.count("negative")
        neu = sentiments.count("neutral")
        total = len(sentiments)
        overall = "Confident" if pos > neg else ("Mixed" if pos == neg else "Hesitant")
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 9, "Behavioural Signals", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(
            0,
            7,
            f"Confident responses: {pos}/{total}.  "
            f"Neutral responses: {neu}/{total}.  "
            f"Uncertain responses: {neg}/{total}.  "
            f"Overall tone: {overall}.",
        )

    return bytes(pdf.output())
