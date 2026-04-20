from fpdf import FPDF
import matplotlib.pyplot as plt
import io, os, tempfile
from datetime import datetime


def generate_report(state: dict) -> bytes:
    """Generate a PDF talent report. Returns raw PDF bytes."""

    # --- 1. Create bar chart ---
    tech_names = [ts["technology"] for ts in state["tech_scores"]]
    scores = [ts["score"] for ts in state["tech_scores"]]

    fig, ax = plt.subplots(figsize=(6, max(2, len(tech_names) * 0.6)))
    bars = ax.barh(tech_names, scores, color="#4F8EF7")
    ax.set_xlim(0, 10)
    ax.set_xlabel("Score (0-10)")
    ax.set_title("Technical Proficiency by Technology")
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

    # --- 2. Build PDF ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "TalentScout — Candidate Report", ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        6,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ln=True,
        align="C",
    )
    pdf.ln(6)

    # Candidate Info
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Candidate Information", ln=True)
    pdf.set_font("Helvetica", "", 11)
    info_fields = [
        ("Name", state.get("full_name", "N/A")),
        ("Position", state.get("desired_position", "N/A")),
        ("Experience", f"{state.get('years_experience', 'N/A')} years"),
        ("Location", state.get("current_location", "N/A")),
        ("Tech Stack", ", ".join(state.get("tech_stack", []))),
    ]
    # Note: email/phone intentionally omitted from PDF (PII)
    for label, value in info_fields:
        pdf.cell(50, 7, f"{label}:", border=0)
        pdf.cell(0, 7, str(value), ln=True)

    pdf.ln(4)

    # Bar chart
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Technical Assessment", ln=True)
    pdf.image(chart_path, w=170)
    pdf.ln(4)

    # Per-tech breakdown
    for ts in state["tech_scores"]:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(
            0,
            7,
            f"{ts['technology']}  —  Score: {ts['score']:.1f}/10  (Reached: {ts['difficulty_reached']})",
            ln=True,
        )

    pdf.ln(6)

    # Sentiment summary
    sentiments = state.get("sentiment_history", [])
    if sentiments:
        pos = sentiments.count("positive")
        neg = sentiments.count("negative")
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Behavioural Signals", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(
            0,
            7,
            f"Positive responses: {pos}/{len(sentiments)}. "
            f"Negative signals: {neg}/{len(sentiments)}. "
            f"Overall tone: {'Confident' if pos > neg else 'Hesitant'}.",
        )

    os.unlink(chart_path)
    return bytes(pdf.output())
