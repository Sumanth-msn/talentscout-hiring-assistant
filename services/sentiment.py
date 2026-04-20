"""
Lightweight sentiment analysis on candidate answers.
Uses TextBlob (no API calls, runs locally).
Returns a label and score for display in the sidebar.
"""

from textblob import TextBlob


def analyze_sentiment(text: str) -> dict:
    """
    Returns:
        label: "Confident" | "Neutral" | "Uncertain"
        polarity: float -1.0 to 1.0
        subjectivity: float 0.0 to 1.0
        emoji: str  (for UI display)
    """
    if not text or len(text.strip()) < 5:
        return {"label": "Neutral", "polarity": 0.0, "subjectivity": 0.0, "emoji": "😐"}

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Map polarity to hiring-context labels
    if polarity >= 0.15:
        label, emoji = "Confident", "🟢"
    elif polarity <= -0.15:
        label, emoji = "Uncertain", "🔴"
    else:
        label, emoji = "Neutral", "🟡"

    return {
        "label": label,
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "emoji": emoji,
    }
