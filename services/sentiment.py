from textblob import TextBlob


def analyze_sentiment(text: str) -> dict:
    """
    Returns sentiment label and polarity score.
    Used to show candidate confidence badge in UI.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1.0 to 1.0

    if polarity > 0.1:
        label = "positive"
        emoji = "🟢"
    elif polarity < -0.1:
        label = "negative"
        emoji = "🔴"
    else:
        label = "neutral"
        emoji = "🟡"

    return {"label": label, "polarity": round(polarity, 2), "emoji": emoji}
