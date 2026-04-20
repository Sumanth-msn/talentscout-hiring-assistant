import re

# Patterns for common PII
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_AADHAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")


def redact_pii(text: str) -> str:
    """Remove PII before sending text to the LLM."""
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = _AADHAR_RE.sub("[REDACTED_ID]", text)
    return text


def redact_state_for_llm(state: dict) -> dict:
    """Return a copy of state safe to include in LLM prompts."""
    safe = state.copy()
    safe["email"] = "[REDACTED_EMAIL]" if state.get("email") else None
    safe["phone"] = "[REDACTED_PHONE]" if state.get("phone") else None
    return safe
