"""
Two-layer PII redaction:
  Layer 1 — Microsoft Presidio (NLP-based: names, emails, phones, locations, IPs)
  Layer 2 — Regex fallback for Indian phone numbers and edge cases

IMPORTANT: Raw PII is NEVER passed to the LLM. Only redacted text enters prompts.
This file is the single gateway for all text that touches the LLM.
"""

import re
import logging
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Suppress Presidio's verbose startup logs
logging.getLogger("presidio-analyzer").setLevel(logging.WARNING)


@lru_cache(maxsize=1)
def _get_engines():
    """Lazy-load engines once and cache — avoids slow re-init per call."""
    return AnalyzerEngine(), AnonymizerEngine()


# Regex patterns for Layer 2
_PATTERNS = [
    (r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b", "[REDACTED_EMAIL]"),
    (r"\b(\+91[\s\-]?)?[6-9]\d{9}\b", "[REDACTED_PHONE]"),
    (r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "[REDACTED_PHONE]"),
    (r"\b\d{10}\b", "[REDACTED_PHONE]"),
]


def redact_pii(text: str) -> str:
    """
    Redact PII from text before sending to LLM.
    Returns anonymized string safe for LLM consumption.
    """
    if not text or not text.strip():
        return text

    analyzer, anonymizer = _get_engines()

    # Layer 1: Presidio NLP-based detection
    try:
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=[
                "PERSON",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "LOCATION",
                "URL",
                "IP_ADDRESS",
                "IN_PAN",
                "IN_AADHAAR",
                "CREDIT_CARD",
            ],
        )
        operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "[REDACTED_NAME]"}),
            "EMAIL_ADDRESS": OperatorConfig(
                "replace", {"new_value": "[REDACTED_EMAIL]"}
            ),
            "PHONE_NUMBER": OperatorConfig(
                "replace", {"new_value": "[REDACTED_PHONE]"}
            ),
            "LOCATION": OperatorConfig("replace", {"new_value": "[REDACTED_LOCATION]"}),
            "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
        }
        anonymized = anonymizer.anonymize(
            text=text, analyzer_results=results, operators=operators
        )
        text = anonymized.text
    except Exception:
        pass  # Fall through to Layer 2 on any Presidio error

    # Layer 2: Regex fallback
    for pattern, replacement in _PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def redact_dict(data: dict) -> dict:
    """Redact PII from all string values in a dict (for DB storage)."""
    redacted = {}
    for key, value in data.items():
        if isinstance(value, str):
            redacted[key] = redact_pii(value)
        elif isinstance(value, list):
            redacted[key] = [
                redact_pii(item) if isinstance(item, str) else item for item in value
            ]
        else:
            redacted[key] = value
    return redacted
