"""
SQLite session storage.
GDPR compliance:
  - Only Presidio-redacted data is stored — raw PII never touches the DB.
  - Sessions auto-delete after 24 hours.
  - Users can delete their session manually via the UI.
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from services.pii_redactor import redact_pii, redact_dict

DB_PATH = Path("db/sessions.db")
RETENTION_HOURS = 24

logger = logging.getLogger(__name__)


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Called once at app startup."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                phase        TEXT,
                candidate    TEXT,   -- JSON, all values Presidio-redacted
                scores       TEXT,   -- JSON array of ScoreEntry
                language     TEXT
            )
        """)
        conn.commit()
    _purge_expired()


def save_session(state: dict):
    """
    Persist session to DB. All candidate PII fields are redacted before storage.
    Only non-sensitive metadata and anonymized text is written.
    """
    try:
        # Extract and redact candidate profile fields
        candidate = redact_dict(
            {
                "full_name": state.get("full_name", ""),
                "email": state.get("email", ""),
                "phone": state.get("phone", ""),
                "current_location": state.get("current_location", ""),
                "years_of_experience": state.get("years_of_experience", ""),
                "desired_positions": state.get("desired_positions", ""),
                "tech_stack": json.dumps(state.get("tech_stack") or []),
            }
        )

        # Redact answers in score entries
        raw_scores = state.get("answer_scores", [])
        clean_scores = []
        for s in raw_scores:
            clean_scores.append({**s, "answer": redact_pii(s.get("answer", ""))})

        now = datetime.utcnow().isoformat()
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, created_at, updated_at, phase, candidate, scores, language)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    phase      = excluded.phase,
                    candidate  = excluded.candidate,
                    scores     = excluded.scores,
                    language   = excluded.language
            """,
                (
                    state.get("session_id", "unknown"),
                    now,
                    now,
                    state.get("current_phase", ""),
                    json.dumps(candidate),
                    json.dumps(clean_scores),
                    state.get("language", "English"),
                ),
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"DB save failed: {e}")


def delete_session(session_id: str) -> bool:
    """Allow candidate to delete their data on request (GDPR right to erasure)."""
    try:
        with _get_conn() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        return True
    except Exception:
        return False


def _purge_expired():
    """Auto-delete sessions older than RETENTION_HOURS (runs at startup)."""
    try:
        cutoff = (datetime.utcnow() - timedelta(hours=RETENTION_HOURS)).isoformat()
        with _get_conn() as conn:
            conn.execute("DELETE FROM sessions WHERE created_at < ?", (cutoff,))
            conn.commit()
    except Exception as e:
        logger.warning(f"Purge failed: {e}")
