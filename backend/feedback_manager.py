"""
Feedback Manager — Spark Insight Agent.

Stores thumbs-up/down ratings in PostgreSQL and exposes aggregate stats.
Table is created on first import if it does not exist.
"""

from __future__ import annotations

import logging
from datetime import datetime

import psycopg2

from backend.config import cfg

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id               SERIAL PRIMARY KEY,
    query            TEXT        NOT NULL,
    answer           TEXT        NOT NULL,
    rating           VARCHAR(20) NOT NULL CHECK (rating IN ('thumbs_up', 'thumbs_down')),
    timestamp        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    confidence_score FLOAT,
    path_taken       VARCHAR(50)
);
"""


class FeedbackManager:
    def _get_conn(self) -> psycopg2.extensions.connection:
        db = cfg.database
        return psycopg2.connect(
            host=db.host,
            port=db.port,
            dbname=db.name,
            user=db.user,
            password=db.password,
        )

    def ensure_table(self) -> None:
        """Create the feedback table if it doesn't exist."""
        try:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(_CREATE_TABLE_SQL)
                conn.commit()
                logger.info("Feedback table ready.")
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("Could not ensure feedback table: %s", exc)

    def save_feedback(
        self,
        query: str,
        answer: str,
        rating: str,
        confidence_score: float | None = None,
        path_taken: str | None = None,
    ) -> None:
        """Insert a feedback row. rating must be 'thumbs_up' or 'thumbs_down'."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feedback
                        (query, answer, rating, timestamp, confidence_score, path_taken)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (query, answer, rating, datetime.utcnow(), confidence_score, path_taken),
                )
            conn.commit()
            logger.info("Feedback saved: %s", rating)
        except Exception as exc:
            conn.rollback()
            logger.error("Failed to save feedback: %s", exc)
            raise
        finally:
            conn.close()

    def get_stats(self) -> dict:
        """Return {total, thumbs_up, thumbs_down, thumbs_up_pct}."""
        try:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            COUNT(*) AS total,
                            SUM(CASE WHEN rating = 'thumbs_up'   THEN 1 ELSE 0 END) AS thumbs_up,
                            SUM(CASE WHEN rating = 'thumbs_down' THEN 1 ELSE 0 END) AS thumbs_down
                        FROM feedback
                        """
                    )
                    row = cur.fetchone()
                total = row[0] or 0
                thumbs_up = row[1] or 0
                thumbs_down = row[2] or 0
                pct = round(thumbs_up / total * 100) if total > 0 else 0
                return {
                    "total": total,
                    "thumbs_up": thumbs_up,
                    "thumbs_down": thumbs_down,
                    "thumbs_up_pct": pct,
                }
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("Could not fetch feedback stats: %s", exc)
            return {"total": 0, "thumbs_up": 0, "thumbs_down": 0, "thumbs_up_pct": 0}


# Singleton — import this everywhere
feedback_manager = FeedbackManager()
feedback_manager.ensure_table()
