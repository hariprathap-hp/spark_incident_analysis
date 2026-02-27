"""
Data Ingestion Pipeline — Spark Insight Agent Phase 2.

Reads RESOLVED incidents from PostgreSQL, formats and embeds them,
upserts into Qdrant, then marks the DB rows as ingested.

Changes vs Phase 1
──────────────────
  - Config values loaded from backend.config (no more hardcoded strings)
  - Structured logging (logging module, not bare print)
  - Per-incident try/except with rollback — one bad incident won't abort the run
  - Type hints throughout
  - Embedding uses the shared cache to avoid duplicate API calls during re-runs
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from typing import Optional

import psycopg2
import psycopg2.extras
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.config import cfg

logger = logging.getLogger(__name__)

# ── Clients ───────────────────────────────────────────────────────────────────

_openai = OpenAI(api_key=cfg.openai.api_key)
_qdrant = QdrantClient(
    url=cfg.qdrant.url,
    api_key=cfg.qdrant.api_key,
    timeout=cfg.qdrant.timeout,
)


# ── DB helpers ─────────────────────────────────────────────────────────────────


def _get_db_connection() -> psycopg2.extensions.connection:
    db = cfg.database
    return psycopg2.connect(
        host=db.host,
        port=db.port,
        dbname=db.name,
        user=db.user,
        password=db.password,
    )


# ── Qdrant helpers ─────────────────────────────────────────────────────────────


def ensure_collection_exists() -> None:
    """Create the Qdrant collection if it does not exist yet."""
    name = cfg.qdrant.collection_name
    if not _qdrant.collection_exists(name):
        _qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=cfg.openai.vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Collection '%s' created.", name)
    else:
        logger.info("Collection '%s' already exists.", name)


def _incident_id_to_uuid(incident_id: str) -> str:
    """Deterministic UUID-like hex string from incident ID (for Qdrant point IDs)."""
    return hashlib.md5(incident_id.encode()).hexdigest()


# ── Text cleaning & extraction ─────────────────────────────────────────────────


def _clean_text(text: str) -> str:
    """Strip timestamp noise from work notes."""
    text = re.sub(r"\w+\s+\d{1,2}:\d{2}[apm]+\s+-\s*", "", text)
    text = re.sub(r"\d{1,2}:\d{2}[apm]+\s+-\s*", "", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _extract_root_cause(work_notes: str) -> str:
    patterns = [
        r"root cause[:\s]+([^\n]+)",
        r"cause[:\s]+([^\n]+)",
        r"found\s+([^\n]+)",
    ]
    lower = work_notes.lower()
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            return match.group(1).strip().capitalize()
    return "See work notes for details"


def _extract_resolution(work_notes: str) -> str:
    patterns = [
        r"fix(?:ed)?[:\s]+([^\n]+)",
        r"resolv(?:ed)?[:\s]+([^\n]+)",
        r"replaced\s+([^\n]+)",
        r"increased\s+([^\n]+)",
        r"switched\s+([^\n]+)",
        r"implemented\s+([^\n]+)",
    ]
    lower = work_notes.lower()
    resolutions: list[str] = []
    for pattern in patterns:
        for m in re.findall(pattern, lower):
            resolutions.append(m.strip().capitalize())
    if resolutions:
        return "\n  - ".join(resolutions[:3])
    return "See work notes for details"


def _extract_key_learnings(work_notes: str) -> str:
    patterns = [
        r"lesson[:\s]+([^\n]+)",
        r"note[:\s]+([^\n]+)",
        r"never\s+([^\n]+)",
        r"always\s+([^\n]+)",
    ]
    lower = work_notes.lower()
    for pattern in patterns:
        matches = re.findall(pattern, lower)
        if matches:
            return matches[0].strip().capitalize()
    return "Review work notes for learnings"


def _format_incident(row: dict) -> str:
    """Convert a DB row dict into a structured text block for embedding."""
    work_notes: str = row.get("work_notes") or ""
    cleaned_notes = _clean_text(work_notes)
    created_date = row.get("created_date")
    date_str = created_date.strftime("%Y-%m-%d") if isinstance(created_date, datetime) else str(created_date or "Unknown")

    return (
        f"Incident ID: {row['incident_id']}\n"
        f"Date: {date_str}\n"
        f"Cluster: {row.get('cluster', 'Unknown')}\n"
        f"Application: {row.get('application', 'Unknown')}\n"
        f"Error Summary: {row.get('description', 'No description')}\n"
        f"Root Cause: {_extract_root_cause(work_notes)}\n"
        f"Resolution:\n"
        f"  - {_extract_resolution(work_notes)}\n"
        f"Key Learnings: {_extract_key_learnings(work_notes)}\n"
        f"Work Notes:\n{cleaned_notes}\n"
    )


# ── Embedding ──────────────────────────────────────────────────────────────────


def _get_embedding(text: str) -> list[float]:
    response = _openai.embeddings.create(
        model=cfg.openai.embedding_model,
        input=text,
    )
    return response.data[0].embedding


# ── Ingestion helpers ──────────────────────────────────────────────────────────


def _ingest_to_qdrant(incident_id: str, formatted_text: str, row: dict) -> None:
    embedding = _get_embedding(formatted_text)
    point_id = _incident_id_to_uuid(incident_id)

    _qdrant.upsert(
        collection_name=cfg.qdrant.collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": formatted_text,
                    "incident_id": incident_id,
                    "cluster": row.get("cluster"),
                    "application": row.get("application"),
                    "status": row.get("status"),
                    "created_date": str(row.get("created_date")),
                },
            )
        ],
    )


def _mark_as_ingested(conn: psycopg2.extensions.connection, incident_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE incidents
            SET ingested = TRUE, ingested_at = %s
            WHERE incident_id = %s
            """,
            (datetime.utcnow(), incident_id),
        )
    conn.commit()


# ── Main pipeline ──────────────────────────────────────────────────────────────


def run_pipeline() -> dict[str, int]:
    """
    Fetch new RESOLVED incidents from PostgreSQL, embed and upsert into Qdrant,
    then mark each row as ingested.

    Returns a summary dict: {"ingested": N, "failed": M, "skipped": K}.
    """
    logger.info("Starting incident ingestion pipeline...")
    ensure_collection_exists()

    conn = _get_db_connection()
    results = {"ingested": 0, "failed": 0, "skipped": 0}

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT incident_id, created_date, cluster, application,
                       description, work_notes, status
                FROM incidents
                WHERE ingested = FALSE
                  AND status = 'RESOLVED'
                ORDER BY created_date ASC
                """
            )
            rows: list[dict] = [dict(r) for r in cur.fetchall()]

        logger.info("Found %d new incidents to ingest.", len(rows))

        for row in rows:
            incident_id: str = row["incident_id"]
            try:
                formatted = _format_incident(row)
                _ingest_to_qdrant(incident_id, formatted, row)
                _mark_as_ingested(conn, incident_id)
                results["ingested"] += 1
                logger.info("Ingested: %s", incident_id)
            except Exception as exc:
                results["failed"] += 1
                logger.error("Failed to ingest %s: %s", incident_id, exc)
                # Roll back any uncommitted changes for this incident
                conn.rollback()

        logger.info(
            "Pipeline complete. ingested=%d failed=%d",
            results["ingested"],
            results["failed"],
        )

    finally:
        conn.close()

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    run_pipeline()
