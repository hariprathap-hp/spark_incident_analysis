import os
import re
import hashlib
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI

load_dotenv()

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "spark_incidents",
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

COLLECTION_NAME = "spark-incidents-openai"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30
)

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def ensure_collection_exists():
    """Create Qdrant collection if not exists"""
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created")
    else:
        print(f"✅ Collection '{COLLECTION_NAME}' already exists")

def get_embedding(text: str):
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def incident_id_to_uuid(incident_id: str) -> str:
    """Convert incident_id to consistent UUID for Qdrant"""
    return str(hashlib.md5(incident_id.encode()).hexdigest())

def clean_text(text: str) -> str:
    text = re.sub(r'\w+\s+\d{1,2}:\d{2}[apm]+\s+-\s*', '', text)
    text = re.sub(r'\d{1,2}:\d{2}[apm]+\s+-\s*', '', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def extract_root_cause(work_notes: str) -> str:
    patterns = [
        r"root cause[:\s]+([^\n]+)",
        r"cause[:\s]+([^\n]+)",
        r"found\s+([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, work_notes.lower())
        if match:
            return match.group(1).strip().capitalize()
    return "See work notes for details"

def extract_resolution(work_notes: str) -> str:
    patterns = [
        r"fix[ed]*[:\s]+([^\n]+)",
        r"resolv[ed]*[:\s]+([^\n]+)",
        r"replaced\s+([^\n]+)",
        r"increased\s+([^\n]+)",
        r"switched\s+([^\n]+)",
        r"implemented\s+([^\n]+)",
    ]
    resolutions = []
    for pattern in patterns:
        matches = re.findall(pattern, work_notes.lower())
        resolutions.extend(matches)
    if resolutions:
        return "\n  - ".join([r.strip().capitalize() for r in resolutions[:3]])
    return "See work notes for details"

def extract_key_learnings(work_notes: str) -> str:
    patterns = [
        r"lesson[:\s]+([^\n]+)",
        r"note[:\s]+([^\n]+)",
        r"never\s+([^\n]+)",
        r"always\s+([^\n]+)",
    ]
    learnings = []
    for pattern in patterns:
        matches = re.findall(pattern, work_notes.lower())
        learnings.extend(matches)
    if learnings:
        return learnings[0].strip().capitalize()
    return "Review work notes for learnings"

def format_incident(row: dict) -> str:
    """Format DB row into structured incident text (in memory)"""
    work_notes = row.get('work_notes', '') or ''
    cleaned_notes = clean_text(work_notes)
    root_cause = extract_root_cause(work_notes)
    resolution = extract_resolution(work_notes)
    key_learnings = extract_key_learnings(work_notes)

    return f"""Incident ID: {row['incident_id']}
Date: {row['created_date'].strftime('%Y-%m-%d') if row['created_date'] else 'Unknown'}
Cluster: {row.get('cluster', 'Unknown')}
Application: {row.get('application', 'Unknown')}
Error Summary: {row.get('description', 'No description')}
Root Cause: {root_cause}
Resolution:
  - {resolution}
Key Learnings: {key_learnings}
Work Notes:
{cleaned_notes}
"""

def mark_as_ingested(conn, incident_id: str):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE incidents 
            SET ingested = TRUE, ingested_at = %s 
            WHERE incident_id = %s
        """, (datetime.utcnow(), incident_id))
    conn.commit()

def ingest_to_qdrant(incident_id: str, formatted_text: str, row: dict):
    """Directly upsert incident to Qdrant (no text files)"""
    embedding = get_embedding(formatted_text)
    point_id = incident_id_to_uuid(incident_id)

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": formatted_text,
                    "incident_id": incident_id,
                    "cluster": row.get('cluster'),
                    "application": row.get('application'),
                    "status": row.get('status'),
                    "created_date": str(row.get('created_date'))
                }
            )
        ]
    )

def run_pipeline():
    """Main pipeline: fetch new → format → ingest → mark done"""
    print("🚀 Starting incident pipeline...")
    ensure_collection_exists()

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT incident_id, created_date, cluster, application,
                       description, work_notes, status
                FROM incidents
                WHERE ingested = FALSE
                AND status = 'RESOLVED'
                ORDER BY created_date ASC
            """)
            columns = [desc[0] for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]

        print(f"📄 Found {len(rows)} new incidents")

        for row in rows:
            incident_id = row['incident_id']
            formatted = format_incident(row)
            ingest_to_qdrant(incident_id, formatted, row)
            mark_as_ingested(conn, incident_id)
            print(f"✅ Ingested: {incident_id}")

        print(f"🎉 Pipeline complete! {len(rows)} incidents ingested.")

    finally:
        conn.close()

if __name__ == "__main__":
    run_pipeline()