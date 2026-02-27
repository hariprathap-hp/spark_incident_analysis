import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from pathlib import Path
import uuid

load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "spark-incidents-openai"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536  # text-embedding-3-small size

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30
)

def get_embedding(text: str):
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def chunk_text(text: str, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest_incidents():
    print("🚀 Loading Spark incident reports...")

    # Create collection
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created")

    # Load txt files
    incident_dir = Path("spark_incidents")
    files = list(incident_dir.glob("*.txt"))
    print(f"📄 Found {len(files)} incident files")

    points = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for chunk in chunks:
            embedding = get_embedding(chunk)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "source": file.name
                    }
                )
            )
        print(f"✅ Processed: {file.name} ({len(chunks)} chunks)")

    # Upload to Qdrant
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"✅ Ingested {len(points)} chunks into Qdrant!")

if __name__ == "__main__":
    ingest_incidents()