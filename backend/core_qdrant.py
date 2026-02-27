from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI
import time
import os

load_dotenv()

COLLECTION_NAME = "spark-incidents-openai"
QDRANT_URL = "http://localhost:6333"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def search_incidents(query: str, k: int = 3):
    query_embedding = get_embedding(query)
    
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=k
    ).points
    return results

def run_llm(query: str):
    start = time.time()
    
    # Search similar incidents
    search_results = search_incidents(query)
    
    # Build context from results
    context = "\n\n".join([
        f"Incident: {r.payload.get('text', '')}"
        for r in search_results
    ])
    
    # Call OpenAI directly
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are a Spark incident analysis expert. 
                Use the provided incidents to answer questions accurately.
                If the answer isn't in the context, say so."""
            },
            {
                "role": "user",
                "content": f"""Context from similar incidents:
{context}

Question: {query}"""
            }
        ]
    )
    
    return {
        "answer": response.choices[0].message.content,
        "response_time": round(time.time() - start, 2),
        "source_documents": search_results
    }