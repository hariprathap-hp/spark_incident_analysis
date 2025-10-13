import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "spark-incidents-openai"

def ingest_incidents():
    print("🚀 Loading Spark incident reports...")

    loader = DirectoryLoader("spark_incidents", glob="*.txt", loader_cls=TextLoader)
    raw_documents = loader.load()
    print(f"📄 Loaded {len(raw_documents)} incident files")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_documents)
    print(f"🔹 Split into {len(documents)} chunks")

    print("🧠 Using OpenAI embeddings (text-embedding-3-small)")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    client = QdrantClient(url=QDRANT_URL)
    print(f"📦 Uploading to Qdrant collection: {COLLECTION_NAME}")

    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
    )

    print("✅ Successfully ingested incidents into Qdrant (OpenAI embeddings)!")

if __name__ == "__main__":
    ingest_incidents()
