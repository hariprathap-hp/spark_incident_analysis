# 🔥 Spark Incident Analysis Assistant
**AI-powered assistant for analyzing Apache Spark incidents using LangChain, OpenAI, and Qdrant**

---

## 🎯 Purpose

This project demonstrates a **Retrieval-Augmented Generation (RAG)** system that helps engineers analyze **Spark incident reports** efficiently.  
Instead of reading through multiple logs or reports, users can query an assistant that retrieves the most relevant incidents and summarizes their root cause, resolutions, and key learnings.

---

## ⚙️ Implementation Overview

### 🔹 Architecture
User Query ─► LangChain Retrieval Chain ─► Qdrant (Vector DB)
│
▼
OpenAI (Embeddings + GPT-4)
│
▼
Contextual Root Cause Explanation


### 🔹 Components
| Component | Purpose |
|------------|----------|
| **LangChain** | Handles text chunking, retrieval chain, and prompt orchestration |
| **OpenAI** | Provides embeddings (`text-embedding-3-small`) and reasoning via `gpt-4-turbo` |
| **Qdrant** | Vector database storing Spark incident chunks |
| **Streamlit** | User interface for chat-based querying |

### 🔹 Dataset
The dataset consists of ~20 structured Spark incident reports (executor lost, shuffle failures, OOM errors, etc.) stored in `spark_incidents/`.

---

## 💰 Current Cost Factor

- **Embeddings**: `text-embedding-3-small` — ~$0.02 per 1,000,000 tokens  
- **LLM Queries**: `gpt-4-turbo` — ~$0.01 per 1,000 input tokens, ~$0.03 per 1,000 output tokens  
- A typical query costs **less than $0.005**  
- HuggingFace embeddings can be used as a **free fallback** (by updating `USE_OPENAI=False`)

---

## 🚀 Future Improvements

### 1️⃣ **MCP Integration**
- Integrate the [Kubeflow MCP for Spark History Server](https://github.com/kubeflow/mcp-apache-spark-history-server)
- Enables querying live Spark job metadata (tasks, stages, metrics)
- Converts this assistant into a **real-time Spark Reliability Copilot**

### 2️⃣ **Hybrid Search**
- Combine dense vector search (semantic similarity via OpenAI embeddings) with sparse lexical search (BM25 or keyword-based retrieval).

Benefits:
- Handles both natural language and technical keyword queries (e.g., “GC overhead limit exceeded” or “OOM executor”).
- Increases recall — retrieves relevant results even when exact phrasing differs.
- Implementation: Use Qdrant Hybrid Search or LangChain’s MultiVectorRetriever to merge embedding and keyword scores.

### 3️⃣ **Re-ranking**
- Apply a reranking model (e.g., bge-reranker-large or Cohere reranker) to re-score the top retrieved chunks.

Benefits:
- Increases precision by pushing the most relevant chunk to the top.
- Provides context clarity — especially when multiple incidents mention similar errors.

Implementation:
- Retrieve top 10 chunks from Qdrant
- Pass them through the reranker model
- Feed top 3–5 re-ranked results into GPT-4 for final answer synthesis.

### 4 Metadata Filtering

Introduce metadata filters such as:
- cluster_name, error_type, incident_date, severity

Benefits:
- Enables targeted search, e.g.:
- “Incidents from Cluster-B in March 2024”
- “Only OutOfMemoryError incidents”
- Reduces irrelevant retrievals and speeds up query time.

Implementation:
- Store metadata in Qdrant during ingestion
- Use search_kwargs={"filter": {"must": [{"key": "cluster", "match": {"value": "Cluster-B"}}]}}

4️⃣ Evaluation Metrics (Post-Improvement)
- Implement Precision@K, Recall@K, and MRR to quantify improvements from hybrid search and reranking.
- Use a labeled evaluation dataset (query → expected incident IDs) for benchmarking.

---

## 🧩 How to Run the App

1️⃣ Install dependencies
```bash
pip install -r requirements.txt

2️⃣ Set OpenAI key
Create a .env file:
   OPENAI_API_KEY=sk-your-api-key

3️⃣ Start Qdrant locally
docker run -p 6333:6333 qdrant/qdrant

4️⃣ Ingest incidents into Qdrant
python ingestion_qdrant_openai.py

5️⃣ Launch Streamlit app
streamlit run main.py

Then open http://localhost:8501

💬 Example Queries
What caused the executor lost issue in incident INC-2024-001?
Explain the resolution for shuffle fetch failures.
List incidents related to OutOfMemoryError in Spark drivers.
Summarize all incidents involving data skew.
What are the key learnings from memory-related failures?

🧠 Summary

This project showcases an end-to-end, locally deployable RAG assistant for analyzing Spark incidents.
It combines:

OpenAI reasoning (semantic understanding),

Qdrant retrieval (efficient similarity search), and

LangChain orchestration (retrieval + synthesis).

With MCP, hybrid retrieval, and reranking, this can evolve into a production-grade Spark Reliability Copilot capable of reasoning over both historical incidents and live Spark job data.