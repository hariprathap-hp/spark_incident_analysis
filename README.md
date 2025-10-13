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
- Combine **dense (embeddings)** + **sparse (keyword/BM25)** retrieval
- Improves recall for queries mixing technical terms + natural language

### 3️⃣ **Re-ranking**
- Use **cross-encoder reranking** (e.g., `bge-reranker-large`) on top of Qdrant results
- Boosts precision by scoring top-10 retrieved chunks before sending to GPT

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