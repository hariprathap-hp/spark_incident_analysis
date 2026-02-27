# 🔥 Spark Insight Agent — Phase 2

**Production-grade AI assistant for Apache Spark incident analysis.**
Combines deterministic intelligence with LLM synthesis, multi-layer caching, and full cost tracking.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. QUERY CACHE  (exact-match, 1h TTL)                           │
│     hit → return instantly ($0 cost)                             │
└──────────────────────┬───────────────────────────────────────────┘
                       │ miss
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. EMBEDDING LAYER  (text-embedding-3-small)                    │
│     + Embedding Cache (SHA-256 content hash, 24h TTL)           │
└──────────────────────┬───────────────────────────────────────────┘
                       │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. QDRANT SEARCH  (top-10 by cosine similarity)                 │
│     Qdrant Cloud — collection: spark-incidents-openai            │
└──────────────────────┬───────────────────────────────────────────┘
                       │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  4. DETERMINISTIC ANALYZER                                       │
│     • Similarity scoring & spread (top-3 consistency)           │
│     • Root-cause cluster detection (12-category regex taxonomy) │
│     • Recurrence pattern flagging (≥2 occurrences)             │
│     • Confidence score: weighted A+B+C+D (0.0 – 1.0)           │
└──────────────────────┬───────────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
   confidence ≥ 0.70         confidence < 0.70
          │                         │
          ▼                         ▼
   Structured Markdown        LLM Cache check
   (no LLM, ~$0 cost)              │
                            ┌───────┴────────┐
                            │ hit            │ miss
                            ▼                ▼
                       cached answer    GPT-4o-mini call
                                        (30min LLM cache)
          │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  5. EVALUATOR  — per-query JSONL log                             │
│     tokens, cost, latency, confidence, path, cache hits         │
└──────────────────────────────────────────────────────────────────┘
          │
    ▼
Streamlit UI (chat + metrics panel + sidebar stats)
```

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `backend/config.py` | **Single source of truth** for all config values — models, thresholds, TTLs, cost rates. Load from `.env`. |
| `backend/cache_manager.py` | **3-layer cache**: query cache, embedding cache, LLM response cache. In-memory + pickle persistence. |
| `backend/deterministic_analyzer.py` | **Deterministic intelligence layer**: root-cause taxonomy, cluster detection, confidence scoring, answer generation without LLM. |
| `backend/evaluator.py` | **Metrics tracking**: cost, latency p95, path distribution, cache hit rates. Appends JSONL. |
| `backend/core_qdrant.py` | **Query pipeline**: wires all layers together. Phase 1 compatible return type. |
| `backend/data_pipeline.py` | **Ingestion**: PostgreSQL → format → embed → Qdrant. Per-incident error handling. |
| `main.py` | **Streamlit UI**: chat + metrics panel per response + sidebar session stats. |

---

## Cost Comparison

| | Phase 1 | Phase 2 |
|--|---------|---------|
| **LLM model** | `gpt-4-turbo` ($10/$30 per 1M) | `gpt-4o-mini` ($0.15/$0.60 per 1M) |
| **LLM calls** | Every query | Only when confidence < 0.70 |
| **Caching** | None | 3 layers (query + embedding + LLM) |
| **Typical query cost** | ~$0.005 | ~$0.00002 (deterministic) / ~$0.0002 (LLM) |
| **Cost reduction** | — | **~25–250×** depending on query mix |

---

## Confidence Score Components

```
A. Top similarity score        weight 0.40  — how close is the best match?
B. Score consistency (top-3)   weight 0.20  — are results tightly clustered?
C. Root-cause cohesion         weight 0.25  — do results agree on the cause?
D. Recurrence flag             weight 0.15  — is this a known repeat pattern?
```

When `A + B + C + D >= 0.70`, the deterministic layer returns a structured
answer with zero LLM cost. The LLM is only invoked for genuinely ambiguous
or novel queries.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env
cat > .env <<EOF
OPENAI_API_KEY=sk-...
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key
DB_USER=postgres
DB_PASSWORD=your-db-password
EOF

# 3. Ingest incidents from PostgreSQL
python -m backend.data_pipeline

# 4. Launch Streamlit
streamlit run main.py
```

Open http://localhost:8501

---

## Example Queries

```
What caused the executor lost issue in INC-2024-001?
Show me all memory-related incidents
What clusters are experiencing data skew?
Summarize recurring OOM errors
What are the key learnings from shuffle failures?
```

---

## Metrics Log

Every query appends a JSON line to `metrics.jsonl`:

```json
{
  "timestamp": "2024-03-15T10:23:01.234Z",
  "confidence": 0.82,
  "path": "deterministic",
  "model_used": null,
  "embedding_tokens": 0,
  "llm_input_tokens": 0,
  "total_cost_usd": 0.0,
  "latency_ms": 312.4,
  "top_similarity": 0.913,
  "clusters_detected": 1,
  "query_cache_hit": false,
  "embedding_cache_hit": true
}
```

---

## Phase Roadmap

- **Phase 1** ✅ — Basic RAG: PostgreSQL → Qdrant → GPT-4-turbo → Streamlit
- **Phase 2** ✅ — Deterministic intelligence + caching + cost tracking + evaluator
- **Phase 3** 🔜 — Hybrid search (BM25 + dense), reranking, MCP Spark History Server integration
