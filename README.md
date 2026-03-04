# Spark Insight Agent — Phase 2

**Production-grade AI assistant for Apache Spark incident analysis.**
Combines deterministic intelligence with LLM synthesis, multi-layer caching, and full cost tracking.

---

## Problem Statement

Apache Spark incidents — executor failures, OOM errors, shuffle timeouts, data skew — are complex and often recur across clusters. Without institutional memory, engineers spend **2–6 hours per incident** re-investigating root causes that have been solved before.

**Spark Insight Agent** turns your historical incident database into a queryable knowledge base. Engineers get instant, structured answers drawn from past incidents — with citations, root-cause taxonomy, and recurrence flags — in seconds rather than hours.

---

## Results (Phase 2 Testing)

Real measurements from 4 representative test queries against a live Qdrant Cloud collection:

| Metric | Value |
|--------|-------|
| Total queries | 4 |
| LLM calls avoided (deterministic path) | **75%** |
| Total cost for all 4 queries | **$0.0002** |
| Deterministic path latency | **~0 ms LLM overhead** |
| LLM path cost per query | ~$0.0002 |
| Deterministic path cost per query | ~$0.00000 |

75% of queries were answered entirely by the deterministic analyzer — no GPT call, no latency, no cost. Only one genuinely ambiguous query reached the LLM.

---

## Key Features

### 3-Path Routing

Every query is routed through one of three paths based on data signals:

| Path | Trigger | Cost | Latency |
|------|---------|------|---------|
| **Query cache hit** | Exact match seen before (1h TTL) | $0 | <1 ms |
| **Deterministic** | Confidence score ≥ 0.70 | ~$0 | No LLM wait |
| **LLM (GPT-4o-mini)** | Confidence < 0.70, novel query | ~$0.0002 | +API latency |

### Deterministic Intelligence
A four-factor confidence score determines whether the LLM is needed at all:
- **Similarity strength** (40%) — how well does the top result match?
- **Score consistency** (20%) — are the top-3 results in agreement?
- **Root-cause cohesion** (25%) — do results cluster around one taxonomy category?
- **Recurrence flag** (15%) — is this a known repeat pattern?

### 3-Layer Cache
Query cache → Embedding cache → LLM response cache. Each layer has its own TTL and is pickle-persisted across restarts.

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

# 3. (Optional) Insert test incidents into PostgreSQL
#    Use the sample SQL scripts in /data or connect your own incidents table.

# 4. Ingest incidents from PostgreSQL into Qdrant
python -m backend.data_pipeline

# 5. Launch Streamlit
streamlit run main.py
```

Open http://localhost:8501

---

## Production Metrics

When deployed against a real incident history (hundreds to thousands of incidents):

| Metric | Expected Range |
|--------|---------------|
| MTTR reduction | 60–80% for recurring incident types |
| LLM calls avoided | 60–80% of queries (deterministic path) |
| Cost per query (mixed traffic) | $0.00002 – $0.0002 |
| Query response time (deterministic) | 300–600 ms (embedding + Qdrant, no LLM) |
| Query response time (LLM path) | 2–5 s (adds GPT-4o-mini latency) |
| Embedding cache hit rate | ~90% after warm-up |

The MTTR reduction comes from engineers getting structured, cited answers in seconds instead of searching Confluence, Slack, and JIRA manually.

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
| `backend/feedback_manager.py` | **Feedback loop**: thumbs up/down per response, persisted to PostgreSQL `feedback` table. |
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

## What I Learned

Building this system surfaced several non-obvious lessons:

1. **Deterministic beats LLM for structured data.** When incidents are tagged with consistent metadata (cluster, component, root cause), a confidence-scored retrieval layer handles the majority of queries faster and cheaper than any LLM.

2. **Caching at every layer compounds.** Query, embedding, and LLM caches each individually give modest gains; together they make repeated and near-repeated queries essentially free.

3. **Cost visibility changes behavior.** Displaying per-query cost in the UI (even fractions of a cent) makes engineers immediately aware of what's expensive and why. It also validates that the deterministic path is working.

4. **The LLM is best used for synthesis, not retrieval.** Retrieval is a solved problem with vector search. The LLM's value is in synthesizing multi-incident patterns and generating actionable recommendations — tasks the deterministic layer cannot do well.

5. **Feedback loops close the loop.** Thumbs up/down per response surfaces answer quality problems that metrics alone miss. Low-confidence queries with negative feedback are the highest-priority items for training data improvement.

---

## Deployment

The app is designed to run locally or on any Python-capable server. No special infrastructure beyond Qdrant Cloud and a PostgreSQL instance is required.

```
Infrastructure:
  Qdrant Cloud   — managed vector DB (free tier sufficient for <1M vectors)
  PostgreSQL     — source of truth for raw incidents + feedback storage
  OpenAI API     — embeddings + optional LLM synthesis
  Streamlit      — UI server (can be hosted on Streamlit Community Cloud)
```

For production use, run Streamlit behind a reverse proxy (nginx/Caddy) with authentication. The `metrics.jsonl` file can be shipped to any log aggregator (Datadog, Grafana Loki, CloudWatch) for dashboarding.

---

## Phase Roadmap

- **Phase 1** ✅ — Basic RAG: PostgreSQL → Qdrant → GPT-4-turbo → Streamlit
- **Phase 2** ✅ — Deterministic intelligence + caching + cost tracking + evaluator + feedback
- **Phase 3** — Hybrid search (BM25 + dense), reranking, MCP Spark History Server integration
