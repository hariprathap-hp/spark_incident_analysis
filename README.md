# Spark Insight Agent

**An AI assistant that learns from your Apache Spark incident history and helps engineers fix problems faster.**

---

## Problem Statement

When Apache Spark jobs fail — out-of-memory errors, slow queries, cluster crashes — engineers typically spend hours digging through old tickets, Slack threads, and wikis to find a fix. Most of the time, someone has already solved the same problem before.

**Spark Insight Agent** makes that institutional knowledge instantly searchable. Instead of starting from scratch, engineers ask a question and get a structured answer drawn from real past incidents — in seconds.

---

## Solution Overview

The agent stores your incident history in a searchable database and matches new problems against past ones. For common, recurring issues it answers instantly using pattern matching — no AI model needed, no waiting, near-zero cost. For unusual or complex queries, it calls an AI model to synthesize an answer from multiple related incidents.

The result: faster answers, lower costs, and a system that gets more useful as your incident history grows.

---

## Key Features

- **Instant answers for known problems** — recurring issues are answered in milliseconds without calling an AI model
- **AI synthesis for novel queries** — complex or ambiguous questions are answered by GPT-4o-mini, drawing on multiple past incidents
- **Cost-aware by design** — most queries cost a fraction of a cent; the UI shows per-query cost so nothing is hidden
- **Cites its sources** — every answer includes the incident IDs it was drawn from
- **Thumbs up / down feedback** — engineers can flag unhelpful answers, surfacing gaps in the incident history
- **Session metrics panel** — see response time, cost, and how each answer was generated

---

## How It Works

```
1. You ask a question
   e.g. "Why did the ETL job fail with OOM on cluster-prod-3?"

2. The agent searches past incidents
   It finds the most relevant incidents from your history
   and checks whether the answer is clear-cut or needs AI synthesis

3. You get a structured answer
   — For clear patterns: instant answer with citations, no AI call
   — For complex queries: AI-generated summary of related incidents
```

---

## Architecture

<details>
<summary>Click to expand full architecture diagram</summary>

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. QUERY CACHE  (exact-match, 1-hour window)                    │
│     hit → return instantly ($0 cost)                             │
└──────────────────────┬───────────────────────────────────────────┘
                       │ miss
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. EMBEDDING LAYER  (text-embedding-3-small)                    │
│     + Embedding Cache (content hash, 24-hour window)            │
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
│  4. PATTERN MATCHING LAYER                                       │
│     Confidence score (0.0 – 1.0) based on:                      │
│     • Similarity strength (40%) — how well does the top match?  │
│     • Score consistency (20%) — do the top-3 results agree?     │
│     • Root-cause cohesion (25%) — do results share a cause?     │
│     • Recurrence flag (15%) — is this a known repeat pattern?   │
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
                                        (30-min cache)
          │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  5. METRICS LOGGER — per-query log                               │
│     tokens, cost, latency, confidence, path, cache hits         │
└──────────────────────────────────────────────────────────────────┘
          │
    ▼
Streamlit UI (chat + metrics panel + sidebar stats)
```

**Module reference:**

| Module | Purpose |
|--------|---------|
| `backend/config.py` | All config values — models, thresholds, cost rates. Loaded from `.env`. |
| `backend/cache_manager.py` | 3-layer cache: query, embedding, and LLM response. In-memory + disk persistence. |
| `backend/deterministic_analyzer.py` | Pattern matching layer: cluster detection, confidence scoring, answer generation. |
| `backend/evaluator.py` | Metrics tracking: cost, latency, path distribution, cache hit rates. |
| `backend/core_qdrant.py` | Query pipeline: wires all layers together. |
| `backend/data_pipeline.py` | Ingestion: PostgreSQL → embed → Qdrant. Per-incident error handling. |
| `backend/feedback_manager.py` | Feedback loop: thumbs up/down per response, stored to PostgreSQL. |
| `main.py` | Streamlit UI: chat + metrics panel + sidebar session stats. |

</details>

---

## Results & Metrics

Real measurements from 4 representative test queries against a live Qdrant Cloud collection:

| Metric | Value |
|--------|-------|
| Total queries tested | 4 |
| Queries answered without AI | **75%** |
| Total cost for all 4 queries | **$0.0002** |
| Pattern-matched query response time | **< 1 ms LLM overhead** |
| AI-assisted query cost | ~$0.0002 per query |
| Pattern-matched query cost | ~$0.00 per query |

3 out of 4 queries were answered entirely by pattern matching — no AI call, no latency, no cost. Only one genuinely novel query reached the model.

**At scale (hundreds to thousands of incidents):**

| Metric | Expected Range |
|--------|---------------|
| Time-to-resolution reduction | 60–80% for recurring incident types |
| Queries answered without AI | 60–80% |
| Cost per query (mixed traffic) | $0.00002 – $0.0002 |
| Response time (pattern match) | 300–600 ms |
| Response time (AI path) | 2–5 s |

**Cost comparison vs. Phase 1:**

| | Phase 1 | Phase 2 |
|--|---------|---------|
| **AI model** | `gpt-4-turbo` ($10/$30 per 1M tokens) | `gpt-4o-mini` ($0.15/$0.60 per 1M tokens) |
| **AI calls** | Every query | Only for ambiguous queries (~25%) |
| **Caching** | None | 3 layers (query + embedding + AI response) |
| **Typical query cost** | ~$0.005 | ~$0.00002 – $0.0002 |
| **Cost reduction** | — | **~25–250×** |

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

# 5. Launch the app
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

## What I Learned

1. **Pattern matching beats AI for structured data.** When incidents are tagged with consistent metadata, a confidence-scored retrieval layer handles the majority of queries faster and cheaper than any AI model.

2. **Caching at every layer compounds.** Query, embedding, and AI response caches each give modest gains individually; together they make repeated and near-repeated queries essentially free.

3. **Cost visibility changes behavior.** Showing per-query cost in the UI — even fractions of a cent — makes engineers aware of what's expensive and validates that the fast path is working.

4. **AI is best used for synthesis, not retrieval.** Vector search handles retrieval. The AI's value is in synthesizing multi-incident patterns and generating actionable recommendations — tasks pattern matching cannot do well.

5. **Feedback loops close the loop.** Thumbs up/down per response surfaces answer quality problems that metrics alone miss. Low-confidence queries with negative feedback are the highest-priority items for improvement.

---

## Deployment

The app runs locally or on any Python-capable server. No special infrastructure beyond Qdrant Cloud and a PostgreSQL instance is required.

```
Infrastructure:
  Qdrant Cloud   — managed vector database (free tier covers < 1M vectors)
  PostgreSQL     — source of truth for raw incidents + feedback storage
  OpenAI API     — embeddings + optional AI synthesis
  Streamlit      — UI server (hostable on Streamlit Community Cloud)
```

For production, run Streamlit behind a reverse proxy (nginx or Caddy) with authentication. Query metrics are written to a log file that can be forwarded to any log aggregator (Datadog, Grafana, CloudWatch) for dashboarding.

---

## Roadmap

- **Phase 1** ✅ — Basic RAG: PostgreSQL → Qdrant → GPT-4-turbo → Streamlit
- **Phase 2** ✅ — Pattern matching + caching + cost tracking + evaluator + feedback
- **Phase 3** — Hybrid search (keyword + semantic), reranking, Spark History Server integration
