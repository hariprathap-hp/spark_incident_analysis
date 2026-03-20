# Spark Incident Analysis Agent — Presentation Q&A Guide

---

## Part 1: End-to-End Flow (Step by Step)

### Step 1 — User Enters a Query
The engineer types a Spark-related question into the Streamlit chat UI (e.g., *"Why is my Spark job failing with OOM errors?"*).

### Step 2 — Query Cache Lookup (Layer 1 — Exact Match)
- The query is normalized (`lowercase + strip`) and hashed with **SHA-256**.
- If an exact hash match exists in cache (TTL: **1 hour**) → **return immediately, cost = $0.00**.
- `last_best_similarity` is set to **1.0** (perfect match).

### Step 3 — Generate Query Embedding
- If no exact cache hit, the query is sent to **OpenAI text-embedding-3-small** (1536 dimensions).
- Before calling the API, the **embedding cache (Layer 2)** is checked (TTL: **24 hours**).
- If the embedding was generated before for the same text → reuse it, skip the API call.

### Step 4 — Semantic Cache Lookup (Layer 1 — Fuzzy Match)
- The new query embedding is compared against **all cached query embeddings** using cosine similarity.
- If any cached query has similarity **>= 0.82** → return that cached response.
- This catches natural rephrasings (e.g., *"job failing intermittently"* ≈ *"job failing now and then"*).

### Step 5 — Qdrant Vector Search
- The embedding is sent to **Qdrant** (collection: `spark-incidents-openai`, distance: COSINE).
- Returns **top-10** most similar incidents with similarity scores.

### Step 6 — Deterministic Analyzer
This is the core innovation. It processes the 10 search results **without any LLM call**:

1. **Cluster Detection** — Groups results by root-cause theme using regex patterns (12 categories: Memory/OOM, Data Skew, Executor Failure, Shuffle Failure, etc.).
2. **Recurrence Detection** — Flags themes appearing in **2+ results** as recurring patterns.
3. **Confidence Scoring** — Calculates a 0–1 confidence score from 4 weighted components:

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| Top Similarity | 40% | How closely the best result matches the query |
| Score Consistency | 20% | How tight the spread is among top-3 results |
| Cluster Cohesion | 25% | Whether results point to a single root cause |
| Recurrence Flag | 15% | Whether the pattern is a known recurring issue |

4. **Decision Point:**
   - Confidence **>= 0.70** → Generate answer deterministically (structured markdown with citations). **No LLM invoked. Cost ≈ $0.00005** (embedding only).
   - Confidence **< 0.70** → Fall through to LLM.

### Step 7 — LLM Response Cache Check (Layer 3)
- Before calling the LLM, check cache keyed on `SHA256(query + context[:2000])` (TTL: **30 minutes**).
- If hit → return cached LLM answer without API call.

### Step 8 — LLM Synthesis (Only When Needed)
- Model: **gpt-4o-mini** (temperature: 0.0 for reproducibility).
- Receives: system prompt + top-5 incident contexts.
- Generates a synthesized markdown answer.
- Response is cached for future reuse (Layer 3).

### Step 9 — Metrics Logging
Every query logs: tokens used, cost, latency, path taken, confidence, cache hits → appended to `metrics.jsonl`.

### Step 10 — Response Displayed
The UI shows: answer, confidence bar, path badge (Cache Hit / Deterministic / LLM), cost, latency, expandable clusters, source incidents, and feedback buttons (thumbs up/down).

---

## Part 2: Architecture Diagram (Text)

```
User Query
    │
    ▼
┌──────────────────────┐
│  Query Cache (L1)    │──── Exact SHA-256 match? ──→ Return (cost: $0)
│  TTL: 1 hour         │
└──────────────────────┘
    │ miss
    ▼
┌──────────────────────┐
│  Embedding Cache (L2)│──── Text already embedded? ──→ Reuse vector
│  TTL: 24 hours       │                                (skip OpenAI call)
└──────────────────────┘
    │
    ▼
  Generate Embedding (text-embedding-3-small, 1536-dim)
    │
    ▼
┌──────────────────────┐
│  Semantic Cache (L1) │──── Cosine sim >= 0.82? ──→ Return (cost: ~$0)
│  vs cached queries   │
└──────────────────────┘
    │ miss
    ▼
┌──────────────────────┐
│  Qdrant Vector DB    │──── Top-10 by cosine similarity
│  Collection: openai  │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│ Deterministic Layer  │──── Confidence >= 0.70? ──→ Return (cost: ~$0.00005)
│ Pattern + Clustering │     (structured markdown,
│ + Confidence Score   │      no LLM needed)
└──────────────────────┘
    │ confidence < 0.70
    ▼
┌──────────────────────┐
│  LLM Cache (L3)      │──── Same query+context? ──→ Return (cost: $0)
│  TTL: 30 minutes     │
└──────────────────────┘
    │ miss
    ▼
┌──────────────────────┐
│  gpt-4o-mini         │──── Synthesize answer from top-5 incidents
│  temperature: 0.0    │     (cost: ~$0.00020)
└──────────────────────┘
    │
    ▼
  Cache response → Log metrics → Display to user
```

---

## Part 3: Possible Questions & Answers

### Easy Questions

**Q1: What tech stack are you using?**
> Streamlit (UI), OpenAI API (embeddings + LLM), Qdrant (vector DB), PostgreSQL (incident store + feedback), Redis (optional production cache), Python 3.10+.

**Q2: What embedding model and why?**
> `text-embedding-3-small` — 1536 dimensions, costs $0.02/1M tokens. It's OpenAI's most cost-effective embedding model with strong semantic quality. For an internal incident search tool, it hits the right balance of accuracy vs. cost.

**Q3: What LLM model and why?**
> `gpt-4o-mini` at temperature 0.0. It's 100x cheaper than gpt-4-turbo ($0.15 vs $10 per 1M input tokens) and is only invoked for ambiguous queries (~25% of traffic). Temperature 0.0 ensures reproducible answers.

**Q4: How do you track costs?**
> Every query logs token counts and costs to `metrics.jsonl`. The evaluator calculates per-query and session-level cost using OpenAI's published pricing. The sidebar shows real-time cost metrics.

**Q5: What does the confidence score represent?**
> It's a weighted score (0–1) measuring how well the retrieved incidents answer the query. It combines: match quality (40%), result consistency (20%), theme cohesion (25%), and known-pattern recurrence (15%). Above 0.70, we trust the deterministic answer.

---

### Medium Questions

**Q6: Explain cosine similarity and why you chose it over other distance metrics.**
> Cosine similarity measures the angle between two vectors, returning a value from -1 to 1 (in practice 0 to 1 for embeddings). A value of 1.0 means identical direction (identical meaning), 0.0 means completely unrelated.
>
> **Why cosine over Euclidean distance?**
> - Cosine is **magnitude-invariant** — it only cares about direction, not vector length. Two texts with the same meaning but different lengths get similar scores.
> - Euclidean distance is affected by vector magnitude, which can produce misleading results when comparing texts of varying lengths.
> - OpenAI's embedding models are **optimized for cosine similarity** — their documentation recommends it.
> - Qdrant natively supports cosine distance with optimized indexing (HNSW).
>
> **Why not dot product?**
> Dot product works when vectors are normalized (which OpenAI embeddings are), but cosine similarity is more universally understood and the explicit choice makes the code self-documenting.

**Q7: Why 0.82 as the semantic cache threshold? Why not 0.90 or 0.75?**
> - **0.90** would be too strict — only near-identical rephrasings would match, missing valid semantic matches like "OOM error" ≈ "out of memory issue".
> - **0.75** would be too loose — semantically different queries could incorrectly return cached answers (e.g., "shuffle failure" might match "executor failure").
> - **0.82** was tuned empirically against our Spark incident dataset. It catches rephrasings while avoiding false matches. It's configurable in `config.py` if we need to adjust.

**Q8: Walk me through the 3 cache layers and why you need all three.**
> Each layer catches a different reuse pattern:
>
> | Layer | What It Caches | TTL | Why Needed |
> |-------|---------------|-----|------------|
> | **Query Cache** | Full response by query hash + semantic embedding | 1 hour | Same or similar questions asked repeatedly during an incident |
> | **Embedding Cache** | OpenAI embedding vectors by text hash | 24 hours | Same text chunks appear across different queries; avoids redundant API calls |
> | **LLM Response Cache** | GPT answer by query+context hash | 30 min | Same evidence + question combo shouldn't call the LLM twice |
>
> **Without Layer 1**: Every query hits the embedding API + Qdrant.
> **Without Layer 2**: Re-embedding identical text wastes ~$0.00002/call and adds latency.
> **Without Layer 3**: Two different queries that retrieve the same incidents would both call the LLM.
>
> Together, they create a **compound caching effect** — approximately 75% of queries are answered at $0 cost.

**Q9: How does the deterministic layer generate answers without AI?**
> It uses **regex-based root-cause classification** (12 categories like Memory/OOM, Data Skew, Shuffle Failure) to group the Qdrant search results into clusters. Then it extracts root cause and resolution text from the best-matching incident in each cluster using regex patterns (looking for keywords like "root cause:", "fix:", "resolved by:"). The answer is assembled as structured markdown with:
> - Primary root cause and confidence level
> - Recurring pattern warnings
> - Recommended resolution (quoted from actual incidents)
> - Root-cause breakdown table
> - Links to source incidents
>
> This works because Spark incidents are semi-structured — they follow a template with Root Cause, Resolution, and Key Learnings sections.

**Q10: What happens if the deterministic layer gives a wrong answer?**
> Two safeguards:
> 1. **Confidence threshold (0.70)** — If results are scattered or low-similarity, the system falls through to the LLM which can reason more flexibly.
> 2. **Feedback loop** — Users can thumbs-down any answer. Feedback is stored in PostgreSQL with the confidence score and path taken. Low-confidence + negative feedback signals highlight where the deterministic layer needs tuning (e.g., new regex patterns or threshold adjustment).

---

### Hard Questions

**Q11: Why are you manually ingesting data instead of integrating the data source directly?**
> **Short answer**: This is Phase 2. Automated ingestion from PostgreSQL already exists (`data_pipeline.py`), but we're running manual ingestion (`ingestion_qdrant.py`) for the initial dataset because:
>
> 1. **Data quality control** — The 20 sample incidents were curated to ensure clean root-cause and resolution sections. Production incident data from ServiceNow/Jira is often messy — missing fields, inconsistent formatting, unresolved tickets mixed with resolved ones. We need a clean baseline to validate the system before pointing it at raw data.
>
> 2. **The automated pipeline exists and is ready** — `data_pipeline.py` connects to PostgreSQL, fetches `RESOLVED` incidents where `ingested=FALSE`, extracts structured fields via regex, embeds them, upserts to Qdrant, and marks them as ingested. It handles per-incident errors without cascade failures.
>
> 3. **Phased rollout strategy**:
>    - **Phase 2 (current)**: Manual curation → validate deterministic layer accuracy → tune thresholds
>    - **Phase 3 (next)**: Connect `data_pipeline.py` to the production incident database on a scheduled cron/Airflow job
>    - **Phase 4**: Real-time ingestion via webhook/CDC (Change Data Capture) when incidents are resolved
>
> 4. **The manual approach let us iterate faster** — We could test confidence thresholds, regex patterns, and cache behavior without waiting for a database integration. Once the system is proven, plugging in the data source is a configuration change, not an architecture change.
>
> **Key point**: The architecture already supports automated ingestion. Manual loading is a conscious phase-gating decision, not a limitation.

**Q12: How would you handle data staleness? What if the vector DB has outdated incidents?**
> - **Embedding cache TTL (24h)** ensures re-embedding if text changes.
> - **Query cache TTL (1h)** ensures stale answers expire quickly.
> - The `data_pipeline.py` uses an `ingested` flag — so re-running it only processes new incidents (incremental, not full rebuild).
> - For Phase 3, we'd add a **scheduled pipeline** (e.g., every 6 hours) that pulls newly resolved incidents.
> - For truly stale data, Qdrant supports **point deletion by filter** — we could add a retention policy to remove incidents older than N months.

**Q13: What are the failure modes and how do you handle them?**
> | Failure | Impact | Handling |
> |---------|--------|----------|
> | OpenAI API down | Can't embed or call LLM | Query cache (L1) still works for repeated queries; deterministic layer works if embedding was cached (L2) |
> | Qdrant down | Can't search incidents | Query cache still serves repeated queries; graceful error message for new queries |
> | Redis down | Cache unavailable | Automatic fallback to pickle-based local cache (in-memory + file) |
> | PostgreSQL down | Can't save feedback or ingest | Feedback silently fails (non-blocking); ingestion retries on next run |
> | Bad embedding quality | Poor search results | Low similarity scores → low confidence → LLM fallback provides reasoning |
> | Regex misclassification | Wrong root-cause cluster | Confidence drops (scattered clusters = low cohesion score) → LLM fallback |

**Q14: Why not use a fine-tuned model instead of the deterministic layer?**
> - **Cost**: Fine-tuning requires training data, compute, and ongoing retraining. The deterministic layer costs $0.
> - **Transparency**: Regex patterns are auditable and debuggable. A fine-tuned model is a black box — when it's wrong, you can't easily see why.
> - **Latency**: Pattern matching is <5ms. Even a small fine-tuned model adds 200ms+.
> - **Maintenance**: Adding a new root-cause category means adding one regex line. Retraining a model means collecting examples, labeling, training, evaluating, deploying.
> - **The deterministic layer handles 75% of queries** — fine-tuning would only help the remaining 25%, where gpt-4o-mini already performs well.

**Q15: How does the confidence scoring handle edge cases?**
> - **All results are low similarity** (e.g., novel problem): Top similarity component scores low → confidence drops below 0.70 → LLM handles it.
> - **Results are spread across many clusters** (e.g., vague query): Cluster cohesion component gives 0 for 4+ clusters → confidence drops → LLM.
> - **Single very high match but others are low**: Score consistency drops (large spread) but top similarity is high → borderline confidence → depends on recurrence and cohesion.
> - **Similarity scale (0.50)**: This normalizes scores because chunked documents peak at ~0.45 similarity while full documents peak at ~0.90. Dividing by 0.50 and capping at 1.0 maps both to the same scale.

**Q16: How would you scale this to 100K+ incidents?**
> - **Qdrant**: Already built for scale — uses HNSW indexing, handles millions of vectors. Move to Qdrant Cloud with sharding for horizontal scale.
> - **Cache**: Move from pickle to Redis (already supported). Add Redis cluster for high availability.
> - **Embedding cost**: With 100K incidents, initial ingestion costs ~$2 (at $0.02/1M tokens). The embedding cache prevents re-embedding.
> - **Deterministic layer**: Regex patterns are O(n) on result count (always 10), not on total incidents. No scaling concern.
> - **LLM costs**: Already minimized — only ~25% of queries reach the LLM. With caching, the cost per query stays flat as the dataset grows.
> - **Ingestion**: The pipeline already handles incremental ingestion with per-incident error isolation.

**Q17: Why cosine similarity 0.82 for cache but 0.70 confidence for deterministic — aren't these measuring the same thing?**
> No, they measure fundamentally different things:
> - **0.82 (cache threshold)**: Measures whether two *queries* are asking the same question. This is a *query-to-query* comparison. High threshold because returning a wrong cached answer is worse than recomputing.
> - **0.70 (confidence threshold)**: Measures whether the *retrieved incidents* adequately answer the query. This is a *composite score* of similarity, consistency, cohesion, and recurrence. Lower threshold because the deterministic answer includes citations — the user can verify.
>
> Different purposes → different thresholds → different acceptable error rates.

**Q18: What's the cost comparison between Phase 1 and Phase 2?**
> | Metric | Phase 1 (gpt-4-turbo) | Phase 2 (current) |
> |--------|----------------------|-------------------|
> | Model | gpt-4-turbo | gpt-4o-mini |
> | Input cost | $10.00 / 1M tokens | $0.15 / 1M tokens |
> | Output cost | $30.00 / 1M tokens | $0.60 / 1M tokens |
> | LLM call rate | 100% of queries | ~25% of queries |
> | Avg cost/query | ~$0.05 | ~$0.00005 (det) to $0.0002 (LLM) |
> | **Effective reduction** | baseline | **25x–250x cheaper** |
>
> For 1,000 queries/day: Phase 1 ≈ $50/day → Phase 2 ≈ $0.20–$1.50/day.

---

## Part 4: Quick Reference — Key Numbers

| Parameter | Value | Location |
|-----------|-------|----------|
| Embedding model | text-embedding-3-small | `config.py` |
| Embedding dimensions | 1536 | `config.py` |
| LLM model | gpt-4o-mini | `config.py` |
| LLM temperature | 0.0 | `config.py` |
| Query cache TTL | 1 hour | `config.py` |
| Embedding cache TTL | 24 hours | `config.py` |
| LLM cache TTL | 30 minutes | `config.py` |
| Semantic cache threshold | 0.82 | `config.py` |
| Confidence threshold | 0.70 | `config.py` |
| Similarity scale | 0.50 | `config.py` |
| Min recurrence count | 2 | `config.py` |
| Qdrant search limit | 10 results | `config.py` |
| Root-cause categories | 12 regex patterns | `deterministic_analyzer.py` |
| Sample incidents | 20 files | `spark_incidents/` |
| Deterministic answer rate | ~75% of queries | Measured |
| Cache backend (prod) | Redis | `config.py` |
| Cache backend (fallback) | Pickle (in-memory + file) | `cache_manager.py` |

---

## Part 5: Glossary for Non-Technical Audience

| Term | Plain English |
|------|--------------|
| **Cosine Similarity** | A score (0 to 1) measuring how similar two pieces of text are in meaning. 1.0 = identical, 0.0 = completely unrelated. |
| **Embedding** | Converting text into a list of 1536 numbers that capture its meaning. Similar texts get similar numbers. |
| **Vector Database (Qdrant)** | A database optimized for finding "similar" items rather than exact matches. Like Google search but for internal incidents. |
| **Deterministic Layer** | A rule-based system that answers questions using pattern matching — no AI model needed, no cost, instant response. |
| **RAG (Retrieval-Augmented Generation)** | A pattern where we first *retrieve* relevant documents, then *generate* an answer using AI. The AI answers based on evidence, not guessing. |
| **TTL (Time to Live)** | How long a cached item stays valid before it expires and must be recomputed. |
| **HNSW** | The indexing algorithm Qdrant uses internally to search millions of vectors efficiently (Hierarchical Navigable Small World). |
