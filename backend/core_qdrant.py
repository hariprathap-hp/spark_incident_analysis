"""
Core RAG Query Engine — Spark Insight Agent Phase 2.

Architecture (top → bottom)
════════════════════════════

  User Query
      │
  ┌───▼──────────────────────────────────────────────────────┐
  │  1. QUERY CACHE   (exact-match, 1 h TTL)                 │
  │     hit → return immediately, cost = $0                  │
  └───┬──────────────────────────────────────────────────────┘
      │ miss
  ┌───▼──────────────────────────────────────────────────────┐
  │  2. EMBEDDING     (text-embedding-3-small)               │
  │     + embedding cache (SHA-256, 24 h TTL)               │
  └───┬──────────────────────────────────────────────────────┘
      │
  ┌───▼──────────────────────────────────────────────────────┐
  │  3. QDRANT SEARCH  (top-10 by cosine similarity)         │
  └───┬──────────────────────────────────────────────────────┘
      │
  ┌───▼──────────────────────────────────────────────────────┐
  │  4. DETERMINISTIC ANALYZER                               │
  │     • similarity scoring & spread                        │
  │     • root-cause cluster detection                       │
  │     • recurrence pattern detection                       │
  │     • confidence score (0.0 – 1.0)                       │
  └───┬──────────────────────────────────────────────────────┘
      │
      ├── confidence ≥ 0.7 ──► structured answer (LLM skipped)
      │
      └── confidence < 0.7 ──► LLM cache → GPT-4o-mini call
                                             (if cache miss)
      │
  ┌───▼──────────────────────────────────────────────────────┐
  │  5. EVALUATOR — log tokens, cost, latency, path          │
  └──────────────────────────────────────────────────────────┘

Model
─────
  Phase 1: gpt-4-turbo (~$0.01 input / $0.03 output per 1K tokens)
  Phase 2: gpt-4o-mini  (~$0.00015 input / $0.0006 output per 1K tokens)
           ≈ 30-50× cheaper with comparable quality for structured RAG tasks.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import tiktoken
from openai import OpenAI
from qdrant_client import QdrantClient

from backend.cache_manager import cache_manager
from backend.config import cfg
from backend.deterministic_analyzer import analyze
from backend.evaluator import QueryMetrics, evaluator

logger = logging.getLogger(__name__)

# ── Module-level singletons (created once, reused across requests) ────────────

_openai = OpenAI(api_key=cfg.openai.api_key)

_qdrant = QdrantClient(
    url=cfg.qdrant.url,
    api_key=cfg.qdrant.api_key,
    timeout=cfg.qdrant.timeout,
)

_tokenizer = tiktoken.get_encoding("cl100k_base")

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a Spark incident analysis expert.
Use ONLY the provided incident context to answer questions.
Structure your response with:
  1. Root Cause — what caused the incident
  2. Resolution — steps taken to resolve it
  3. Key Learnings — what to prevent recurrence
If the context does not contain enough information, say so explicitly.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def get_embedding(text: str) -> tuple[list[float], int, bool]:
    """
    Return (embedding_vector, token_count, cache_hit).

    Checks the embedding cache first; falls back to OpenAI API only on miss.
    token_count is 0 on a cache hit (no API call was made).
    """
    cached = cache_manager.get_embedding(text)
    if cached is not None:
        return cached, 0, True

    response = _openai.embeddings.create(
        model=cfg.openai.embedding_model,
        input=text,
    )
    embedding = response.data[0].embedding
    tokens = response.usage.total_tokens
    cache_manager.set_embedding(text, embedding)
    logger.debug("Embedding API call: %d tokens", tokens)
    return embedding, tokens, False


def search_incidents(query_embedding: list[float], k: Optional[int] = None) -> list:
    """Return top-k ScoredPoint results from Qdrant."""
    k = k or cfg.qdrant.search_limit
    return _qdrant.query_points(
        collection_name=cfg.qdrant.collection_name,
        query=query_embedding,
        limit=k,
    ).points


def _call_llm(query: str, context: str) -> tuple[str, int, int, bool]:
    """
    Return (answer, input_tokens, output_tokens, cache_hit).

    Checks LLM cache before calling the API.
    token counts are 0 on a cache hit.
    """
    cached = cache_manager.get_llm(query, context)
    if cached is not None:
        return cached, 0, 0, True

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context from similar incidents:\n\n{context}\n\nQuestion: {query}"
            ),
        },
    ]

    response = _openai.chat.completions.create(
        model=cfg.openai.llm_model,
        temperature=cfg.openai.llm_temperature,
        messages=messages,
    )
    answer: str = response.choices[0].message.content or ""
    in_tok: int = response.usage.prompt_tokens
    out_tok: int = response.usage.completion_tokens

    cache_manager.set_llm(query, context, answer)
    logger.info(
        "LLM call: model=%s in=%d out=%d tokens",
        cfg.openai.llm_model,
        in_tok,
        out_tok,
    )
    return answer, in_tok, out_tok, False


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_llm(query: str) -> dict[str, Any]:
    """
    Full RAG pipeline with deterministic analysis and multi-layer caching.

    Returns a dict compatible with the Phase 1 Streamlit UI, extended with
    Phase 2 metadata fields (confidence, path, cost_usd, clusters, …).
    """
    start_ms = time.time() * 1000
    metrics = QueryMetrics(query_length=len(query))

    # ── 1. Query cache ────────────────────────────────────────────────────────
    cached_response = cache_manager.get_query(query)
    if cached_response is not None:
        metrics.path = "query_cache"
        metrics.query_cache_hit = True
        metrics.latency_ms = round(time.time() * 1000 - start_ms, 1)
        evaluator.log(metrics)
        logger.info("Query cache HIT — returning instantly (%.1f ms)", metrics.latency_ms)
        return {**cached_response, "cache_hit": True, "latency_ms": metrics.latency_ms}

    # ── 2. Embedding (with cache) ─────────────────────────────────────────────
    try:
        embedding, embed_tokens, embed_cached = get_embedding(query)
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        raise

    metrics.embedding_tokens = embed_tokens
    metrics.embedding_cache_hit = embed_cached

    # ── 3. Qdrant search ──────────────────────────────────────────────────────
    try:
        search_results = search_incidents(embedding)
    except Exception as exc:
        logger.error("Qdrant search failed: %s", exc)
        raise

    metrics.results_count = len(search_results)
    logger.debug("Qdrant returned %d results", len(search_results))

    # ── 4. Deterministic analysis ─────────────────────────────────────────────
    det = analyze(query, search_results)
    metrics.confidence = det.confidence
    metrics.top_similarity = det.top_similarity
    metrics.clusters_detected = len(det.clusters)
    metrics.recurring_patterns = len(det.recurring_patterns)

    answer: str

    if det.path == "deterministic":
        # High confidence → skip LLM entirely
        metrics.path = "deterministic"
        answer = det.answer or ""
    else:
        # Low confidence → build context and call LLM (or hit LLM cache)
        context = "\n\n---\n\n".join(
            f"Incident:\n{r.payload.get('text', '')}" for r in search_results[:5]
        )
        try:
            llm_answer, in_tok, out_tok, llm_cached = _call_llm(query, context)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise

        answer = llm_answer
        metrics.path = "llm"
        metrics.model_used = cfg.openai.llm_model
        metrics.llm_input_tokens = in_tok
        metrics.llm_output_tokens = out_tok
        metrics.llm_cache_hit = llm_cached

    # ── 5. Log metrics ────────────────────────────────────────────────────────
    metrics.latency_ms = round(time.time() * 1000 - start_ms, 1)
    evaluator.log(metrics)

    response: dict[str, Any] = {
        # Phase 1 compatible
        "answer": answer,
        "response_time": round(metrics.latency_ms / 1000, 2),
        "source_documents": search_results,
        # Phase 2 extras
        "confidence": det.confidence,
        "top_similarity": det.top_similarity,
        "path": metrics.path,
        "model": metrics.model_used,
        "cost_usd": metrics.total_cost_usd,
        "latency_ms": metrics.latency_ms,
        "clusters": [
            {
                "theme": c.theme,
                "size": c.size,
                "avg_similarity": c.avg_similarity,
                "incident_ids": c.incident_ids,
            }
            for c in det.clusters
        ],
        "recurring_patterns": det.recurring_patterns,
        "cache_hit": False,
        "debug": det.debug_info,
    }

    # Store in query cache for the next identical query
    cache_manager.set_query(query, response)
    return response
