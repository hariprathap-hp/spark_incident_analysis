"""
3-Layer Cache Manager — Spark Insight Agent Phase 2.

Layer 1 — Query Cache  (SEMANTIC matching via embedding cosine similarity)
    Store:  query embedding + response dict, keyed by unique ID
    Match:  cosine similarity >= semantic_similarity_threshold (default 0.90)
    TTL:    1 hour
    Goal:   return cached response for semantically equivalent queries
            e.g. "job failing intermittently" ≈ "job failing now and then"

Layer 2 — Embedding Cache  (exact SHA-256 match)
    Key:   SHA-256 of input text content
    Value: list[float] embedding vector
    TTL:   24 hours
    Goal:  avoid paying for re-embedding the same text

Layer 3 — LLM Response Cache  (exact SHA-256 match)
    Key:   SHA-256 of (normalized_query + context[:2000])
    Value: LLM answer string
    TTL:   30 minutes
    Goal:  skip LLM when the same evidence yields the same question

Backend: Redis (if REDIS_URL is set) or in-memory dict + pickle fallback.
Thread-safe via RLock (in-memory) or Redis atomicity.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from backend.config import cfg

logger = logging.getLogger(__name__)

# ── Backend abstraction ──────────────────────────────────────────────────────


class _PickleBackend:
    """In-memory dict backed by a pickle file (original behaviour)."""

    def __init__(self, name: str, ttl: int, cache_dir: str) -> None:
        self.name = name
        self.ttl = ttl
        self._path = Path(cache_dir) / f"{name}.pkl"
        self._lock = threading.RLock()
        self._store: dict[str, tuple[Any, float]] = {}
        self._hits = 0
        self._misses = 0
        self._load_from_disk()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, expire_at = entry
            if time.time() > expire_at:
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (value, time.time() + self.ttl)
            self._persist()

    def scan_values(self) -> list[tuple[str, Any]]:
        """Return all non-expired (key, value) pairs."""
        with self._lock:
            now = time.time()
            return [
                (k, v) for k, (v, exp) in self._store.items() if exp > now
            ]

    def delete(self, key: str) -> bool:
        """Delete a single key. Returns True if the key existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                self._persist()
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._persist()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            active = sum(1 for _, (_, exp) in self._store.items() if exp > now)
            total = self._hits + self._misses
            return {
                "backend": "pickle",
                "total_entries": len(self._store),
                "active_entries": active,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
            }

    def _persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "wb") as f:
                pickle.dump(self._store, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            logger.warning("Cache persist failed (%s): %s", self.name, exc)

    def _load_from_disk(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "rb") as f:
                raw: dict = pickle.load(f)
            now = time.time()
            self._store = {k: v for k, v in raw.items() if v[1] > now}
            logger.info(
                "Cache '%s' restored: %d active entries",
                self.name,
                len(self._store),
            )
        except Exception as exc:
            logger.warning(
                "Cache load failed (%s), starting fresh: %s", self.name, exc
            )
            self._store = {}


class _RedisBackend:
    """Redis-backed cache layer with TTL handled by Redis SETEX."""

    def __init__(self, name: str, ttl: int, redis_client: Any) -> None:
        self.name = name
        self.ttl = ttl
        self._r = redis_client
        self._prefix = f"{cfg.cache.redis_key_prefix}{name}:"
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        raw = self._r.get(self._prefix + key)
        if raw is None:
            self._misses += 1
            return None
        self._hits += 1
        return json.loads(raw)

    def set(self, key: str, value: Any) -> None:
        self._r.setex(self._prefix + key, self.ttl, json.dumps(value))

    def scan_values(self) -> list[tuple[str, Any]]:
        """Return all (key, value) pairs in this layer."""
        results = []
        prefix_len = len(self._prefix)
        for full_key in self._r.scan_iter(match=self._prefix + "*"):
            raw = self._r.get(full_key)
            if raw is not None:
                short_key = full_key.decode()[prefix_len:] if isinstance(full_key, bytes) else full_key[prefix_len:]
                results.append((short_key, json.loads(raw)))
        return results

    def delete(self, key: str) -> bool:
        """Delete a single key. Returns True if the key existed."""
        return bool(self._r.delete(self._prefix + key))

    def clear(self) -> None:
        keys = list(self._r.scan_iter(match=self._prefix + "*"))
        if keys:
            self._r.delete(*keys)

    def stats(self) -> dict[str, Any]:
        count = sum(1 for _ in self._r.scan_iter(match=self._prefix + "*"))
        total = self._hits + self._misses
        return {
            "backend": "redis",
            "total_entries": count,
            "active_entries": count,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }


def _make_backend(name: str, ttl: int, cache_dir: str) -> _PickleBackend | _RedisBackend:
    """Create the appropriate cache backend."""
    if cfg.cache.redis_url:
        try:
            import redis

            client = redis.from_url(cfg.cache.redis_url, decode_responses=True)
            client.ping()
            logger.info("Redis connected for cache layer '%s'", name)
            return _RedisBackend(name, ttl, client)
        except Exception as exc:
            logger.warning(
                "Redis unavailable for '%s', falling back to pickle: %s",
                name,
                exc,
            )
    return _PickleBackend(name, ttl, cache_dir)


# ── Semantic query matching helpers ──────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ── Stampede protection ─────────────────────────────────────────────────────


class _InflightEntry:
    """Tracks a single in-flight computation."""

    __slots__ = ("event", "result", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None


class QueryCoalescer:
    """Collapse concurrent identical queries into a single computation.

    Usage (in the caller)::

        coalescer = QueryCoalescer()
        key = CacheManager._query_key(query)

        should_compute, entry = coalescer.acquire(key)
        if should_compute:
            try:
                result = expensive_computation()
                coalescer.resolve(key, result)
            except BaseException as exc:
                coalescer.reject(key, exc)
                raise
            return result
        else:
            return coalescer.wait(entry)  # blocks until first caller finishes
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: dict[str, _InflightEntry] = {}

    def acquire(self, key: str) -> tuple[bool, _InflightEntry]:
        """Try to become the computing thread for *key*.

        Returns ``(True, entry)`` if this caller should compute the result.
        Returns ``(False, entry)`` if another thread is already computing —
        the caller should call :meth:`wait` on the returned entry.
        """
        with self._lock:
            if key in self._inflight:
                return False, self._inflight[key]
            entry = _InflightEntry()
            self._inflight[key] = entry
            return True, entry

    def resolve(self, key: str, result: Any) -> None:
        """Store the computed result and wake all waiting threads."""
        with self._lock:
            entry = self._inflight.pop(key, None)
        if entry is not None:
            entry.result = result
            entry.event.set()

    def reject(self, key: str, error: BaseException) -> None:
        """Signal failure so waiters can raise the same error."""
        with self._lock:
            entry = self._inflight.pop(key, None)
        if entry is not None:
            entry.error = error
            entry.event.set()

    def wait(self, entry: _InflightEntry, timeout: float = 30.0) -> Any:
        """Block until the computing thread finishes, then return its result.

        Args:
            entry: The ``_InflightEntry`` returned by :meth:`acquire`.
            timeout: Max seconds to wait (safety valve).

        Raises:
            TimeoutError: If the computing thread takes longer than *timeout*.
            BaseException: Re-raises whatever the computing thread raised.
        """
        if not entry.event.wait(timeout=timeout):
            raise TimeoutError(
                f"Cache stampede wait exceeded {timeout}s — falling through"
            )
        if entry.error is not None:
            raise entry.error
        return entry.result


# ── Main CacheManager ───────────────────────────────────────────────────────


class CacheManager:
    """Unified 3-layer cache with semantic query matching and pluggable backend."""

    def __init__(self) -> None:
        d = cfg.cache.cache_dir
        self.query_cache = _make_backend("query", cfg.cache.query_ttl_seconds, d)
        self.embedding_cache = _make_backend(
            "embedding", cfg.cache.embedding_ttl_seconds, d
        )
        self.llm_cache = _make_backend("llm", cfg.cache.llm_ttl_seconds, d)
        self.last_best_similarity: float | None = None  # best score from most recent get_query()
        self.coalescer = QueryCoalescer()

    # ── Layer 1: Query cache (SEMANTIC matching) ─────────────────────────────

    def get_query(self, query: str, query_embedding: list[float] | None = None) -> Optional[dict]:
        """
        Look up the query cache.

        1. Try exact hash match first (fast path).
        2. If miss AND query_embedding is provided, scan all cached entries
           and return the best match above the semantic similarity threshold.
        """
        # Fast path: exact match
        exact = self.query_cache.get(self._query_key(query))
        if exact is not None:
            logger.debug("Query cache HIT (exact): %.60s", query)
            self.last_best_similarity = 1.0
            result = {k: v for k, v in exact.items() if k != "_cache_embedding"}
            result["_cache_similarity"] = 1.0  # exact match
            return result

        # Semantic path: compare embeddings
        if query_embedding is not None:
            best_score = 0.0
            best_response = None
            absolute_best_score = 0.0  # Track highest score regardless of threshold
            threshold = cfg.cache.semantic_similarity_threshold

            all_entries = self.query_cache.scan_values()
            entries_with_embedding = 0

            for _key, entry in all_entries:
                cached_embedding = entry.get("_cache_embedding")
                if cached_embedding is None:
                    continue
                entries_with_embedding += 1
                try:
                    score = _cosine_similarity(query_embedding, cached_embedding)
                except Exception as exc:
                    logger.warning("Cosine similarity failed: %s", exc)
                    continue
                logger.info(
                    "Semantic compare: score=%.4f threshold=%.2f key=%.40s",
                    score, threshold, _key,
                )
                if score > absolute_best_score:
                    absolute_best_score = score
                if score >= threshold and score > best_score:
                    best_score = score
                    best_response = entry

            # Always store the best similarity found (even on miss)
            self.last_best_similarity = round(absolute_best_score, 4) if entries_with_embedding > 0 else None

            logger.info(
                "Semantic scan: %d total entries, %d with embeddings, best=%.4f, threshold=%.2f, hit=%s",
                len(all_entries), entries_with_embedding, absolute_best_score, threshold,
                best_response is not None,
            )

            if best_response is not None:
                # Strip internal embedding before returning
                result = {k: v for k, v in best_response.items() if k != "_cache_embedding"}
                result["_cache_similarity"] = round(best_score, 4)
                return result

        return None

    def set_query(self, query: str, response: dict, query_embedding: list[float] | None = None) -> None:
        """Cache a query response, storing the embedding alongside for semantic matching."""
        entry = {**response}
        if query_embedding is not None:
            entry["_cache_embedding"] = query_embedding
            logger.info(
                "Storing query in cache with embedding (%d dims): %.60s",
                len(query_embedding), query,
            )
        self.query_cache.set(self._query_key(query), entry)

    # ── Layer 2: Embedding cache ─────────────────────────────────────────────

    def get_embedding(self, text: str) -> Optional[list[float]]:
        result = self.embedding_cache.get(self._content_hash(text))
        if result is not None:
            logger.debug("Embedding cache HIT (%d chars)", len(text))
        return result

    def set_embedding(self, text: str, embedding: list[float]) -> None:
        self.embedding_cache.set(self._content_hash(text), embedding)

    # ── Layer 3: LLM response cache ──────────────────────────────────────────

    def get_llm(self, query: str, context: str) -> Optional[str]:
        result = self.llm_cache.get(self._llm_key(query, context))
        if result is not None:
            logger.debug("LLM cache HIT")
        return result

    def set_llm(self, query: str, context: str, answer: str) -> None:
        self.llm_cache.set(self._llm_key(query, context), answer)

    # ── Aggregate stats ──────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, dict]:
        return {
            "query": self.query_cache.stats(),
            "embedding": self.embedding_cache.stats(),
            "llm": self.llm_cache.stats(),
        }

    def invalidate_similar(
        self,
        incident_embedding: list[float],
        threshold: float | None = None,
    ) -> int:
        """Evict Layer 1 (query cache) entries semantically similar to a new incident.

        Layer 3 (LLM cache) is self-healing: evicted queries will miss Layer 1,
        re-query Qdrant (which now includes the new incident), produce a different
        context string, and therefore generate a new LLM cache key automatically.

        Args:
            incident_embedding: The embedding of the newly ingested incident.
            threshold: Cosine similarity threshold for eviction.
                       Defaults to ``cfg.cache.invalidation_similarity_threshold``.

        Returns:
            Number of cache entries evicted.
        """
        if threshold is None:
            threshold = cfg.cache.invalidation_similarity_threshold

        evicted = 0
        for key, entry in self.query_cache.scan_values():
            cached_embedding = entry.get("_cache_embedding")
            if cached_embedding is None:
                continue
            try:
                score = _cosine_similarity(incident_embedding, cached_embedding)
            except Exception:
                continue
            if score >= threshold:
                self.query_cache.delete(key)
                evicted += 1
                logger.info(
                    "Cache invalidated: key=%.40s similarity=%.4f", key, score
                )

        if evicted:
            logger.info(
                "Targeted invalidation: evicted %d/%d query cache entries (threshold=%.2f)",
                evicted,
                evicted + len(self.query_cache.scan_values()),
                threshold,
            )
        return evicted

    def clear_all(self) -> None:
        for layer in (self.query_cache, self.embedding_cache, self.llm_cache):
            layer.clear()
        logger.info("All cache layers cleared.")

    # ── Key helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _query_key(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _llm_key(query: str, context: str) -> str:
        payload = query.strip().lower() + "||" + context[:2000]
        return hashlib.sha256(payload.encode()).hexdigest()


# Singleton — import this everywhere
cache_manager = CacheManager()
