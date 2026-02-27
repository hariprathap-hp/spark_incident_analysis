"""
3-Layer Cache Manager — Spark Insight Agent Phase 2.

Layer 1 — Query Cache
    Key:   SHA-256 of normalized query string
    Value: full response dict
    TTL:   1 hour
    Goal:  instant response for repeated identical queries

Layer 2 — Embedding Cache
    Key:   SHA-256 of input text content
    Value: list[float] embedding vector
    TTL:   24 hours
    Goal:  avoid paying for re-embedding the same text

Layer 3 — LLM Response Cache
    Key:   SHA-256 of (normalized_query + context[:2000])
    Value: LLM answer string
    TTL:   30 minutes
    Goal:  skip LLM when the same evidence yields the same question

Each layer = in-memory dict (fast) + disk pickle (survives restarts).
Thread-safe via RLock.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Optional

from backend.config import cfg

logger = logging.getLogger(__name__)


class _TTLCache:
    """Thread-safe in-memory dict backed by a pickle file, with per-entry TTL."""

    def __init__(self, name: str, ttl_seconds: int, cache_dir: str) -> None:
        self.name = name
        self.ttl = ttl_seconds
        self._path = Path(cache_dir) / f"{name}.pkl"
        self._lock = threading.RLock()
        # store: key → (value, expire_at_epoch)
        self._store: dict[str, tuple[Any, float]] = {}
        self._hits = 0
        self._misses = 0
        self._load_from_disk()

    # ── Public API ──────────────────────────────────────────────────────────

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
            self._persist_to_disk()

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._persist_to_disk()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            active = sum(1 for _, (_, exp) in self._store.items() if exp > now)
            total = self._hits + self._misses
            return {
                "total_entries": len(self._store),
                "active_entries": active,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
            }

    # ── Private ─────────────────────────────────────────────────────────────

    def _persist_to_disk(self) -> None:
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
            # Evict already-expired entries on load
            self._store = {k: v for k, v in raw.items() if v[1] > now}
            logger.info(
                "Cache '%s' restored: %d active entries", self.name, len(self._store)
            )
        except Exception as exc:
            logger.warning(
                "Cache load failed (%s), starting fresh: %s", self.name, exc
            )
            self._store = {}


class CacheManager:
    """Unified 3-layer cache with per-layer statistics."""

    def __init__(self) -> None:
        d = cfg.cache.cache_dir
        self.query_cache = _TTLCache("query", cfg.cache.query_ttl_seconds, d)
        self.embedding_cache = _TTLCache(
            "embedding", cfg.cache.embedding_ttl_seconds, d
        )
        self.llm_cache = _TTLCache("llm", cfg.cache.llm_ttl_seconds, d)

    # ── Layer 1: Query cache ────────────────────────────────────────────────

    def get_query(self, query: str) -> Optional[dict]:
        result = self.query_cache.get(self._query_key(query))
        if result is not None:
            logger.debug("Query cache HIT: %.60s", query)
        return result

    def set_query(self, query: str, response: dict) -> None:
        self.query_cache.set(self._query_key(query), response)

    # ── Layer 2: Embedding cache ────────────────────────────────────────────

    def get_embedding(self, text: str) -> Optional[list[float]]:
        result = self.embedding_cache.get(self._content_hash(text))
        if result is not None:
            logger.debug("Embedding cache HIT (%d chars)", len(text))
        return result

    def set_embedding(self, text: str, embedding: list[float]) -> None:
        self.embedding_cache.set(self._content_hash(text), embedding)

    # ── Layer 3: LLM response cache ─────────────────────────────────────────

    def get_llm(self, query: str, context: str) -> Optional[str]:
        result = self.llm_cache.get(self._llm_key(query, context))
        if result is not None:
            logger.debug("LLM cache HIT")
        return result

    def set_llm(self, query: str, context: str, answer: str) -> None:
        self.llm_cache.set(self._llm_key(query, context), answer)

    # ── Aggregate stats ─────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, dict]:
        return {
            "query": self.query_cache.stats(),
            "embedding": self.embedding_cache.stats(),
            "llm": self.llm_cache.stats(),
        }

    def clear_all(self) -> None:
        for layer in (self.query_cache, self.embedding_cache, self.llm_cache):
            layer.clear()
        logger.info("All cache layers cleared.")

    # ── Key helpers ─────────────────────────────────────────────────────────

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
