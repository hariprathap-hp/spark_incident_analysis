"""
Evaluation Framework — Spark Insight Agent Phase 2.

Tracks per-query metrics and writes them to a JSONL log file.
Provides session-level and all-time summaries via get_session_summary()
and get_all_time_summary().

Metrics tracked per query
─────────────────────────
  - timestamp (UTC ISO-8601)
  - query_length (chars)
  - confidence score from deterministic analyzer
  - path: "query_cache" | "deterministic" | "llm"
  - model_used (str or None when deterministic/cache)
  - embedding_tokens, llm_input_tokens, llm_output_tokens
  - embedding_cost_usd, llm_cost_usd, total_cost_usd
  - latency_ms (wall clock, ms)
  - top_similarity (best Qdrant score)
  - results_count, clusters_detected, recurring_patterns
  - query_cache_hit, embedding_cache_hit, llm_cache_hit (booleans)

Cost model (gpt-4o-mini, Feb 2026 pricing)
  Input:  $0.150 / 1M tokens
  Output: $0.600 / 1M tokens
  Embed:  $0.020 / 1M tokens  (text-embedding-3-small)
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.config import cfg

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    query_length: int = 0
    confidence: float = 0.0
    path: str = ""                      # "query_cache" | "deterministic" | "llm"
    model_used: Optional[str] = None
    embedding_tokens: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    embedding_cost_usd: float = 0.0
    llm_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    latency_ms: float = 0.0
    top_similarity: float = 0.0
    results_count: int = 0
    clusters_detected: int = 0
    recurring_patterns: int = 0
    query_cache_hit: bool = False
    embedding_cache_hit: bool = False
    llm_cache_hit: bool = False

    def calculate_costs(self) -> None:
        """Populate cost fields from token counts using config pricing."""
        ocfg = cfg.openai
        self.embedding_cost_usd = round(
            self.embedding_tokens * ocfg.embedding_cost_per_token, 8
        )
        self.llm_cost_usd = round(
            self.llm_input_tokens * ocfg.llm_input_cost_per_token
            + self.llm_output_tokens * ocfg.llm_output_cost_per_token,
            8,
        )
        self.total_cost_usd = round(self.embedding_cost_usd + self.llm_cost_usd, 8)


class Evaluator:
    """Thread-safe metrics collector with JSONL persistence."""

    def __init__(self, log_path: str = cfg.metrics_log_path) -> None:
        self._log_path = Path(log_path)
        self._lock = threading.Lock()
        self._session: list[QueryMetrics] = []

    # ── Public API ───────────────────────────────────────────────────────────

    def log(self, metrics: QueryMetrics) -> None:
        """Finalize costs, store in-memory, and append to JSONL file."""
        metrics.calculate_costs()
        with self._lock:
            self._session.append(metrics)
            self._append_jsonl(metrics)
        logger.debug(
            "Metrics logged: path=%s conf=%.3f cost=$%.6f lat=%.0fms",
            metrics.path,
            metrics.confidence,
            metrics.total_cost_usd,
            metrics.latency_ms,
        )

    def get_session_summary(self) -> dict:
        """Aggregate stats for queries run in this process lifetime."""
        with self._lock:
            metrics = list(self._session)

        if not metrics:
            return {"message": "No queries logged this session."}

        latencies = [m.latency_ms for m in metrics]
        costs = [m.total_cost_usd for m in metrics]
        confidences = [m.confidence for m in metrics]

        path_counts: dict[str, int] = {}
        for m in metrics:
            path_counts[m.path] = path_counts.get(m.path, 0) + 1

        n = len(metrics)
        llm_avoided = path_counts.get("deterministic", 0) + path_counts.get(
            "query_cache", 0
        )

        return {
            "total_queries": n,
            "total_cost_usd": round(sum(costs), 6),
            "avg_cost_usd": round(statistics.mean(costs), 6) if costs else 0.0,
            "latency_p50_ms": round(statistics.median(latencies), 1),
            "latency_p95_ms": self._percentile(latencies, 95),
            "avg_confidence": round(statistics.mean(confidences), 3) if confidences else 0.0,
            "path_distribution": path_counts,
            "llm_avoided_pct": round(100 * llm_avoided / n, 1) if n else 0.0,
            "cache_hit_rates": {
                "query": self._hit_rate(metrics, "query_cache_hit"),
                "embedding": self._hit_rate(metrics, "embedding_cache_hit"),
                "llm": self._hit_rate(metrics, "llm_cache_hit"),
            },
        }

    def get_all_time_summary(self) -> dict:
        """Read the full JSONL log for all-time aggregate stats."""
        if not self._log_path.exists():
            return {"message": "No historical metrics found."}

        rows: list[dict] = []
        try:
            with open(self._log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except Exception as exc:
            logger.error("Failed to read metrics log: %s", exc)
            return {"error": str(exc)}

        if not rows:
            return {"message": "Metrics log is empty."}

        costs = [r.get("total_cost_usd", 0.0) for r in rows]
        return {
            "total_queries_all_time": len(rows),
            "total_cost_all_time_usd": round(sum(costs), 6),
            "avg_cost_per_query_usd": round(statistics.mean(costs), 6) if costs else 0.0,
        }

    # ── Private ──────────────────────────────────────────────────────────────

    def _append_jsonl(self, metrics: QueryMetrics) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a") as f:
                f.write(json.dumps(asdict(metrics)) + "\n")
        except Exception as exc:
            logger.warning("Failed to write metrics: %s", exc)

    @staticmethod
    def _hit_rate(metrics: list[QueryMetrics], attr: str) -> float:
        if not metrics:
            return 0.0
        hits = sum(1 for m in metrics if getattr(m, attr, False))
        return round(hits / len(metrics), 3)

    @staticmethod
    def _percentile(values: list[float], pct: int) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * pct / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return round(sorted_vals[idx], 1)


# Singleton
evaluator = Evaluator()
