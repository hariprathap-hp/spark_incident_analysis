"""
Centralized configuration for Spark Insight Agent — Phase 2.

All previously hardcoded values (model names, URLs, thresholds) are here.
Values are loaded from .env via python-dotenv so nothing secrets-related
lives in source code.

Usage:
    from backend.config import cfg
    model = cfg.openai.llm_model
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class OpenAIConfig:
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = "text-embedding-3-small"
    vector_size: int = 1536
    # gpt-4o-mini is ~10x cheaper than gpt-4-turbo with comparable quality for RAG
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # Cost per token (USD) — update if OpenAI pricing changes
    embedding_cost_per_token: float = 2e-8        # $0.020 / 1M tokens
    llm_input_cost_per_token: float = 1.5e-7      # $0.150 / 1M tokens (gpt-4o-mini)
    llm_output_cost_per_token: float = 6e-7       # $0.600 / 1M tokens


@dataclass
class QdrantConfig:
    url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    collection_name: str = "spark-incidents-openai"
    search_limit: int = 10   # Fetch top-10; deterministic layer filters further
    timeout: int = 30


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "spark_incidents"
    user: str = field(default_factory=lambda: os.getenv("DB_USER", ""))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))


@dataclass
class AnalysisConfig:
    # Confidence >= threshold → skip LLM, return deterministic answer
    confidence_threshold: float = 0.7
    # Incidents with similarity this close = same cluster
    cluster_spread_threshold: float = 0.10
    # Minimum occurrences to flag as a recurring pattern
    min_recurrence_count: int = 2
    # Expected "perfect match" cosine similarity for this dataset.
    # Full-document embeddings typically peak at 0.85-0.95.
    # Chunk-based embeddings (short text) typically peak at 0.40-0.55.
    # Component A is normalized: score / similarity_scale, capped at 1.0.
    similarity_scale: float = 0.50


@dataclass
class CacheConfig:
    cache_dir: str = ".cache"
    query_ttl_seconds: int = 3600      # 1 hour  — exact query matches
    embedding_ttl_seconds: int = 86400  # 24 hours — embedding vectors
    llm_ttl_seconds: int = 1800         # 30 minutes — LLM responses


@dataclass
class AppConfig:
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    metrics_log_path: str = "metrics.jsonl"
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


# Singleton — import this everywhere
cfg = AppConfig()
