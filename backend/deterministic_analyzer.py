"""
Deterministic Analysis Layer — Spark Insight Agent Phase 2.

Runs BEFORE the LLM on every query. If the retrieved incidents are
sufficiently consistent, it returns a structured answer directly,
avoiding an expensive LLM call entirely.

Pipeline
────────
  1. Similarity scoring   — rank retrieved incidents; compute spread
  2. Cluster detection    — group incidents by root-cause theme (regex)
  3. Recurrence analysis  — flag themes appearing ≥ N times
  4. Confidence scoring   — weighted 4-component score (0.0 – 1.0)
  5. Answer generation    — structured Markdown when confidence ≥ threshold

Confidence components
─────────────────────
  A. Top similarity score      (weight 0.40) — how close is the best match?
  B. Score consistency top-3   (weight 0.20) — are results tightly clustered?
  C. Root-cause theme cohesion (weight 0.25) — do results agree on the cause?
  D. Recurrence flag           (weight 0.15) — is this a known repeat offender?
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from backend.config import cfg

logger = logging.getLogger(__name__)


# ── Root-cause taxonomy ──────────────────────────────────────────────────────
#   (pattern, human-readable label)
_ROOT_CAUSE_PATTERNS: list[tuple[str, str]] = [
    (r"\b(oom|out.of.memory|memory.pressure|heap|gc.overhead|outofmemory)\b", "Memory / OOM"),
    (r"\b(data.skew|skew|partition.*imbalance|hot.partition|uneven.partition)\b", "Data Skew"),
    (r"\b(executor.lost|executor.fail|executor.*crash|worker.*lost)\b", "Executor Failure"),
    (r"\b(shuffle.*fail|shuffle.*error|shuffle.fetch|fetch.failed)\b", "Shuffle Failure"),
    (r"\b(disk.space|no.space|storage.full|disk.full|disk.*exhaust)\b", "Disk Space"),
    (r"\b(network|timeout|connection.refused|unreachable|socket)\b", "Network / Timeout"),
    (r"\b(stage.fail|task.fail|job.fail|stage.*abort)\b", "Stage / Task Failure"),
    (r"\b(hdfs|s3|gcs|storage.*error|read.*error|write.*error|io.error)\b", "Storage I/O"),
    (r"\b(config|setting|parameter|misconfigur|wrong.*value)\b", "Misconfiguration"),
    (r"\b(dependency|jar|class.*not.*found|library|import.error)\b", "Dependency Issue"),
    (r"\b(broadcast|broadcast.*join|broadcast.*timeout)\b", "Broadcast Join"),
    (r"\b(spill|disk.spill|spilling.to.disk)\b", "Memory Spill"),
]


def _classify_root_cause(text: str) -> str:
    lower = text.lower()
    for pattern, label in _ROOT_CAUSE_PATTERNS:
        if re.search(pattern, lower):
            return label
    return "General Failure"


def _extract_field(text: str, field_name: str) -> str:
    """Extract a single-line field value from formatted incident text."""
    match = re.search(rf"^{field_name}:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_resolution(text: str) -> str:
    """Extract the Resolution block (may be multi-line)."""
    match = re.search(
        r"Resolution:\s*\n(.*?)(?:\n(?:[A-Z][a-z]|\Z))",
        text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    # Fallback: single-line
    single = _extract_field(text, "Resolution")
    return single or "See incident for details."


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class IncidentCluster:
    theme: str
    incident_ids: list[str]
    similarity_scores: list[float]
    common_root_cause: str
    common_resolution: str

    @property
    def size(self) -> int:
        return len(self.incident_ids)

    @property
    def avg_similarity(self) -> float:
        if not self.similarity_scores:
            return 0.0
        return round(sum(self.similarity_scores) / len(self.similarity_scores), 3)


@dataclass
class DeterministicResult:
    confidence: float
    top_similarity: float
    clusters: list[IncidentCluster]
    recurring_patterns: list[str]
    path: str                        # "deterministic" | "llm_required"
    answer: Optional[str] = None     # Populated only when path == "deterministic"
    debug_info: dict = field(default_factory=dict)


# ── Analysis helpers ─────────────────────────────────────────────────────────


def _score_consistency(scores: list[float]) -> float:
    """
    How tight is the similarity spread across the top-3 results?
    Returns 1.0 if all identical, 0.0 if spread ≥ 0.2.
    """
    if len(scores) < 2:
        return 1.0
    top = scores[: min(3, len(scores))]
    spread = max(top) - min(top)
    return max(0.0, 1.0 - spread * 5.0)


def _detect_clusters(search_results: list) -> list[IncidentCluster]:
    """Group results by root-cause theme; sort by avg similarity descending."""
    groups: dict[str, list] = {}
    for r in search_results:
        text = r.payload.get("text", "")
        theme = _classify_root_cause(text)
        groups.setdefault(theme, []).append(r)

    clusters: list[IncidentCluster] = []
    for theme, results in groups.items():
        best = max(results, key=lambda r: r.score)
        best_text = best.payload.get("text", "")
        clusters.append(
            IncidentCluster(
                theme=theme,
                incident_ids=[r.payload.get("incident_id", "?") for r in results],
                similarity_scores=[r.score for r in results],
                common_root_cause=_extract_field(best_text, "Root Cause")
                or _classify_root_cause(best_text),
                common_resolution=_extract_resolution(best_text),
            )
        )

    return sorted(clusters, key=lambda c: c.avg_similarity, reverse=True)


def _detect_recurrence(search_results: list) -> list[str]:
    """Return root-cause themes seen ≥ min_recurrence_count times."""
    counts: dict[str, int] = {}
    for r in search_results:
        theme = _classify_root_cause(r.payload.get("text", ""))
        counts[theme] = counts.get(theme, 0) + 1

    threshold = cfg.analysis.min_recurrence_count
    return [theme for theme, cnt in counts.items() if cnt >= threshold]


def _calculate_confidence(
    scores: list[float],
    clusters: list[IncidentCluster],
    recurring: list[str],
) -> float:
    if not scores:
        return 0.0

    # A — Top similarity, normalized to the dataset's expected peak score.
    # Dividing by similarity_scale maps the data's realistic "perfect match"
    # to 1.0, so short-chunk embeddings (peak ~0.45) behave the same as
    # full-doc embeddings (peak ~0.90). Weight: 0.40.
    scale = cfg.analysis.similarity_scale
    comp_a = min(scores[0] / scale, 1.0) * 0.40

    # B — Score consistency in top-3 (0 → 0.20)
    comp_b = _score_consistency(scores) * 0.20

    # C — Cluster cohesion: fewer themes = more confident (0 → 0.25)
    n = len(clusters)
    if n == 1:
        comp_c = 0.25
    elif n == 2:
        comp_c = 0.15
    elif n == 3:
        comp_c = 0.08
    else:
        comp_c = 0.0

    # D — Recurring pattern (0 → 0.15)
    comp_d = 0.15 if recurring else 0.0

    raw = comp_a + comp_b + comp_c + comp_d
    return round(min(raw, 1.0), 3)


# ── Answer builder ───────────────────────────────────────────────────────────


def _build_deterministic_answer(
    query: str,
    search_results: list,
    clusters: list[IncidentCluster],
    recurring: list[str],
    confidence: float,
) -> str:
    top_sim = round(search_results[0].score, 3) if search_results else 0.0
    primary = clusters[0] if clusters else None

    lines: list[str] = [
        f"### Deterministic Analysis",
        f"*Confidence: **{confidence:.0%}** | Top similarity: **{top_sim:.3f}** | "
        f"No LLM call — answer derived from pattern matching*",
        "",
    ]

    if primary:
        lines += [
            f"**Likely Root Cause:** `{primary.theme}`",
            f"**Details:** {primary.common_root_cause}",
            "",
        ]

    if recurring:
        lines += [
            f"**Recurring Pattern:** {', '.join(recurring)} "
            f"(detected in {len(search_results)} similar incidents)",
            "",
        ]

    if primary and primary.common_resolution:
        lines += [
            "**Recommended Resolution** *(from best-matching incident)*:",
            primary.common_resolution,
            "",
        ]

    if len(clusters) > 1:
        lines += [
            "**Root-Cause Breakdown:**",
            *[
                f"- `{c.theme}` — {c.size} incident(s), avg similarity {c.avg_similarity:.3f}"
                for c in clusters
            ],
            "",
        ]

    lines.append("**Similar Incidents Retrieved:**")
    for r in search_results[:5]:
        inc_id = r.payload.get("incident_id", "?")
        cluster = r.payload.get("cluster", "?")
        theme = _classify_root_cause(r.payload.get("text", ""))
        lines.append(
            f"- **{inc_id}** — similarity `{r.score:.3f}` | cluster: {cluster} | theme: {theme}"
        )

    lines += [
        "",
        "---",
        "*Generated deterministically — LLM not invoked. "
        "Rephrase your query for a conversational answer.*",
    ]
    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────────────────


def analyze(query: str, search_results: list) -> DeterministicResult:
    """
    Analyze Qdrant search results and return a DeterministicResult.

    If confidence >= cfg.analysis.confidence_threshold, ``result.answer``
    contains a structured Markdown answer and ``result.path`` is
    ``"deterministic"``.  Otherwise ``path`` is ``"llm_required"`` and
    the caller should fall through to the LLM layer.

    Args:
        query:          The original user query string.
        search_results: List of qdrant_client ScoredPoint objects.

    Returns:
        DeterministicResult
    """
    if not search_results:
        logger.warning("Deterministic analyzer: empty search results")
        return DeterministicResult(
            confidence=0.0,
            top_similarity=0.0,
            clusters=[],
            recurring_patterns=[],
            path="llm_required",
        )

    scores = [r.score for r in search_results]
    clusters = _detect_clusters(search_results)
    recurring = _detect_recurrence(search_results)
    confidence = _calculate_confidence(scores, clusters, recurring)

    threshold = cfg.analysis.confidence_threshold
    path = "deterministic" if confidence >= threshold else "llm_required"

    answer: Optional[str] = None
    if path == "deterministic":
        answer = _build_deterministic_answer(
            query, search_results, clusters, recurring, confidence
        )
        logger.info(
            "Deterministic path: confidence=%.3f (threshold=%.2f)", confidence, threshold
        )
    else:
        logger.info(
            "LLM required: confidence=%.3f < threshold=%.2f", confidence, threshold
        )

    return DeterministicResult(
        confidence=confidence,
        top_similarity=round(scores[0], 3),
        clusters=clusters,
        recurring_patterns=recurring,
        path=path,
        answer=answer,
        debug_info={
            "scores_top5": [round(s, 3) for s in scores[:5]],
            "n_clusters": len(clusters),
            "score_consistency": round(_score_consistency(scores), 3),
            "recurring_count": len(recurring),
            "threshold": threshold,
        },
    )
