"""
Spark Insight Agent — Phase 2 UI
Streamlit chat interface with deterministic intelligence metrics panel.
"""

from __future__ import annotations

import logging

import streamlit as st

from backend.cache_manager import cache_manager
from backend.config import cfg
from backend.core_qdrant import run_llm
from backend.evaluator import evaluator

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spark Insight Agent",
    page_icon="🔥",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"query": str, "response": dict}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _path_badge(path: str) -> str:
    badges = {
        "deterministic": "🟢 Deterministic (no LLM)",
        "llm": "🟡 LLM (GPT-4o-mini)",
        "query_cache": "⚡ Query Cache Hit",
    }
    return badges.get(path, path)


def _confidence_bar(confidence: float) -> str:
    filled = int(confidence * 10)
    return "█" * filled + "░" * (10 - filled) + f"  {confidence:.0%}"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔥 Spark Insight Agent")
    st.caption("Phase 2 — Deterministic Intelligence")
    st.divider()

    st.subheader("⚙️ Configuration")
    st.write(f"**LLM model:** `{cfg.openai.llm_model}`")
    st.write(f"**Embedding:** `{cfg.openai.embedding_model}`")
    st.write(f"**Confidence threshold:** `{cfg.analysis.confidence_threshold}`")
    st.write(f"**Qdrant collection:** `{cfg.qdrant.collection_name}`")

    st.divider()

    st.subheader("📊 Session Stats")
    summary = evaluator.get_session_summary()
    if "total_queries" in summary:
        col1, col2 = st.columns(2)
        col1.metric("Queries", summary["total_queries"])
        col2.metric("Total cost", f"${summary['total_cost_usd']:.4f}")
        col1.metric("Avg cost", f"${summary['avg_cost_usd']:.4f}")
        col2.metric("LLM avoided", f"{summary['llm_avoided_pct']}%")
        col1.metric("P50 latency", f"{summary['latency_p50_ms']:.0f}ms")
        col2.metric("P95 latency", f"{summary['latency_p95_ms']:.0f}ms")

        st.write("**Path distribution:**")
        for path, count in summary.get("path_distribution", {}).items():
            st.write(f"  • {_path_badge(path)}: {count}")

        cache_rates = summary.get("cache_hit_rates", {})
        st.write("**Cache hit rates:**")
        st.write(f"  • Query: {cache_rates.get('query', 0):.0%}")
        st.write(f"  • Embedding: {cache_rates.get('embedding', 0):.0%}")
        st.write(f"  • LLM: {cache_rates.get('llm', 0):.0%}")
    else:
        st.caption("No queries yet this session.")

    st.divider()

    if st.button("🗑️ Clear caches"):
        cache_manager.clear_all()
        st.success("All cache layers cleared.")

    st.divider()
    all_time = evaluator.get_all_time_summary()
    if "total_queries_all_time" in all_time:
        st.subheader("📈 All-Time Stats")
        st.write(f"Total queries: **{all_time['total_queries_all_time']}**")
        st.write(f"Total cost: **${all_time['total_cost_all_time_usd']:.4f}**")
        st.write(f"Avg per query: **${all_time['avg_cost_per_query_usd']:.4f}**")


# ── Main chat area ────────────────────────────────────────────────────────────

st.header("🔥 Spark Incident Analysis Assistant")
st.caption(
    "Ask about Spark incidents — root causes, resolutions, patterns. "
    "Powered by Qdrant + GPT-4o-mini with deterministic analysis layer."
)

# ── Replay chat history ───────────────────────────────────────────────────────

for item in st.session_state.chat_history:
    st.chat_message("user").write(item["query"])

    with st.chat_message("assistant"):
        resp = item["response"]
        st.markdown(resp["answer"])

        # Compact metrics row
        path = resp.get("path", "")
        confidence = resp.get("confidence", 0.0)
        cost = resp.get("cost_usd", 0.0)
        latency = resp.get("latency_ms", resp.get("response_time", 0) * 1000)
        cache_hit = resp.get("cache_hit", False)

        cols = st.columns(4)
        cols[0].metric("Path", _path_badge(path).split(" ", 1)[1] if " " in _path_badge(path) else path)
        cols[1].metric("Confidence", f"{confidence:.0%}")
        cols[2].metric("Cost", "$0.0000" if cache_hit else f"${cost:.5f}")
        cols[3].metric("Latency", f"{latency:.0f}ms")

        # Clusters & patterns
        clusters = resp.get("clusters", [])
        recurring = resp.get("recurring_patterns", [])

        if clusters:
            with st.expander(f"🔍 {len(clusters)} root-cause cluster(s) detected"):
                for c in clusters:
                    st.write(
                        f"**{c['theme']}** — {c['size']} incident(s), "
                        f"avg similarity `{c['avg_similarity']:.3f}`"
                    )
                    if c.get("incident_ids"):
                        st.caption("Incidents: " + ", ".join(c["incident_ids"]))

        if recurring:
            st.info(f"♻️ Recurring patterns: {', '.join(recurring)}")

        # Source incidents
        sources = resp.get("source_documents", [])
        if sources:
            with st.expander(f"📄 {len(sources)} source incident(s)"):
                for r in sources[:5]:
                    inc_id = r.payload.get("incident_id", "?")
                    cluster = r.payload.get("cluster", "?")
                    st.write(
                        f"**{inc_id}** — {cluster} — similarity `{r.score:.3f}`"
                    )
                    st.caption(r.payload.get("text", "")[:300] + "…")
                    st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────

prompt = st.chat_input("Ask about Spark incidents…")

if prompt:
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):
            try:
                resp = run_llm(query=prompt)
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

        st.markdown(resp["answer"])

        path = resp.get("path", "")
        confidence = resp.get("confidence", 0.0)
        cost = resp.get("cost_usd", 0.0)
        latency = resp.get("latency_ms", resp.get("response_time", 0) * 1000)
        cache_hit = resp.get("cache_hit", False)

        cols = st.columns(4)
        cols[0].metric("Path", _path_badge(path).split(" ", 1)[1] if " " in _path_badge(path) else path)
        cols[1].metric("Confidence", f"{confidence:.0%}")
        cols[2].metric("Cost", "$0.0000" if cache_hit else f"${cost:.5f}")
        cols[3].metric("Latency", f"{latency:.0f}ms")

        clusters = resp.get("clusters", [])
        recurring = resp.get("recurring_patterns", [])

        if clusters:
            with st.expander(f"🔍 {len(clusters)} root-cause cluster(s) detected"):
                for c in clusters:
                    st.write(
                        f"**{c['theme']}** — {c['size']} incident(s), "
                        f"avg similarity `{c['avg_similarity']:.3f}`"
                    )
                    if c.get("incident_ids"):
                        st.caption("Incidents: " + ", ".join(c["incident_ids"]))

        if recurring:
            st.info(f"♻️ Recurring patterns: {', '.join(recurring)}")

        sources = resp.get("source_documents", [])
        if sources:
            with st.expander(f"📄 {len(sources)} source incident(s)"):
                for r in sources[:5]:
                    inc_id = r.payload.get("incident_id", "?")
                    cluster = r.payload.get("cluster", "?")
                    st.write(
                        f"**{inc_id}** — {cluster} — similarity `{r.score:.3f}`"
                    )
                    st.caption(r.payload.get("text", "")[:300] + "…")
                    st.divider()

    # Store in history
    st.session_state.chat_history.append({"query": prompt, "response": resp})

    # Trigger sidebar rerun to update stats
    st.rerun()
