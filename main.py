"""
Spark Insight Agent — Phase 2 UI
Streamlit chat interface with deterministic intelligence metrics panel.
"""

from __future__ import annotations

import logging
import uuid

import streamlit as st

from backend.cache_manager import cache_manager
from backend.config import cfg
from backend.core_qdrant import run_llm
from backend.evaluator import evaluator
from backend.feedback_manager import feedback_manager

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
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()  # set of chat_history indices that got a rating
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = ""  # populated by login widget below


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

    st.subheader("👤 User")
    user_name = st.text_input(
        "Your name / ID",
        value=st.session_state.user_id,
        placeholder="e.g. john.doe",
        help="Logged alongside each query for audit trail.",
    )
    if user_name != st.session_state.user_id:
        st.session_state.user_id = user_name
    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

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

    cache_col1, cache_col2 = st.columns(2)
    if cache_col1.button("🗑️ Clear all"):
        cache_manager.clear_all()
        st.success("All cache layers cleared.")
    if cache_col2.button("🧹 Purge expired"):
        evicted = cache_manager.evict_all_expired()
        st.success(f"Purged {evicted} expired entries.")

    st.divider()

    st.subheader("👍 Feedback Stats")
    fb_stats = feedback_manager.get_stats()
    if fb_stats["total"] > 0:
        fb_col1, fb_col2 = st.columns(2)
        fb_col1.metric("Total ratings", fb_stats["total"])
        fb_col2.metric("Thumbs up", f"{fb_stats['thumbs_up_pct']}%")
        st.progress(fb_stats["thumbs_up_pct"] / 100)
    else:
        st.caption("No feedback yet.")

    # ── Business Impact (session) ────────────────────────────────────────────
    if "total_time_saved_mins" in summary:
        st.divider()
        st.subheader("💡 Business Impact (Session)")
        imp_col1, imp_col2 = st.columns(2)
        imp_col1.metric(
            "Time saved",
            f"{summary['total_time_saved_mins']:.0f} min",
            help=f"vs. {summary['manual_baseline_mins']:.0f} min manual investigation per query",
        )
        imp_col2.metric("Avg response", f"{summary['avg_response_secs']:.1f}s")
        imp_col1.metric("Users", summary.get("unique_users", 0))
        imp_col2.metric("Sessions", summary.get("unique_sessions", 0))
        low_sim = summary.get("low_similarity_count", 0)
        if low_sim:
            st.caption(
                f"⚠️ {low_sim} / {summary['total_queries']} "
                f"queries had low-similarity warnings"
            )

    st.divider()
    all_time = evaluator.get_all_time_summary()
    if "total_queries_all_time" in all_time:
        st.subheader("📈 All-Time Stats")
        st.write(f"Total queries: **{all_time['total_queries_all_time']}**")
        st.write(f"Total cost: **${all_time['total_cost_all_time_usd']:.4f}**")
        st.write(f"Avg per query: **${all_time['avg_cost_per_query_usd']:.4f}**")
        time_hrs = all_time.get("total_time_saved_all_time_hrs", 0)
        if time_hrs:
            st.write(f"Total time saved: **{time_hrs:.1f} hours**")
        users_all = all_time.get("unique_users_all_time", 0)
        if users_all:
            st.write(f"Unique users: **{users_all}**")


# ── Main chat area ────────────────────────────────────────────────────────────

st.header("🔥 Spark Incident Analysis Assistant")
st.caption(
    "Ask about Spark incidents — root causes, resolutions, patterns. "
    "Powered by Qdrant + GPT-4o-mini with deterministic analysis layer."
)

# ── Replay chat history ───────────────────────────────────────────────────────

for i, item in enumerate(st.session_state.chat_history):
    st.chat_message("user").write(item["query"])

    with st.chat_message("assistant"):
        resp = item["response"]
        if resp.get("low_similarity_warning"):
            st.warning(
                "This query had no strong match in the incident database. "
                "The answer below may not be reliable.",
                icon="⚠️",
            )
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

        # Feedback buttons
        if i not in st.session_state.feedback_given:
            st.write("")
            fb_cols = st.columns([1, 1, 8])
            if fb_cols[0].button("👍", key=f"up_{i}", help="Helpful response"):
                feedback_manager.save_feedback(
                    query=item["query"],
                    answer=resp["answer"],
                    rating="thumbs_up",
                    confidence_score=resp.get("confidence"),
                    path_taken=resp.get("path"),
                )
                st.session_state.feedback_given.add(i)
                st.rerun()
            if fb_cols[1].button("👎", key=f"down_{i}", help="Not helpful"):
                feedback_manager.save_feedback(
                    query=item["query"],
                    answer=resp["answer"],
                    rating="thumbs_down",
                    confidence_score=resp.get("confidence"),
                    path_taken=resp.get("path"),
                )
                st.session_state.feedback_given.add(i)
                st.rerun()
        else:
            st.caption("✓ Feedback recorded")

# ── Input ─────────────────────────────────────────────────────────────────────

prompt = st.chat_input("Ask about Spark incidents…")

if prompt:
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):
            try:
                resp = run_llm(
                    query=prompt,
                    session_id=st.session_state.session_id,
                    user_id=st.session_state.user_id,
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

        if resp.get("low_similarity_warning"):
            st.warning(
                "This query had no strong match in the incident database. "
                "The answer below may not be reliable.",
                icon="⚠️",
            )
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
