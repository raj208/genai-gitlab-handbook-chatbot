"""
Streamlit chat UI for the GenAI GitLab Handbook Chatbot.

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from src.config import settings

if TYPE_CHECKING:
    from src.rag import RagPipeline, Source


# ── Paths ──────────────────────────────────────────────────────────────────
CSS_PATH = Path(__file__).parent / "assets" / "style.css"


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GitLab Handbook Chatbot",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Inject custom CSS ─────────────────────────────────────────────────────
def _inject_css() -> None:
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)
    # Additional inline CSS for elements hard to target from external file
    st.markdown("""
    <style>
    /* Hero header gradient text */
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #A78BFA 0%, #7C3AED 40%, #6D28D9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.15rem;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.92rem;
        font-weight: 400;
        margin-bottom: 1rem;
    }
    /* Disclaimer banner */
    .disclaimer-bar {
        background: rgba(245, 158, 11, 0.06);
        border: 1px solid rgba(245, 158, 11, 0.15);
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-size: 0.82rem;
        color: #CBD5E1;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .disclaimer-bar .icon { font-size: 1rem; }
    /* Starter prompt cards */
    .starter-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.65rem;
        margin: 1rem 0 1.5rem;
    }
    .starter-card {
        background: #16162A;
        border: 1px solid rgba(148, 163, 184, 0.08);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        cursor: default;
        transition: all 0.25s ease;
        font-size: 0.88rem;
        color: #CBD5E1;
        line-height: 1.4;
    }
    .starter-card:hover {
        border-color: rgba(124, 58, 237, 0.35);
        background: rgba(124, 58, 237, 0.06);
        transform: translateY(-1px);
    }
    .starter-card .label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #7C3AED;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    /* Sidebar brand */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.08);
    }
    .sidebar-brand .logo {
        font-size: 1.8rem;
        line-height: 1;
    }
    .sidebar-brand .name {
        font-weight: 700;
        font-size: 1rem;
        color: #F1F5F9;
        letter-spacing: -0.02em;
    }
    .sidebar-brand .version {
        font-size: 0.7rem;
        color: #64748B;
        font-weight: 400;
    }
    /* Sidebar stat pills */
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0;
        font-size: 0.82rem;
    }
    .stat-label { color: #94A3B8; }
    .stat-value {
        color: #A78BFA;
        font-weight: 600;
        font-family: 'Inter', monospace;
    }
    /* Source item in expander */
    .src-item {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.4rem;
        background: rgba(124, 58, 237, 0.04);
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.05);
        transition: border-color 0.2s;
    }
    .src-item:hover { border-color: rgba(124, 58, 237, 0.25); }
    .src-badge {
        font-size: 0.7rem;
        font-weight: 700;
        background: rgba(124, 58, 237, 0.15);
        color: #A78BFA;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        white-space: nowrap;
        margin-top: 0.1rem;
    }
    .src-meta {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.15rem;
    }
    .src-title a {
        color: #E2E8F0 !important;
        text-decoration: none;
        font-weight: 500;
        font-size: 0.85rem;
    }
    .src-title a:hover { color: #A78BFA !important; }
    /* Confidence indicator */
    .confidence-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        margin-bottom: 0.4rem;
    }
    .confidence-high {
        background: rgba(16, 185, 129, 0.1);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .confidence-low {
        background: rgba(245, 158, 11, 0.1);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    /* Footer */
    .app-footer {
        text-align: center;
        color: #475569;
        font-size: 0.72rem;
        padding: 2rem 0 0.5rem;
        border-top: 1px solid rgba(148, 163, 184, 0.06);
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


_inject_css()


# ── Cached resources ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading retrieval index and pipeline...")
def get_pipeline() -> RagPipeline:
    """Build the pipeline once per session (cached across reruns)."""
    from src.rag import RagPipeline

    return RagPipeline()


def _load_index_meta() -> dict | None:
    meta_path = settings.index_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def _index_mtime() -> str:
    p = settings.index_dir / "faiss.index"
    if not p.exists():
        return "missing"
    ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def _index_ready() -> bool:
    return (
        (settings.index_dir / "faiss.index").exists()
        and (settings.index_dir / "chunks.jsonl").exists()
    )


# ── Source rendering ───────────────────────────────────────────────────────

def _render_sources(sources: list[Source], confidence: str = "high", top_score: float = 0.0) -> None:
    # Confidence indicator
    if confidence == "low_confidence":
        st.markdown(
            '<div class="confidence-indicator confidence-low">'
            f'⚠ Low confidence · similarity {top_score:.2f}'
            '</div>',
            unsafe_allow_html=True,
        )
    elif confidence == "high" and top_score > 0:
        st.markdown(
            '<div class="confidence-indicator confidence-high">'
            f'✓ High confidence · similarity {top_score:.2f}'
            '</div>',
            unsafe_allow_html=True,
        )

    if not sources:
        return

    with st.expander(f"📚 View sources ({len(sources)})", expanded=False):
        for s in sources:
            badge_text = "Handbook" if s.source == "handbook" else "Direction"
            st.markdown(
                f'<div class="src-item">'
                f'  <span class="src-badge">[{s.n}]</span>'
                f'  <div>'
                f'    <div class="src-title"><a href="{s.url}" target="_blank">{s.title or s.url}</a></div>'
                f'    <div class="src-meta">{badge_text} · similarity {s.best_score:.2f}</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Sidebar ────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        # Brand header
        st.markdown(
            '<div class="sidebar-brand">'
            '  <span class="logo">📘</span>'
            '  <div>'
            '    <div class="name">GitLab Handbook</div>'
            '    <div class="version">RAG Chatbot v1.0</div>'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            "An **unofficial** assistant that answers questions using "
            "publicly available content from GitLab's "
            "[Handbook](https://handbook.gitlab.com/handbook/) and "
            "[Direction](https://about.gitlab.com/direction/) pages."
        )

        st.divider()

        # Index stats
        st.markdown("##### 📊 Index Status")
        meta = _load_index_meta()
        if meta is None:
            st.error("No index found. Run `python -m src.build_index --rebuild`.")
        else:
            pages = meta.get("num_pages", "?")
            chunks = meta.get("num_chunks", "?")
            model = meta.get("embedding_model", "?")
            built = _index_mtime()

            st.markdown(
                f'<div class="stat-row"><span class="stat-label">Pages indexed</span>'
                f'<span class="stat-value">{pages}</span></div>'
                f'<div class="stat-row"><span class="stat-label">Chunks</span>'
                f'<span class="stat-value">{chunks}</span></div>'
                f'<div class="stat-row"><span class="stat-label">Embedding model</span>'
                f'<span class="stat-value">{model}</span></div>'
                f'<div class="stat-row"><span class="stat-label">Built</span>'
                f'<span class="stat-value">{built}</span></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        st.markdown("##### ⚙️ Configuration")
        st.markdown(
            f'<div class="stat-row"><span class="stat-label">Chat model</span>'
            f'<span class="stat-value">{settings.chat_model}</span></div>'
            f'<div class="stat-row"><span class="stat-label">Top-K retrieval</span>'
            f'<span class="stat-value">{settings.top_k}</span></div>'
            f'<div class="stat-row"><span class="stat-label">Sim. threshold</span>'
            f'<span class="stat-value">{settings.similarity_threshold}</span></div>',
            unsafe_allow_html=True,
        )

        st.divider()

        if st.button("🗑️  Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption(
            "Built as a portfolio project. Answers are generated by an LLM and may "
            "contain inaccuracies — always verify against the linked sources."
        )


# ── Chat loop ──────────────────────────────────────────────────────────────

def _ensure_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict] = []


def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(
                    msg["sources"],
                    confidence=msg.get("confidence", "high"),
                    top_score=msg.get("top_score", 0.0),
                )


def _handle_user_turn(pipe: RagPipeline, user_input: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        sources_holder: list[Source] = []
        top_score = 0.0
        confidence = "high"
        answer_text = ""

        try:
            stream = pipe.answer_stream(user_input)
            placeholder = st.empty()

            for item in stream:
                # Refusal short-circuit
                if isinstance(item, dict) and item.get("type") == "refusal":
                    answer_text = item["answer"]
                    placeholder.markdown(answer_text)
                    break

                # Metadata header — emitted before first token
                if isinstance(item, dict) and item.get("type") == "meta":
                    sources_holder = item["sources"]
                    top_score = item["top_score"]
                    confidence = item.get("confidence", "high")
                    continue

                if isinstance(item, dict) and item.get("type") == "done":
                    break

                # Token delta
                if isinstance(item, str):
                    answer_text += item
                    placeholder.markdown(answer_text + "▌")

            placeholder.markdown(answer_text)
            _render_sources(sources_holder, confidence=confidence, top_score=top_score)

        except Exception as e:
            error_msg = (
                "Something went wrong while generating the answer. "
                "This is usually a temporary API issue — please try again in a moment."
            )
            st.error(error_msg)
            # Show the underlying error in an expander, useful for debugging
            with st.expander("Error details (for debugging)"):
                st.code(f"{type(e).__name__}: {e}")
            answer_text = error_msg
            sources_holder = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources_holder,
        "top_score": top_score,
        "confidence": confidence,
    })


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    _ensure_state()
    _render_sidebar()

    # Hero header
    st.markdown('<div class="hero-title">📘 GitLab Handbook Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        'Ask anything about GitLab\'s Handbook or product direction — '
        'every answer cites its sources so you can verify.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Disclaimer
    st.markdown(
        '<div class="disclaimer-bar">'
        '<span class="icon">⚠️</span>'
        '<span>Unofficial assistant. Answers are AI-generated from publicly available '
        'GitLab content and may contain inaccuracies. Always verify against cited sources.</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Starter prompts when chat is empty
    if not st.session_state.messages:
        st.markdown(
            '<div class="starter-grid">'
            '  <div class="starter-card">'
            '    <div class="label">Values</div>'
            '    What are GitLab\'s core values?'
            '  </div>'
            '  <div class="starter-card">'
            '    <div class="label">Communication</div>'
            '    How does GitLab handle async communication?'
            '  </div>'
            '  <div class="starter-card">'
            '    <div class="label">Hiring</div>'
            '    What is GitLab\'s hiring process like?'
            '  </div>'
            '  <div class="starter-card">'
            '    <div class="label">Remote work</div>'
            '    How does GitLab approach remote work?'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )

    _render_history()

    user_input = st.chat_input("Ask a question about GitLab's handbook…")
    if user_input:
        if not _index_ready():
            st.error("No retrieval index found. Run `python -m src.build_index --rebuild` first.")
            return

        with st.spinner("Loading retrieval index..."):
            pipe = get_pipeline()
        _handle_user_turn(pipe, user_input)

    # Footer
    if not st.session_state.messages:
        st.markdown(
            '<div class="app-footer">'
            'Powered by OpenAI · FAISS · Streamlit &nbsp;|&nbsp; '
            'Data sourced from GitLab\'s public Handbook &amp; Direction pages'
            '</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
