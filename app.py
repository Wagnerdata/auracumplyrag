"""Streamlit UI for the AuraComply RAG pipeline.

The pure helpers (auth headers, seal classification) live at module
level so tests can import them without rendering the page; the page
itself only renders under `streamlit run app.py`.
"""
from __future__ import annotations

import os
from typing import Any

import requests

API_URL = os.environ.get("API_URL", "http://localhost:8001")
DEFAULT_QUESTIONS = [
    "What are the obligations for high-risk AI systems?",
    "Which AI practices are prohibited under the EU AI Act?",
    "What transparency obligations apply to general-purpose AI models?",
]


# ── Helpers ──────────────────────────────────────────────────────────
def auth_headers() -> dict[str, str]:
    """Bearer for /query, read from RAG_API_TOKEN at call time. Empty
    when unset — the API then answers 401 unless it runs with
    AURACOMPLY_DEMO_PROFILE=1."""
    token = os.environ.get("RAG_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def classify_seal(seal: Any) -> tuple[str, str]:
    """Map the API's `seal` envelope to (level, message).

    Only a kernel-confirmed seal (an `event_hash` present) is reported
    as success. A buffered seal is queued in the SDK's local buffer but
    not yet confirmed by the kernel, so it is `info`, never success.
    Anything missing or inconsistent is a `warning`.
    """
    if not isinstance(seal, dict) or not seal.get("line_id"):
        return (
            "warning",
            "No seal information in the API response — this answer has "
            "no audit record.",
        )
    if seal.get("event_hash"):
        msg = "Decision sealed into the AuraComply audit chain."
        if seal.get("replayed"):
            msg += " (Idempotent replay of an earlier identical seal.)"
        return "success", msg
    if seal.get("buffered"):
        return (
            "info",
            "Seal queued in the local SDK buffer — not yet confirmed by "
            "the AuraComply kernel. The background flusher will deliver it.",
        )
    return (
        "warning",
        "Seal not confirmed: the API returned no event_hash and the seal "
        "is not buffered.",
    )


def describe_http_error(status_code: int) -> str:
    if status_code == 401:
        return "API rejected the request: missing bearer. Set RAG_API_TOKEN for the UI."
    if status_code == 403:
        return "API rejected the request: RAG_API_TOKEN does not match the server's."
    if status_code == 503:
        return "API is misconfigured: the server has no RAG_API_TOKEN set."
    return f"API returned HTTP {status_code}."


# ── Page ─────────────────────────────────────────────────────────────
def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="AuraComply RAG", page_icon="⚖", layout="wide")
    st.title("AuraComply RAG — EU AI Act Compliance")
    st.caption(
        "LangChain LCEL · ChromaDB · Anthropic Claude · sealed into the AuraComply audit chain."
    )

    with st.sidebar:
        st.subheader("Backend")
        st.code(API_URL, language="text")
        if not auth_headers():
            st.warning("RAG_API_TOKEN is not set — /query will be rejected with 401.")
        st.subheader("Example questions")
        for q in DEFAULT_QUESTIONS:
            if st.button(q, key=f"ex-{hash(q)}"):
                st.session_state["question"] = q

    question = st.text_area(
        "Ask a compliance question",
        value=st.session_state.get("question", DEFAULT_QUESTIONS[0]),
        height=110,
    )
    k = st.slider("Hybrid retrieval depth (k)", min_value=1, max_value=20, value=6)

    if st.button("Run query", type="primary", disabled=not question.strip()):
        with st.spinner("Querying RAG pipeline..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"question": question, "k": k},
                    headers=auth_headers(),
                    timeout=120,
                )
            except requests.RequestException as exc:
                st.error(f"Query failed: {exc}")
                st.stop()
            if not resp.ok:
                st.error(describe_http_error(resp.status_code))
                st.stop()
            data = resp.json()

        st.subheader("Answer")
        st.markdown(data["answer"])

        if data.get("articles_cited"):
            st.markdown(
                "**Articles cited:** "
                + " · ".join(f"Article {a}" for a in data["articles_cited"])
            )

        groundedness = data.get("groundedness") or {}
        if groundedness:
            st.markdown(
                f"**Groundedness:** {groundedness.get('verdict', '?')} "
                f"(confidence {groundedness.get('confidence', 0):.2f}) — "
                f"{groundedness.get('reason', '')}"
            )

        st.subheader("Retrieved sources")
        for src in data["sources"]:
            title = src.get("title") or f"Article {src.get('article_number')}"
            with st.expander(f"Article {src.get('article_number')} — {title}"):
                st.markdown(f"**Chapter:** {src.get('chapter')}")
                st.markdown(src.get("snippet", ""))

        st.subheader("AuraComply seal")
        seal = data.get("seal")
        level, message = classify_seal(seal)
        getattr(st, level)(message)
        if isinstance(seal, dict) and seal:
            st.json(seal)


if __name__ == "__main__":
    main()
