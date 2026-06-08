"""Streamlit UI for the AuraComply RAG pipeline."""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8001")
DEFAULT_QUESTIONS = [
    "What are the obligations for high-risk AI systems?",
    "Which AI practices are prohibited under the EU AI Act?",
    "What transparency obligations apply to general-purpose AI models?",
]

st.set_page_config(page_title="AuraComply RAG", page_icon="⚖", layout="wide")
st.title("AuraComply RAG — EU AI Act Compliance")
st.caption(
    "LangChain LCEL · ChromaDB · Anthropic Claude · sealed into the AuraComply audit chain."
)

with st.sidebar:
    st.subheader("Backend")
    st.code(API_URL, language="text")
    st.subheader("Example questions")
    for q in DEFAULT_QUESTIONS:
        if st.button(q, key=f"ex-{hash(q)}"):
            st.session_state["question"] = q

question = st.text_area(
    "Ask a compliance question",
    value=st.session_state.get("question", DEFAULT_QUESTIONS[0]),
    height=110,
)
k = st.slider("Retrieved chunks (k)", min_value=1, max_value=10, value=3)

if st.button("Run query", type="primary", disabled=not question.strip()):
    with st.spinner("Querying RAG pipeline..."):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": question, "k": k},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            st.stop()

    st.subheader("Answer")
    st.markdown(data["answer"])

    if data.get("articles_cited"):
        st.markdown(
            "**Articles cited:** "
            + " · ".join(f"Article {a}" for a in data["articles_cited"])
        )

    st.subheader("Retrieved sources")
    for src in data["sources"]:
        title = src.get("title") or f"Article {src.get('article_number')}"
        with st.expander(f"Article {src.get('article_number')} — {title}"):
            st.markdown(f"**Chapter:** {src.get('chapter')}")
            st.markdown(src.get("snippet", ""))

    st.subheader("AuraComply seal")
    seal = data.get("seal", {})
    if seal.get("ok"):
        st.success("Decision sealed into the AuraComply audit chain.")
        st.json(seal.get("seal", {}))
    else:
        st.warning(
            "Seal failed — query returned, but the audit record was not written. "
            f"Reason: {seal.get('error', 'unknown')}"
        )
