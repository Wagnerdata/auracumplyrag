"""RAG pipeline: hybrid retrieval (BM25 + dense + RRF) → bge-reranker →
LLM answer → groundedness verifier → agent_decision/v1 seal.

Public entry point is `query(question, k, seal_client)`. The seal is
performed by the kernel SDK (`seal_client.seal(...)`); we never call
/audit/log directly — see CLAUDE.md §R1.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from langchain_anthropic import ChatAnthropic
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import settings
from ingest import get_vector_store, read_corpus_version

if TYPE_CHECKING:
    from auracomply import AuracomplyClient, SealResult


logger = logging.getLogger(__name__)


# ── Prompts ──────────────────────────────────────────────────────────
ANSWER_SYSTEM = """You are an EU AI Act compliance assistant.

Answer ONLY using the provided context. Every factual statement must
cite the relevant article inline using the format (Article XX), recital
((Recital N)), or annex ((Annex III)). If the context does not contain
enough information, say so explicitly and do not invent identifiers.

Context:
{context}
"""
ANSWER_USER = "Question: {question}"

GROUNDEDNESS_SYSTEM = """You are a strict groundedness checker. Given a
QUESTION, an ANSWER, and a list of CHUNKS the answer was supposed to be
grounded in, decide whether every factual claim in the ANSWER is
supported by at least one CHUNK.

Reply with a single JSON object on one line:
{{"verdict": "<yes|partial|no>", "confidence": <float between 0 and 1>, "reason": "<one short sentence>"}}
"""
GROUNDEDNESS_USER = """QUESTION: {question}
ANSWER: {answer}

CHUNKS:
{chunks}
"""


# ── Public data types ────────────────────────────────────────────────
@dataclass
class RagResult:
    """Returned by `query()`.

    `answer` is the LLM's response. `sources` and `articles_cited`
    mirror the structures the existing Streamlit UI already renders.
    `groundedness` is the verifier verdict. `seal` is the SDK's
    SealResult — pass-through to the API response.
    """
    answer: str
    sources: list[dict[str, Any]]
    articles_cited: list[str]
    groundedness: dict[str, Any]
    seal: "SealResult"


# ── Retrieval ────────────────────────────────────────────────────────
def _all_documents() -> list[Document]:
    """Pull every chunk out of Chroma so we can build the BM25 index in
    memory. Chroma's collection is already persisted on disk; BM25 has
    no native vector store, so rebuilding the index per process is the
    canonical pattern in langchain-community."""
    store = get_vector_store()
    raw = store.get(include=["documents", "metadatas"])
    docs: list[Document] = []
    for content, meta in zip(raw.get("documents") or [], raw.get("metadatas") or []):
        docs.append(Document(page_content=content, metadata=meta or {}))
    return docs


def _build_hybrid_retriever(k: int):
    """BM25 + Chroma dense, fused by Reciprocal Rank Fusion via
    `EnsembleRetriever`. Weights 0.5/0.5 — BM25 is the recall floor for
    exact regulatory terms (article numbers, defined acronyms); the
    dense retriever brings in paraphrases."""
    dense = get_vector_store().as_retriever(search_kwargs={"k": k})
    docs = _all_documents()
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])


def _rerank(question: str, docs: list[Document], top_k: int) -> list[Document]:
    """Rerank with a local bge cross-encoder.

    The model is loaded lazily and cached at module scope — a per-query
    instantiation would tank latency. The first call pays the model
    download/load; subsequent calls are sub-second.
    """
    if not docs:
        return []
    reranker = _get_reranker()
    pairs = [(question, d.page_content) for d in docs]
    scores = reranker.compute_score(pairs, normalize=True)
    # Some FlagEmbedding versions return a single float for a single
    # pair — normalise to a list either way.
    if isinstance(scores, float):
        scores = [scores]
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


_reranker_singleton = None


def _get_reranker():
    global _reranker_singleton
    if _reranker_singleton is None:
        from FlagEmbedding import FlagReranker

        _reranker_singleton = FlagReranker(settings.reranker_model, use_fp16=False)
    return _reranker_singleton


def set_reranker(fake) -> None:
    """Test seam: inject a fake reranker. The fake must expose
    `compute_score(pairs, normalize=...)`. Tests reset to None via
    `set_reranker(None)`."""
    global _reranker_singleton
    _reranker_singleton = fake


# ── Formatting helpers ───────────────────────────────────────────────
def _format_context(docs: list[Document]) -> str:
    parts: list[str] = []
    for doc in docs:
        kind = doc.metadata.get("chunk_kind", "article")
        if kind == "recital":
            header = f"[Recital ({doc.metadata.get('article_number', '?').replace('rec-', '')})]"
        elif kind == "annex":
            annex_id = doc.metadata.get("article_number", "?").replace("annex-", "").upper()
            header = f"[Annex {annex_id} — {doc.metadata.get('title', '')}]"
        else:
            article = doc.metadata.get("article_number", "?")
            title = doc.metadata.get("title", "")
            chapter = doc.metadata.get("chapter", "?")
            header = f"[Article {article} | Chapter {chapter}] {title}".strip()
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _sources_from_docs(docs: list[Document]) -> list[dict[str, Any]]:
    return [
        {
            "article_number": d.metadata.get("article_number"),
            "title": d.metadata.get("title"),
            "chapter": d.metadata.get("chapter"),
            "chunk_kind": d.metadata.get("chunk_kind"),
            "snippet": d.page_content[:300],
        }
        for d in docs
    ]


# Identifier regexes for "Article 9", "(Recital 12)", "Annex III".
_ID_PATTERNS = [
    re.compile(r"Article\s+(\d+)", re.IGNORECASE),
    re.compile(r"Recital\s+\(?(\d+)\)?", re.IGNORECASE),
    re.compile(r"Annex\s+(III|IV|I|II|V|VI|VII|VIII|IX|X)", re.IGNORECASE),
]


def _articles_from_docs(docs: list[Document]) -> list[str]:
    """Pull a stable, ordered, deduplicated list of identifiers from
    the retrieved chunks' metadata (NOT the LLM answer text — for the
    answer-text version see `_citations_in_answer`)."""
    seen: list[str] = []
    for d in docs:
        ident = str(d.metadata.get("article_number", ""))
        if ident and ident not in seen:
            seen.append(ident)
    return seen


def _citations_in_answer(answer: str) -> list[str]:
    """Identifiers the LLM actually cited inline. Used for eval
    metrics (precision/recall) without confusing them with what was
    retrieved."""
    found: list[str] = []
    for pat in _ID_PATTERNS:
        for m in pat.finditer(answer):
            ident = m.group(1)
            if ident and ident not in found:
                found.append(ident)
    return found


# ── LLM construction ─────────────────────────────────────────────────
def build_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.anthropic_model,
        anthropic_api_key=settings.anthropic_api_key,
        max_tokens=1024,
        timeout=60,
    )


_llm_override = None


def set_llm(fake) -> None:
    """Test seam: inject a fake llm. Must expose `.invoke(messages)`
    returning an object with `.content`. Pass None to reset."""
    global _llm_override
    _llm_override = fake


def _get_llm():
    return _llm_override or build_llm()


# ── Groundedness verifier ────────────────────────────────────────────
def _check_groundedness(
    question: str, answer: str, docs: list[Document], llm
) -> dict[str, Any]:
    chunks_blob = "\n\n".join(
        f"- {d.metadata.get('article_number', '?')}: {d.page_content[:400]}"
        for d in docs
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", GROUNDEDNESS_SYSTEM), ("human", GROUNDEDNESS_USER)]
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"question": question, "answer": answer, "chunks": chunks_blob})
    return _parse_groundedness(raw)


def _parse_groundedness(raw: str) -> dict[str, Any]:
    """Extract the JSON object the verifier was told to emit. Fail
    safe: if parsing fails, return a `partial` verdict with a clear
    reason — never reject the whole query path on a verifier hiccup."""
    raw = (raw or "").strip()
    # Strip markdown code fences if the model wrapped them.
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)
    try:
        obj = json.loads(raw)
        verdict = str(obj.get("verdict", "partial")).lower()
        if verdict not in {"yes", "partial", "no"}:
            verdict = "partial"
        confidence = float(obj.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        reason = str(obj.get("reason", ""))[:240] or "no reason given"
        return {"verdict": verdict, "confidence": confidence, "reason": reason}
    except (ValueError, TypeError):
        return {
            "verdict": "partial",
            "confidence": 0.5,
            "reason": "groundedness verifier returned unparseable JSON",
        }


# ── Sealing ──────────────────────────────────────────────────────────
def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_agent_decision_payload(
    *,
    question: str,
    answer: str,
    docs: list[Document],
    articles_cited: list[str],
    groundedness: dict[str, Any],
    corpus_version: dict,
) -> dict[str, Any]:
    """Construct the `agent_decision/v1` payload.

    - `trace_id` is `rag:<sha256(question)>` — a 68-char string that
      lets an inspector tie the chain row to a specific question
      without ever sealing the question itself.
    - `input_refs.doc_hashes` carries the source document's SHA-256
      (one element — the EU AI Act PDF). `chunk_hashes` carries one
      SHA-256 per retrieved chunk.
    - `decision.rationale_summary` is the only human-readable field;
      it summarises the answer and the groundedness verdict and is
      capped at 2000 chars. The kernel's PII boundary runs on it.
    - `human_oversight` is `NOT_REQUIRED` for this demo — there is no
      human-in-the-loop. A production deployment using HITL would
      pair each decision with a `human_oversight_approval/v1`.
    """
    chunk_hashes = [_sha256_text(d.page_content) for d in docs]
    rationale = _rationale_summary(answer, groundedness, articles_cited)

    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "agent_decision/v1",
        "trace_id": f"rag:{_sha256_text(question)}",
        "graph_node": None,
        "graph_path": [],
        "model": {
            "provider": "anthropic",
            "id": settings.anthropic_model[:128],
            "version": settings.anthropic_model[:128],
            "effort": None,
        },
        "corpus_version": {
            "source_doc_sha256": corpus_version["source_doc_sha256"],
            "ingested_at": corpus_version["ingested_at"],
        },
        "input_refs": {
            "doc_hashes": [corpus_version["source_doc_sha256"]],
            "chunk_hashes": chunk_hashes,
        },
        "decision": {
            "outcome": "answered",
            "confidence": groundedness["confidence"],
            "rationale_summary": rationale,
        },
        "human_oversight": {
            "required": False,
            "status": "not_required",
            "approver_ref": None,
        },
        "data_categories": ["regulatory_text"],
        "timestamp": now,
        "time_source": "wall_clock",
        # TODO Phase 3.1: include a crypto-shredding blob carrying
        # the ciphertext of (question, answer) so a future erasure
        # request can wipe the plaintext while the chain row stays
        # cryptographically intact. Blocked on the kernel module.
    }


def _rationale_summary(
    answer: str, groundedness: dict[str, Any], articles_cited: list[str]
) -> str:
    prefix = (
        f"Grounded: {groundedness['verdict']} "
        f"(conf {groundedness['confidence']:.2f}). "
        f"Cited: {', '.join(articles_cited) or 'none'}. "
        f"Verifier: {groundedness['reason']}. "
        f"Answer: "
    )
    budget = 2000 - len(prefix)
    if budget <= 0:
        return prefix[:2000]
    body = answer.strip().replace("\n", " ")
    if len(body) > budget:
        body = body[: budget - 1] + "…"
    return prefix + body


# ── Public entry point ───────────────────────────────────────────────
def query(
    question: str,
    *,
    k: int = 6,
    rerank_top_k: int = 3,
    seal_client: "AuracomplyClient | None" = None,
) -> RagResult:
    """Run the hybrid → rerank → answer → verify → seal pipeline.

    `k` is the hybrid retrieval depth before rerank; `rerank_top_k` is
    the number of chunks the LLM sees. Defaults reflect a Recall@K vs.
    prompt-window trade-off chosen for the EU AI Act corpus.
    """
    if seal_client is None:
        from seal_client import get_seal_client

        seal_client = get_seal_client()

    corpus_version = read_corpus_version()
    if corpus_version is None:
        raise RuntimeError(
            "No corpus_version persisted yet — run POST /ingest before "
            "querying so an agent_decision/v1 can be sealed against a "
            "known knowledge state."
        )

    retriever = _build_hybrid_retriever(k=k)
    fused = retriever.invoke(question)
    reranked = _rerank(question, fused, top_k=rerank_top_k)

    llm = _get_llm()
    answer_chain = (
        ChatPromptTemplate.from_messages(
            [("system", ANSWER_SYSTEM), ("human", ANSWER_USER)]
        )
        | llm
        | StrOutputParser()
    )
    answer = answer_chain.invoke(
        {"context": _format_context(reranked), "question": question}
    )

    groundedness = _check_groundedness(question, answer, reranked, llm)
    sources = _sources_from_docs(reranked)
    articles_cited = _citations_in_answer(answer) or _articles_from_docs(reranked)

    payload = _build_agent_decision_payload(
        question=question,
        answer=answer,
        docs=reranked,
        articles_cited=articles_cited,
        groundedness=groundedness,
        corpus_version=corpus_version,
    )
    seal = seal_client.seal(
        product="rag-eu-ai-act",
        step="agent_decision",
        payload=payload,
        idempotency_key=f"rag:{_sha256_text(question)}",
    )

    return RagResult(
        answer=answer,
        sources=sources,
        articles_cited=articles_cited,
        groundedness=groundedness,
        seal=seal,
    )


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What are high-risk AI systems?"
    out = query(q)
    print(f"Q: {q}\n\nA: {out.answer}\n")
    print(f"Articles cited: {out.articles_cited}")
    print(f"Groundedness: {out.groundedness}")
    print(f"Seal: line_id={out.seal.line_id} buffered={out.seal.buffered}")
