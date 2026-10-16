"""End-to-end shape of the payloads the RAG layer seals via the SDK.

These tests fake the LLM and the reranker, and pre-stage a tiny
corpus_version.json so query() can run without ingest. They focus on
ONE thing each: what the sealed payload looks like.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from langchain_core.documents import Document


# ── Helpers ──────────────────────────────────────────────────────────
def _stage_corpus_version(persist_dir: str) -> dict:
    """Write a corpus_version.json next to the Chroma persist dir so
    query() doesn't refuse to run."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "source_doc_sha256": "0" * 64,
        "ingested_at": datetime(2026, 6, 13, 10, 0, tzinfo=timezone.utc).isoformat(),
        "collection_name": "eu_ai_act_test",
        "embedding_model": "BAAI/bge-m3",
    }
    (Path(persist_dir) / "corpus_version.json").write_text(json.dumps(payload))
    return payload


@pytest.fixture
def fake_retrieval(monkeypatch):
    """Patch the hybrid retriever to return a fixed two-chunk result so
    we don't need an embedding model in tests."""
    docs = [
        Document(
            page_content="Article 5 prohibits social scoring by public authorities.",
            metadata={
                "article_number": "5",
                "title": "Prohibited practices",
                "chapter": "II",
                "chunk_kind": "article",
            },
        ),
        Document(
            page_content="Annex III lists biometric identification as high-risk.",
            metadata={
                "article_number": "annex-iii",
                "title": "High-risk AI systems",
                "chapter": "Annex III",
                "chunk_kind": "annex",
            },
        ),
    ]

    class _FakeRetriever:
        def invoke(self, q):
            return list(docs)

    import rag_chain

    monkeypatch.setattr(rag_chain, "_build_hybrid_retriever", lambda k: _FakeRetriever())
    return docs


# ── Sealed shape: corpus_ingested/v1 ─────────────────────────────────
def test_ingest_seals_corpus_ingested_v1(tiny_pdf, fake_reranker, seal_recorder, monkeypatch):
    """A successful ingest emits exactly one sealed event whose payload
    matches corpus_ingested/v1 — hashes only, no plaintext, no tenant
    identity."""
    # Re-route persistence into tmp space already configured by conftest.
    from ingest import ingest

    result = ingest(tiny_pdf)
    assert result["source_doc_sha256"], "source_doc_sha256 missing"
    assert result["seal"].event_hash, "seal must have event_hash in strict mode"

    assert seal_recorder.posted_paths == ["/audit/log"]
    assert len(seal_recorder.posted_payloads) == 1
    posted = seal_recorder.posted_payloads[0]

    # Step name and payload shape match the vendored schema.
    assert posted["step"] == "corpus_ingested"
    payload = posted["payload"]
    assert payload["schema_version"] == "corpus_ingested/v1"
    assert payload["source_doc_sha256"] == result["source_doc_sha256"]
    assert payload["collection_name"]
    assert payload["embedding_model"]
    assert payload["time_source"] == "wall_clock"

    # No tenant identity anywhere.
    assert "tenant_id" not in posted
    assert "client_id" not in posted
    assert "tenant_id" not in payload
    assert "client_id" not in payload


# ── Sealed shape: agent_decision/v1 ──────────────────────────────────
def test_query_seals_agent_decision_v1(
    seal_recorder, fake_llm, fake_reranker, fake_retrieval, monkeypatch
):
    """A query emits one agent_decision/v1 with: trace_id =
    rag:sha256(question), corpus_version from disk, input_refs with
    one doc hash and N chunk hashes, decision.rationale_summary
    capped and PII-screened, no plaintext question/answer fields."""
    from config import settings
    _stage_corpus_version(settings.chroma_persist_dir)

    fake_llm.script([
        # Answer
        "Biometric identification is high-risk under (Article 5) and (Annex III).",
        # Groundedness
        '{"verdict": "yes", "confidence": 0.92, "reason": "Both citations supported."}',
    ])

    from rag_chain import query

    result = query("What is high-risk under the AI Act?")

    assert seal_recorder.posted_paths == ["/audit/log"]
    posted = seal_recorder.posted_payloads[0]
    assert posted["step"] == "agent_decision"
    payload = posted["payload"]

    # Schema version + trace.
    assert payload["schema_version"] == "agent_decision/v1"
    expected_trace = "rag:" + hashlib.sha256(
        b"What is high-risk under the AI Act?"
    ).hexdigest()
    assert payload["trace_id"] == expected_trace

    # Model identity.
    assert payload["model"]["provider"] == "anthropic"
    assert payload["model"]["id"]

    # Corpus version mirrors what we staged.
    assert payload["corpus_version"]["source_doc_sha256"] == "0" * 64

    # Input refs: only hashes. One doc hash, one chunk hash per
    # retrieved document.
    assert payload["input_refs"]["doc_hashes"] == ["0" * 64]
    assert len(payload["input_refs"]["chunk_hashes"]) == len(fake_retrieval)
    for ch in payload["input_refs"]["chunk_hashes"]:
        assert len(ch) == 64

    # Decision.
    assert payload["decision"]["outcome"] == "answered"
    assert 0.0 <= payload["decision"]["confidence"] <= 1.0
    summary = payload["decision"]["rationale_summary"]
    assert len(summary) <= 2000
    assert summary.startswith("Grounded: yes")

    # Human oversight.
    assert payload["human_oversight"] == {
        "required": False,
        "status": "not_required",
        "approver_ref": None,
    }

    # NEVER plaintext question (atable only via trace_id = sha256(q)).
    flat = json.dumps(payload).lower()
    assert "what is high-risk under the ai act" not in flat
    # No top-level question/answer fields — the schema forbids extras
    # (extra="forbid" on BasePayload) but pin it explicitly so a
    # regression in the schema can't pass silently here.
    assert "question" not in payload
    assert "answer" not in payload
    # The LLM answer text DOES appear inside decision.rationale_summary
    # (the only human-readable, length-capped, PII-screened field —
    # CLAUDE.md §5 of the kernel). That's the kernel contract, not a leak.
    assert "biometric identification is high-risk" in summary.lower()


def test_query_payload_has_no_tenant_id(
    seal_recorder, fake_llm, fake_reranker, fake_retrieval
):
    from config import settings
    _stage_corpus_version(settings.chroma_persist_dir)
    fake_llm.script([
        "See (Article 5).",
        '{"verdict": "yes", "confidence": 0.8, "reason": "ok"}',
    ])
    from rag_chain import query
    query("question?")

    posted = seal_recorder.posted_payloads[0]
    flat = json.dumps(posted).lower()
    for forbidden in ("tenant_id", "client_id", "tenant\":", "cid\":"):
        assert forbidden not in flat, f"forbidden key {forbidden!r} in seal body"


def test_query_refuses_when_no_corpus_ingested_yet(
    seal_recorder, fake_llm, fake_reranker, fake_retrieval, monkeypatch
):
    """An agent_decision/v1 must reference a corpus_version. If ingest
    has not run, query() refuses rather than sealing a decision against
    an empty knowledge base."""
    # Make sure no stale corpus_version.json from a previous test trips
    # us up.
    from config import settings
    cv = Path(settings.chroma_persist_dir) / "corpus_version.json"
    if cv.exists():
        cv.unlink()

    from rag_chain import query

    with pytest.raises(RuntimeError, match="corpus_version"):
        query("question?")

    # And nothing was sealed.
    assert seal_recorder.posted_payloads == []


def test_groundedness_verifier_handles_garbage_response(
    seal_recorder, fake_llm, fake_reranker, fake_retrieval
):
    """If the groundedness LLM hiccups and returns non-JSON, the
    pipeline must still seal — verdict 'partial' with a clear reason
    — instead of crashing or refusing to log the decision."""
    from config import settings
    _stage_corpus_version(settings.chroma_persist_dir)
    fake_llm.script(["See (Article 5).", "not even close to json"])
    from rag_chain import query

    result = query("question?")
    assert result.groundedness["verdict"] == "partial"
    assert "unparseable" in result.groundedness["reason"].lower()

    posted = seal_recorder.posted_payloads[0]
    assert posted["payload"]["decision"]["rationale_summary"].startswith(
        "Grounded: partial"
    )
