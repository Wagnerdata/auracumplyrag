"""Test suite for AuraComply RAG pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import auracomply_client
import ingest as ingest_mod
import rag_chain
from api import app
from auracomply_client import build_seal_payload, seal_decision


SAMPLE_TEXT = """CHAPTER I
GENERAL PROVISIONS

Article 1
Subject matter
This Regulation lays down harmonised rules on artificial intelligence,
and refers to Article 114 of the Treaty as legal basis.

Article 2
Scope
This Regulation applies to providers placing AI systems on the Union market.

CHAPTER II
PROHIBITED AI PRACTICES

Article 5
Prohibited artificial intelligence practices
The following AI practices shall be prohibited: social scoring by public authorities.
"""


def test_article_boundary_split_extracts_articles_and_chapters() -> None:
    chunks = ingest_mod._split_by_article(SAMPLE_TEXT)
    nums = [c.article_number for c in chunks]
    assert nums == ["1", "2", "5"]
    article_5 = next(c for c in chunks if c.article_number == "5")
    assert article_5.chapter == "II"
    assert "Prohibited" in article_5.content


def test_fallback_split_used_when_no_articles_present() -> None:
    chunks = ingest_mod._split_by_article("Just some unrelated regulatory text without articles.")
    assert chunks == []
    fallback = ingest_mod._fallback_split("Just some unrelated regulatory text without articles.")
    assert len(fallback) >= 1
    assert fallback[0].article_number.startswith("chunk-")


def test_chunks_carry_required_metadata() -> None:
    chunks = ingest_mod._split_by_article(SAMPLE_TEXT)
    docs = ingest_mod._to_documents(chunks)
    assert docs, "expected at least one document"
    for doc in docs:
        assert "article_number" in doc.metadata
        assert "title" in doc.metadata
        assert "chapter" in doc.metadata


def test_format_context_includes_article_headers() -> None:
    docs = [
        Document(
            page_content="High-risk systems require risk management.",
            metadata={"article_number": "9", "title": "Risk management", "chapter": "III"},
        ),
        Document(
            page_content="Providers must keep technical documentation.",
            metadata={"article_number": "11", "title": "Technical documentation", "chapter": "III"},
        ),
    ]
    formatted = rag_chain._format_context(docs)
    assert "Article 9" in formatted
    assert "Article 11" in formatted
    assert "Chapter III" in formatted


def test_articles_from_docs_deduplicates_in_order() -> None:
    docs = [
        Document(page_content="a", metadata={"article_number": "9"}),
        Document(page_content="b", metadata={"article_number": "11"}),
        Document(page_content="c", metadata={"article_number": "9"}),
    ]
    assert rag_chain._articles_from_docs(docs) == ["9", "11"]


def test_build_seal_payload_matches_mcp_contract() -> None:
    payload = build_seal_payload(
        question="Are biometric systems high-risk?",
        answer="Yes, see (Article 6).",
        articles_cited=["6"],
        tenant_id="rag-demo",
    )
    assert payload["decision_type"] == "compliance_query"
    assert payload["tenant_id"] == "rag-demo"
    assert payload["content"]["question"].startswith("Are biometric")
    assert payload["content"]["articles_cited"] == ["6"]


@respx.mock
def test_seal_decision_posts_to_auracomply() -> None:
    auracomply_client.settings.auracomply_url = "http://auracomply:8000"
    route = respx.post("http://auracomply:8000/seal_decision").mock(
        return_value=httpx.Response(200, json={"seal_id": "seal-123", "chain_hash": "abc"})
    )
    result = seal_decision("Q", "A", ["5"])
    assert route.called
    assert result["ok"] is True
    assert result["seal"]["seal_id"] == "seal-123"


@respx.mock
def test_seal_decision_swallows_errors() -> None:
    auracomply_client.settings.auracomply_url = "http://auracomply:8000"
    respx.post("http://auracomply:8000/seal_decision").mock(
        return_value=httpx.Response(500, json={"error": "down"})
    )
    result = seal_decision("Q", "A", ["5"])
    assert result["ok"] is False
    assert "error" in result


def _fake_rag_result(answer: str = "High-risk systems require oversight (Article 9)."):
    return rag_chain.RagResult(
        answer=answer,
        sources=[
            {
                "article_number": "9",
                "title": "Risk management",
                "chapter": "III",
                "snippet": "Snippet",
            }
        ],
        articles_cited=["9"],
    )


@respx.mock
def test_query_endpoint_returns_answer_and_seal() -> None:
    auracomply_client.settings.auracomply_url = "http://auracomply:8000"
    respx.post("http://auracomply:8000/seal_decision").mock(
        return_value=httpx.Response(200, json={"seal_id": "seal-xyz"})
    )

    with patch("api.run_query", return_value=_fake_rag_result()):
        client = TestClient(app)
        resp = client.post("/query", json={"question": "What are high-risk systems?"})

    assert resp.status_code == 200
    body = resp.json()
    assert "Article 9" in body["answer"]
    assert body["articles_cited"] == ["9"]
    assert body["seal"]["ok"] is True
    assert body["seal"]["seal"]["seal_id"] == "seal-xyz"


@respx.mock
def test_query_endpoint_succeeds_when_seal_fails() -> None:
    auracomply_client.settings.auracomply_url = "http://auracomply:8000"
    respx.post("http://auracomply:8000/seal_decision").mock(
        side_effect=httpx.ConnectError("no route")
    )

    with patch("api.run_query", return_value=_fake_rag_result()):
        client = TestClient(app)
        resp = client.post("/query", json={"question": "Whatever"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["seal"]["ok"] is False


def test_health_endpoint_reports_model() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "model" in resp.json()


def test_ingest_endpoint_missing_pdf_returns_404() -> None:
    with patch("api.ingest", side_effect=FileNotFoundError("missing")):
        client = TestClient(app)
        resp = client.post("/ingest", json={"pdf_path": "./does-not-exist.pdf"})
    assert resp.status_code == 404
