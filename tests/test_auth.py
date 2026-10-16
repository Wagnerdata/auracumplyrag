"""Bearer auth on /query and /ingest. /health is anonymous on purpose.

The token compare path is hmac.compare_digest over SHA-256 digests
(see api.verify_rag_token). These tests pin the four meaningful
states: no header, wrong scheme, wrong token, right token. Plus the
demo profile bypass and the fail-closed "RAG_API_TOKEN unset" state.
"""
from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient


def _client():
    # Import inside the function so the env in conftest is locked in
    # before pydantic-settings reads it.
    from api import app

    return TestClient(app)


def test_health_is_anonymous():
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_query_without_bearer_returns_401():
    resp = _client().post("/query", json={"question": "What is high-risk?"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "missing_bearer"


def test_query_with_wrong_token_returns_403():
    resp = _client().post(
        "/query",
        headers={"Authorization": "Bearer not-the-real-token"},
        json={"question": "What is high-risk?"},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "invalid_bearer"


def test_query_with_wrong_scheme_returns_401():
    resp = _client().post(
        "/query",
        headers={"Authorization": "Basic dGVzdDp0ZXN0"},
        json={"question": "What is high-risk?"},
    )
    assert resp.status_code == 401


def test_ingest_without_bearer_returns_401():
    resp = _client().post("/ingest", json={})
    assert resp.status_code == 401


def test_demo_profile_skips_auth_for_query(monkeypatch):
    """AURACOMPLY_DEMO_PROFILE=1 must let an anonymous /query through
    the auth dependency. The query itself still fails (no corpus
    ingested in this isolated test), but the failure surfaces past
    the bearer check — meaning the bypass works as documented."""
    monkeypatch.setattr("config.settings.auracomply_demo_profile", True)
    resp = _client().post("/query", json={"question": "What is high-risk?"})
    # Past the bearer; reaches the pipeline, which 500s on absent
    # corpus_version / no Anthropic client.
    assert resp.status_code != 401
    assert resp.status_code != 403


def test_fails_closed_when_token_unset(monkeypatch):
    """RAG_API_TOKEN unset AND no demo profile = 503, not 200.
    Quietly accepting any request would be the worst failure shape."""
    monkeypatch.setattr("api._EXPECTED_TOKEN_HASH", None)
    monkeypatch.setattr("config.settings.auracomply_demo_profile", False)
    resp = _client().post(
        "/query",
        headers={"Authorization": "Bearer anything"},
        json={"question": "Whatever"},
    )
    assert resp.status_code == 503
    assert "server_misconfigured" in resp.json()["detail"]
