"""Streamlit UI helpers: bearer header and seal classification.

The page itself is not rendered here; these tests pin that the UI
reads the bearer from the environment and never reports a seal as
successful unless the kernel confirmed it (event_hash present).
"""
from __future__ import annotations

import pytest

from app import auth_headers, classify_seal, describe_http_error


def test_auth_headers_reads_token_from_env(monkeypatch):
    monkeypatch.setenv("RAG_API_TOKEN", "ui-token")
    assert auth_headers() == {"Authorization": "Bearer ui-token"}


def test_auth_headers_empty_when_token_unset(monkeypatch):
    monkeypatch.delenv("RAG_API_TOKEN", raising=False)
    assert auth_headers() == {}


@pytest.mark.parametrize("seal", [None, {}, "garbage", {"event_hash": "abc"}])
def test_missing_or_malformed_seal_is_a_warning(seal):
    level, _ = classify_seal(seal)
    assert level == "warning"


def test_confirmed_seal_is_success():
    level, _ = classify_seal(
        {"line_id": "l1", "event_hash": "h", "log_id": "log-1",
         "replayed": False, "buffered": False}
    )
    assert level == "success"


def test_buffered_seal_is_not_reported_as_success():
    level, message = classify_seal(
        {"line_id": "l1", "event_hash": None, "log_id": None,
         "replayed": False, "buffered": True}
    )
    assert level == "info"
    assert "not yet confirmed" in message


def test_unconfirmed_unbuffered_seal_is_a_warning():
    level, _ = classify_seal(
        {"line_id": "l1", "event_hash": None, "log_id": None,
         "replayed": False, "buffered": False}
    )
    assert level == "warning"


def test_legacy_ok_shape_is_not_trusted():
    level, _ = classify_seal({"ok": True, "seal": {"seal_id": "x"}})
    assert level == "warning"


@pytest.mark.parametrize("code", [401, 403, 503])
def test_auth_errors_have_specific_messages(code):
    assert "RAG_API_TOKEN" in describe_http_error(code)
