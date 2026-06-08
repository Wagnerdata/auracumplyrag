"""Thin client for the AuraComply MCP seal_decision endpoint."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)

SEAL_PATH = "/seal_decision"
DEFAULT_TIMEOUT = 5.0


def build_seal_payload(
    question: str,
    answer: str,
    articles_cited: list[str],
    tenant_id: str | None = None,
) -> dict[str, Any]:
    return {
        "decision_type": "compliance_query",
        "content": {
            "question": question,
            "answer": answer,
            "articles_cited": articles_cited,
        },
        "tenant_id": tenant_id or settings.tenant_id,
    }


def seal_decision(
    question: str,
    answer: str,
    articles_cited: list[str],
    tenant_id: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """POST seal payload to AuraComply. Returns the seal envelope on success.

    Failures are logged but never raised so the caller can decide whether to
    surface a soft warning; query responses must not be blocked by seal outages.
    """
    url = settings.auracomply_url.rstrip("/") + SEAL_PATH
    payload = build_seal_payload(question, answer, articles_cited, tenant_id)
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            body = resp.json()
            return {"ok": True, "seal": body}
    except Exception as exc:
        logger.warning("AuraComply seal failed: %s", exc)
        return {"ok": False, "error": str(exc)}
