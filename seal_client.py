"""Singleton accessor for the AuraComply kernel SDK.

The RAG layer never constructs HTTP requests to /audit/log on its own
— the SDK owns policy, buffering, HMAC, idempotency, retry, and the
kernel contract (see CLAUDE.md §R1 / Sealing). This module wires
config.settings + environment to a single AuracomplyClient instance
that the rest of the codebase reuses.

Construction is lazy: the SDK starts a daemon flusher thread under
BUFFERED policy, so a test process that never reaches a sealing path
should never see it spin up.

Tests inject a `_transport=httpx.MockTransport(...)` by calling
`set_seal_client(client)` directly — that bypasses the singleton and
the env-var validation, and is the ONLY supported test seam.
"""
from __future__ import annotations

import atexit
import threading
from typing import TYPE_CHECKING

from config import settings

if TYPE_CHECKING:
    from auracomply import AuracomplyClient


_client: "AuracomplyClient | None" = None
_lock = threading.Lock()


def get_seal_client() -> "AuracomplyClient":
    """Return the process-wide AuracomplyClient, constructing it on
    first call. Subsequent callers share the same instance — the SDK's
    buffer assumes single-writer per buffer_dir."""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        _client = _build_client()
        atexit.register(_close)
        return _client


def set_seal_client(client: "AuracomplyClient") -> None:
    """Replace the singleton (tests only). Closes any prior instance
    so daemon threads do not leak between tests."""
    global _client
    with _lock:
        if _client is not None and _client is not client:
            try:
                _client.close()
            except Exception:  # noqa: BLE001 — test path; nothing to do
                pass
        _client = client


def reset_seal_client() -> None:
    """Tear down the singleton (tests only)."""
    set_seal_client.__wrapped__ if False else None  # keep linters quiet
    global _client
    with _lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:  # noqa: BLE001
                pass
            _client = None


def _close() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


def _build_client() -> "AuracomplyClient":
    # Local import keeps the SDK out of test import paths that never
    # exercise sealing — and out of the dependency graph until the
    # first /query or /ingest hits.
    from auracomply import AuracomplyClient, SealPolicy

    api_url = settings.auracomply_url.rstrip("/")
    token = settings.auracomply_api_token
    if not token:
        raise RuntimeError(
            "AURACOMPLY_API_TOKEN is not set. The kernel SDK refuses "
            "to construct a client without a bearer; tenant identity "
            "depends on it."
        )

    # Policy default is BUFFERED; the SDK still honours an explicit
    # env override (AURACOMPLY_SEAL_POLICY). The kwarg form keeps
    # this call site explicit for readers.
    policy = SealPolicy.from_env(default=SealPolicy.BUFFERED)

    return AuracomplyClient(
        api_url=api_url,
        token=token,
        buffer_dir=settings.auracomply_buffer_dir,
        policy=policy,
    )
