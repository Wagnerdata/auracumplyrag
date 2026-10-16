"""Shared fixtures for the auracomply-rag hermetic test suite.

Three test seams:

  1. Kernel SDK transport — `httpx.MockTransport` plumbed into
     `AuracomplyClient(_transport=...)`. Captures every /audit/log
     POST so tests can inspect the sealed payload directly.

  2. LLM — `rag_chain.set_llm(fake)`. The fake exposes `.invoke(...)`
     returning an object with `.content`; the test can script answers
     and groundedness verdicts in order.

  3. Reranker — `rag_chain.set_reranker(fake)`. The fake exposes
     `.compute_score(pairs, normalize=...)` returning a list of
     floats. Tests that don't care about reranking get an identity
     fake from the `fake_reranker` fixture.

Hermetic: no real network, no real model downloads, no Chroma
persistence outside tmp_path.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Kernel SDK lives in the sibling repo. Tests import it as a regular
# package; requirements.txt installs it the same way for production
# via `auracomply @ file:../AuraComply/sdk`, but the test path keeps
# import-time deps minimal.
SDK_PATH = ROOT.parent / "AuraComply" / "sdk"
if SDK_PATH.is_dir() and str(SDK_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PATH))


# ── Env: must be set BEFORE the app/SDK import ───────────────────────
# pydantic-settings reads at first `from config import settings`, so we
# populate os.environ at conftest.py module-import time — that runs
# before any test file's top-level imports because pytest collects
# conftests first.
_TEST_TMP = Path(tempfile.mkdtemp(prefix="rag-hermetic-"))
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic")
os.environ.setdefault("ANTHROPIC_MODEL", "claude-test-model")
os.environ.setdefault("AURACOMPLY_URL", "http://kernel.test")
os.environ.setdefault("AURACOMPLY_API_TOKEN", "test-kernel-token")
os.environ.setdefault("AURACOMPLY_BUFFER_DIR", str(_TEST_TMP / "buffer"))
os.environ.setdefault("AURACOMPLY_BUFFER_HMAC_KEY", "a" * 64)
os.environ.setdefault("AURACOMPLY_SEAL_POLICY", "strict")
os.environ.setdefault("CHROMA_PERSIST_DIR", str(_TEST_TMP / "chroma"))
os.environ.setdefault("COLLECTION_NAME", "eu_ai_act_test")
os.environ.setdefault("EU_AI_ACT_PDF", str(_TEST_TMP / "fake.pdf"))
os.environ.setdefault("RAG_API_TOKEN", "test-rag-bearer")
os.environ.setdefault("AURACOMPLY_DEMO_PROFILE", "0")


# ── Recorded /audit/log requests ─────────────────────────────────────
class _SealRecorder:
    """Captures every POST /audit/log and /audit/log/batch the SDK
    sends. Tests inspect `posted_payloads` to assert the sealed shape."""

    def __init__(self):
        self.posted_payloads: list[dict[str, Any]] = []
        self.posted_paths: list[str] = []
        self._lock = threading.Lock()
        self._counter = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        with self._lock:
            self._counter += 1
            seq = self._counter
        self.posted_paths.append(request.url.path)
        if request.url.path == "/audit/log":
            self.posted_payloads.append(body)
            event_hash = hashlib.sha256(
                json.dumps(body, sort_keys=True).encode()
            ).hexdigest()
            return httpx.Response(
                201,
                json={
                    "status": "success",
                    "log_id": f"log-{seq}",
                    "event_hash": event_hash,
                },
            )
        if request.url.path == "/audit/log/batch":
            events = body.get("events", [])
            self.posted_payloads.extend(events)
            results = []
            for i, ev in enumerate(events):
                eh = hashlib.sha256(
                    json.dumps(ev, sort_keys=True).encode()
                ).hexdigest()
                results.append({
                    "event_hash": eh,
                    "log_id": f"log-{seq}-{i}",
                    "replayed": False,
                })
            return httpx.Response(201, json={"status": "success", "results": results})
        if request.url.path == "/verify":
            return httpx.Response(
                200,
                json={"intact": True, "mismatch_log_id": None, "events_checked": 0},
            )
        return httpx.Response(404, json={"error": "test_transport_unknown_path"})


@pytest.fixture
def seal_recorder(tmp_path):
    """Build an AuracomplyClient that posts into the recorder's mock
    transport. Replaces the singleton in seal_client.py so production
    code that calls get_seal_client() also gets the recorded instance."""
    import seal_client
    from auracomply import AuracomplyClient, SealPolicy

    recorder = _SealRecorder()
    transport = httpx.MockTransport(recorder.handler)
    client = AuracomplyClient(
        api_url="http://kernel.test",
        token="test-kernel-token",
        buffer_dir=str(tmp_path / "buffer"),
        hmac_key=b"a" * 32,
        policy=SealPolicy.STRICT,
        _transport=transport,
    )
    seal_client.set_seal_client(client)
    yield recorder
    seal_client.reset_seal_client()


# ── Fake LLM ─────────────────────────────────────────────────────────
class _FakeLLM(Runnable):
    """Returns scripted responses in order. Index N is the Nth
    `.invoke()` call. Use `script(["answer_text", '{"verdict": ...}'])`
    to set the answer + the groundedness JSON.

    Subclasses langchain_core.Runnable so that `prompt | llm | parser`
    composition works (LangChain refuses to compose anything that is
    not a Runnable / callable / dict)."""

    def __init__(self, responses: list[str]):
        super().__init__()
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[Any] = []

    def invoke(self, payload: Any, config=None, **kwargs) -> AIMessage:
        self.calls.append(payload)
        if self._idx >= len(self._responses):
            raise AssertionError(
                "FakeLLM exhausted: scripted "
                f"{len(self._responses)} responses, got {self._idx + 1} calls"
            )
        out = self._responses[self._idx]
        self._idx += 1
        return AIMessage(content=out)


@pytest.fixture
def fake_llm():
    """Returns the FakeLLM class and installs it via rag_chain.set_llm.
    Tests call `fake_llm.script(...)` to set responses for one
    `query()` invocation (answer first, then groundedness JSON)."""
    import rag_chain

    class _Holder:
        def __init__(self):
            self._llm = None

        def script(self, responses: list[str]) -> _FakeLLM:
            llm = _FakeLLM(responses)
            rag_chain.set_llm(llm)
            self._llm = llm
            return llm

        @property
        def last(self) -> _FakeLLM | None:
            return self._llm

    holder = _Holder()
    yield holder
    rag_chain.set_llm(None)


# ── Fake reranker ────────────────────────────────────────────────────
class _IdentityReranker:
    """Returns descending scores so the input order is preserved."""

    def compute_score(self, pairs, normalize: bool = True):
        return [1.0 - i * 0.01 for i, _ in enumerate(pairs)]


@pytest.fixture
def fake_reranker():
    import rag_chain

    rag_chain.set_reranker(_IdentityReranker())
    yield
    rag_chain.set_reranker(None)


# ── No real embeddings / no real Chroma in tests ─────────────────────
# bge-m3 alone is ~2GB; we never want a test run to download it. The
# fixture below is autouse=True so every test runs against an in-memory
# stand-in for embeddings + Chroma. Ingest stays exercised end-to-end
# (PDF -> splitters -> documents -> seal); the vector store is the
# only piece we mock.
class _FakeEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(t) % 7), 0.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 0.0, 0.0]


class _FakeVectorStore:
    """Minimal stand-in for langchain Chroma — supports add_documents
    + get(include=...). Stores documents in memory."""

    def __init__(self):
        self._docs: list[Any] = []

    def add_documents(self, documents):
        self._docs.extend(documents)

    def get(self, include=None):
        return {
            "documents": [d.page_content for d in self._docs],
            "metadatas": [d.metadata for d in self._docs],
        }

    def as_retriever(self, **kwargs):
        class _Ret:
            def __init__(self, docs):
                self._docs = docs

            def invoke(self, q):
                return list(self._docs)

        return _Ret(self._docs)


_fake_store_singleton: _FakeVectorStore | None = None


@pytest.fixture(autouse=True)
def _no_real_embeddings(monkeypatch):
    """Replace get_embeddings + get_vector_store + Chroma() in ingest
    so no test triggers a HF model download or persists to disk."""
    global _fake_store_singleton
    _fake_store_singleton = _FakeVectorStore()
    fake = _fake_store_singleton

    import ingest
    monkeypatch.setattr(ingest, "get_embeddings", lambda: _FakeEmbeddings())
    monkeypatch.setattr(ingest, "get_vector_store", lambda embeddings=None: fake)
    # ingest() also constructs a Chroma() directly inside the function;
    # patch the class itself to return our fake.
    monkeypatch.setattr(ingest, "Chroma", lambda **kwargs: fake)
    yield


# ── Tiny PDF for ingest tests ────────────────────────────────────────
@pytest.fixture
def tiny_pdf(tmp_path):
    """Generate a 1-page PDF containing a stub of the EU AI Act so
    ingest's regex pipeline has real article/recital/annex headers to
    chew on."""
    import pymupdf

    text = (
        "REGULATION (EU) 2024/1689\n\n"
        "(1) Whereas artificial intelligence systems can pose risks.\n"
        "(2) Whereas the Union seeks harmonised rules.\n\n"
        "HAVE ADOPTED THIS REGULATION:\n\n"
        "CHAPTER I\nGENERAL PROVISIONS\n\n"
        "Article 1\nSubject matter\n"
        "This Regulation lays down harmonised rules on artificial intelligence.\n\n"
        "CHAPTER II\nPROHIBITED AI PRACTICES\n\n"
        "Article 5\nProhibited AI practices\n"
        "The following AI practices shall be prohibited.\n\n"
        "ANNEX III\nHigh-risk AI systems\n"
        "Biometric identification systems are high-risk.\n\n"
        "ANNEX IV\nTechnical documentation\n"
        "Providers shall keep technical documentation.\n"
    )
    pdf_path = tmp_path / "tiny.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)
