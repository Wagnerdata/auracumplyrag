"""FastAPI surface for the EU AI Act RAG pipeline.

Two protected endpoints (/query, /ingest) and one anonymous (/health).
Auth is a static bearer compared with hmac.compare_digest — the only
permitted bypass is AURACOMPLY_DEMO_PROFILE=1, which logs a loud
warning at boot.

Sealing goes through the AuraComply kernel SDK (seal_client.py); we
never construct an HTTP call to /audit/log here.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from config import settings
from ingest import ingest
from rag_chain import query as run_query
from seal_client import get_seal_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Auth ─────────────────────────────────────────────────────────────
# Token check is hmac.compare_digest on the SHA-256 of the presented
# value vs. the SHA-256 of the configured one — equal-length compare
# is required for the timing guarantee, and hashing first lets the
# operator rotate tokens of any length without padding tricks.
_EXPECTED_TOKEN_HASH = (
    hashlib.sha256(settings.rag_api_token.encode()).digest()
    if settings.rag_api_token
    else None
)


def verify_rag_token(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency: 401 on missing bearer, 403 on bad bearer.

    Bypassed entirely when AURACOMPLY_DEMO_PROFILE=1 — see the boot
    warning in `app.on_event("startup")`.
    """
    if settings.auracomply_demo_profile:
        return
    if _EXPECTED_TOKEN_HASH is None:
        # Misconfiguration: server has no token set AND is not in demo
        # profile. Fail closed.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="server_misconfigured: RAG_API_TOKEN is not set",
        )
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing_bearer",
            headers={"WWW-Authenticate": "Bearer"},
        )
    presented = authorization.split(" ", 1)[1].strip()
    presented_hash = hashlib.sha256(presented.encode()).digest()
    if not hmac.compare_digest(presented_hash, _EXPECTED_TOKEN_HASH):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid_bearer",
        )


# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="AuraComply RAG",
    description="EU AI Act compliance Q&A with immutable audit sealing.",
    version="1.0.0",
)


@app.on_event("startup")
def _warn_if_demo_profile() -> None:
    if settings.auracomply_demo_profile:
        logger.warning(
            "AURACOMPLY_DEMO_PROFILE=1 is ENABLED. /query and /ingest "
            "accept ANY caller — this disables the tenant boundary and "
            "must never run in production."
        )


# ── Request/response models ──────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Compliance question")
    k: int = Field(6, ge=1, le=20, description="Hybrid retrieval depth before rerank")


class Source(BaseModel):
    article_number: str | None
    title: str | None
    chapter: str | None
    chunk_kind: str | None
    snippet: str


class SealEnvelope(BaseModel):
    """Minimal projection of the SDK's SealResult.

    `event_hash` is the kernel's independent proof of inclusion in
    STRICT mode. In BUFFERED it is null until the background flusher
    seals the line — the response still surfaces `line_id` so the
    caller can look it up later in sealed.jsonl.
    """
    line_id: str
    event_hash: str | None
    log_id: str | None
    replayed: bool
    buffered: bool


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    articles_cited: list[str]
    groundedness: dict[str, Any]
    seal: SealEnvelope


class IngestRequest(BaseModel):
    pdf_path: str | None = None


class IngestResponse(BaseModel):
    articles: int
    documents: int
    persist_dir: str
    source_doc_sha256: str
    seal: SealEnvelope


# ── Routes ───────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.anthropic_model}


@app.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(verify_rag_token)],
)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    try:
        sealed = run_query(req.question, k=req.k, seal_client=get_seal_client())
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=f"query_failed: {exc}") from exc

    return QueryResponse(
        question=req.question,
        answer=sealed.answer,
        sources=[Source(**s) for s in sealed.sources],
        articles_cited=sealed.articles_cited,
        groundedness=sealed.groundedness,
        seal=SealEnvelope(
            line_id=sealed.seal.line_id,
            event_hash=sealed.seal.event_hash,
            log_id=sealed.seal.log_id,
            replayed=sealed.seal.replayed,
            buffered=sealed.seal.buffered,
        ),
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(verify_rag_token)],
)
def ingest_endpoint(req: IngestRequest) -> IngestResponse:
    try:
        result = ingest(req.pdf_path, seal_client=get_seal_client())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"ingest_failed: {exc}") from exc

    return IngestResponse(
        articles=result["articles"],
        documents=result["documents"],
        persist_dir=result["persist_dir"],
        source_doc_sha256=result["source_doc_sha256"],
        seal=SealEnvelope(
            line_id=result["seal"].line_id,
            event_hash=result["seal"].event_hash,
            log_id=result["seal"].log_id,
            replayed=result["seal"].replayed,
            buffered=result["seal"].buffered,
        ),
    )
