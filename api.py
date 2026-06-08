"""FastAPI surface for the EU AI Act RAG pipeline."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from auracomply_client import seal_decision
from config import settings
from ingest import ingest
from rag_chain import query as run_query

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(
    title="AuraComply RAG",
    description="EU AI Act compliance Q&A with immutable audit sealing.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Compliance question")
    k: int = Field(3, ge=1, le=10, description="Number of retrieved chunks")


class Source(BaseModel):
    article_number: str | None
    title: str | None
    chapter: str | None
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    articles_cited: list[str]
    seal: dict[str, Any]


class IngestRequest(BaseModel):
    pdf_path: str | None = None


class IngestResponse(BaseModel):
    articles: int
    documents: int
    persist_dir: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.anthropic_model}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    try:
        result = run_query(req.question, k=req.k)
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=f"query_failed: {exc}") from exc

    seal = seal_decision(
        question=req.question,
        answer=result.answer,
        articles_cited=result.articles_cited,
    )
    if not seal.get("ok"):
        logger.warning("Continuing despite seal failure: %s", seal.get("error"))

    return QueryResponse(
        question=req.question,
        answer=result.answer,
        sources=[Source(**s) for s in result.sources],
        articles_cited=result.articles_cited,
        seal=seal,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest) -> IngestResponse:
    try:
        result = ingest(req.pdf_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"ingest_failed: {exc}") from exc
    return IngestResponse(**result)
