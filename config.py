"""Configuration for the RAG showcase.

Pydantic-settings reads the .env file at import time. Anything that
the SDK reads on its own (AURACOMPLY_BUFFER_HMAC_KEY,
AURACOMPLY_SEAL_POLICY) is NOT mirrored here on purpose — the SDK is
the source of truth for its own env names; this file only carries
fields the RAG application code needs at the type level.
"""
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # ── RAG application ─────────────────────────────────────────────
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "eu_ai_act"
    # bge-m3 — multilingual dense retriever, ~2.3 GB on first download.
    # Phase 4 swap from sentence-transformers/all-MiniLM-L6-v2 was driven
    # by retrieval quality on EU AI Act terminology; the older model
    # consistently lost recall on Recital/Annex references.
    embedding_model: str = "BAAI/bge-m3"
    # bge-reranker-base — cross-encoder run after hybrid (BM25+dense+RRF).
    # Local-only to keep the demo airgap-able; ~280 MB.
    reranker_model: str = "BAAI/bge-reranker-base"
    eu_ai_act_pdf: str = "./data/eu_ai_act.pdf"

    # ── API auth ────────────────────────────────────────────────────
    # Required bearer for /query and /ingest. Falsy values are
    # accepted only when AURACOMPLY_DEMO_PROFILE=1.
    rag_api_token: str = ""
    auracomply_demo_profile: bool = False

    # ── Kernel SDK wiring ───────────────────────────────────────────
    # Kernel HTTP base URL the SDK posts to (NOT to be confused with
    # the kernel's own API — this is the kernel address from the
    # RAG container's perspective).
    auracomply_url: str = "http://localhost:5002"
    # Bearer the SDK presents to the kernel. Tenant identity is
    # derived server-side from this token. The RAG layer never reads
    # nor logs it — the SDK swallows it whole.
    auracomply_api_token: str = ""
    # Where the SDK persists pending.jsonl + sealed.jsonl.
    auracomply_buffer_dir: str = "./.auracomply-buffer"


settings = Settings()

Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(settings.eu_ai_act_pdf) or ".").mkdir(parents=True, exist_ok=True)
Path(settings.auracomply_buffer_dir).mkdir(parents=True, exist_ok=True)
