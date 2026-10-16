# auracomply-rag

![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?logo=langchain&logoColor=white)
![Claude](https://img.shields.io/badge/Anthropic-Claude-D97757)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector-FF6B6B)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

A production-style **Retrieval-Augmented Generation** pipeline for EU AI Act
compliance queries, wired into the **AuraComply** audit chain. Every ingest and
every answered query is sealed into the tenant's audit chain through the
AuraComply kernel SDK (`auracomply.AuracomplyClient`). Only hashes and a bounded
rationale summary are sealed — never the plaintext question.

---

## Architecture

```
                ┌──────────────────────────────────────────────────────┐
                │                  Streamlit UI (app.py)               │
                │   question  ──► answer + citations + seal status     │
                └─────────────────────┬────────────────────────────────┘
                                      │ HTTP
                                      ▼
            ┌─────────────────────────────────────────────────────────┐
            │              FastAPI service (api.py)                   │
            │   POST /query     POST /ingest     GET /health          │
            │   (bearer auth on /query and /ingest)                   │
            └──────┬───────────────────────┬───────────────┬──────────┘
                   │                       │               │
                   ▼                       ▼               ▼
        ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐
        │  RAG chain       │   │  Ingest pipeline │   │  AuraComply SDK    │
        │  (rag_chain.py)  │   │   (ingest.py)    │   │  (seal_client.py)  │
        │                  │   │                  │   │                    │
        │ BM25 + dense     │   │ EU AI Act PDF    │   │ client.seal(...)   │
        │  → RRF fusion    │   │  → split by      │   │  → kernel          │
        │  → bge rerank    │   │    Article /     │   │    /audit/log      │
        │  → Claude answer │   │    Recital /     │   │                    │
        │  → groundedness  │   │    Annex III|IV  │   │ agent_decision/v1  │
        │  → seal          │   │  → embed         │   │ corpus_ingested/v1 │
        └────────┬─────────┘   │  → ChromaDB      │   └────────────────────┘
                 │             │  → seal          │
                 │             └────────┬─────────┘
                 ▼                      ▼
        ┌────────────────────────────────────────────┐
        │  ChromaDB (local persistent store)         │
        │  metadata: article_number / title /        │
        │            chapter / chunk_kind            │
        │  + corpus_version.json                     │
        └────────────────────────────────────────────┘
```

---

## Stack

| Layer            | Choice                                       |
| ---------------- | -------------------------------------------- |
| Orchestration    | LangChain (LCEL Runnable composition)        |
| Vector store     | ChromaDB (local persist dir, `CHROMA_PERSIST_DIR`) |
| Retrieval        | Hybrid BM25 + dense, fused with RRF (`EnsembleRetriever`, weights 0.5/0.5) |
| Embeddings       | `BAAI/bge-m3` (local, HuggingFace)           |
| Reranker         | `BAAI/bge-reranker-base` (local cross-encoder, FlagEmbedding) |
| LLM              | Anthropic Claude (`claude-sonnet-4-6` default, overridable via `ANTHROPIC_MODEL`) |
| API              | FastAPI + Pydantic                           |
| UI               | Streamlit                                    |
| Audit chain      | AuraComply kernel SDK (`auracomply`, installed from `../AuraComply/sdk`) |
| Packaging        | Docker Compose (api, streamlit, chromadb)    |
| Tests            | pytest, SDK mocked via `httpx.MockTransport` |

---

## Design decisions

- **Structure-aware chunking.** `ingest.py` recognises three chunk families —
  `Article N` (strict header), `Recital (N)` (numbered enumeration up to
  "HAVE ADOPTED THIS REGULATION") and `ANNEX III|IV` — and tags every chunk
  with `article_number`, `title`, `chapter` and `chunk_kind`. Chunks longer
  than 512 characters are sub-split with `RecursiveCharacterTextSplitter`
  (size 512, overlap 50). If no structural boundary is found the splitter
  falls back to plain 512-character chunks.
- **Hybrid retrieval + local rerank.** BM25 gives the recall floor for exact
  regulatory identifiers; the dense retriever adds paraphrases. The fused
  candidates (`k`, default 6) are reranked by `bge-reranker-base` and the top 3
  are passed to Claude. Retrieval runs fully in-process — no API keys.
- **Citations and groundedness.** The prompt requires inline `(Article XX)`,
  `(Recital N)` or `(Annex III)` citations and forbids invented identifiers.
  `articles_cited` lists the identifiers cited in the answer (falling back to
  the retrieved ones). A second Claude call checks groundedness and returns
  `{verdict: yes|partial|no, confidence, reason}`; unparseable verifier
  output degrades to `partial` instead of failing the query.
- **Buffered sealing.** Sealing goes through the kernel SDK with
  `AURACOMPLY_SEAL_POLICY=buffered` by default: `seal()` writes to an
  HMAC-signed local buffer and returns immediately, and the SDK's background
  flusher delivers it to the kernel. A kernel outage therefore does not block
  answers; the response carries `seal.buffered=true` and a `line_id` instead
  of an `event_hash`. `strict` and `warn` are available via the same env var.
  If the SDK cannot be constructed (e.g. `AURACOMPLY_API_TOKEN` unset) or
  `seal()` raises, the endpoint returns 500.
- **Hash-only audit rows.** The `agent_decision/v1` payload contains
  `trace_id = "rag:" + sha256(question)`, the source PDF hash, one SHA-256 per
  retrieved chunk, the model id, the `corpus_version`, and a
  `rationale_summary` capped at 2000 characters. No tenant identifier is sent
  in the body — tenant identity comes from the bearer token the SDK presents.
- **Corpus versioning.** Every ingest seals `corpus_ingested/v1` and writes
  `corpus_version.json` next to the Chroma store. `/query` refuses to run
  until an ingest has happened, so every decision is pinned to a known corpus.
- **Embeddings are local.** `bge-m3` (~2.3 GB) and `bge-reranker-base`
  (~280 MB) are downloaded on first use; swap via `EMBEDDING_MODEL` /
  `RERANKER_MODEL`.

---

## Quick start

### 1. Local (Python ≥ 3.11)

The kernel SDK is not on PyPI yet; `requirements.txt` installs it from the
sibling checkout `../AuraComply/sdk`, so clone both repos side by side.

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY, RAG_API_TOKEN, AURACOMPLY_API_TOKEN and
# AURACOMPLY_BUFFER_HMAC_KEY in .env

make install
# place the EU AI Act PDF at ./data/eu_ai_act.pdf
make ingest
make run-api       # http://localhost:8001
make run-ui        # http://localhost:8501
```

### 2. Docker

The image build needs the AuraComply kernel repo checked out next to this
one, because the SDK is copied from `../AuraComply/sdk`:

```
<parent>/
├── AuraComply/        # kernel repo (provides sdk/)
└── auracomply-rag/    # this repo
```

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY, RAG_API_TOKEN, AURACOMPLY_API_TOKEN and
# AURACOMPLY_BUFFER_HMAC_KEY in .env (compose refuses to start without
# the two AURACOMPLY_* secrets)
docker compose up --build -d
# Streamlit: http://localhost:8501
# FastAPI:   http://localhost:8001/docs
# ChromaDB:  http://localhost:8000
```

Compose passes the SDK to the build as the named context `sdk`
(`additional_contexts`, needs BuildKit). Without compose, pass it yourself
from this repo's root:

```bash
docker build --build-context sdk=../AuraComply/sdk -t auracomply-rag .
```

Notes:

- The API reaches the kernel at `http://aura-compliance:5002` by default,
  which only resolves when the `api` container joins the kernel's Docker
  network; otherwise set `AURACOMPLY_URL` in `.env`.
- Pending seals are kept in the `rag-seal-buffer` named volume
  (`/data/auracomply-buffer`), so they survive container restarts.
- `.env`, `chroma_db/`, `data/` and the local seal buffer are excluded from
  the image by `.dockerignore`.
- The `chromadb` service is started but the API uses its local persist
  directory, not that service.

### 3. Tests

```bash
make test
```

22 hermetic tests (`pytest --collect-only -q`): auth, ingest splitting, seal
payloads, and a smoke eval over the golden set. The full eval
(`evals/runner.py`) is opt-in via `RAG_EVAL_FULL=1` and needs
`ANTHROPIC_API_KEY`.

---

## API

`/query` and `/ingest` require `Authorization: Bearer <RAG_API_TOKEN>`
(401 if missing, 403 if wrong, 503 if the server has no token configured).
`GET /health` is anonymous and returns `{"status": "ok", "model": "..."}`.
`AURACOMPLY_DEMO_PROFILE=1` disables the bearer check — **this removes the
tenant boundary protection and must never be enabled in production.**

`POST /query`

```json
{
  "question": "What obligations apply to high-risk AI systems?",
  "k": 6
}
```

`k` (1–20, default 6) is the hybrid retrieval depth before reranking.

Response:

```json
{
  "question": "...",
  "answer": "High-risk AI systems must implement a risk management system (Article 9) ...",
  "sources": [
    {
      "article_number": "9",
      "title": "Risk management system",
      "chapter": "III",
      "chunk_kind": "article",
      "snippet": "..."
    }
  ],
  "articles_cited": ["9", "10", "11"],
  "groundedness": {"verdict": "yes", "confidence": 0.9, "reason": "..."},
  "seal": {
    "line_id": "...",
    "event_hash": null,
    "log_id": null,
    "replayed": false,
    "buffered": true
  }
}
```

`POST /ingest`

```json
{ "pdf_path": "./data/eu_ai_act.pdf" }
```

Response: `articles`, `documents`, `persist_dir`, `source_doc_sha256` and the
same `seal` envelope.

---

## AuraComply integration

All sealing goes through `seal_client.get_seal_client()`, a lazily built
process-wide `AuracomplyClient` configured from `AURACOMPLY_URL`,
`AURACOMPLY_API_TOKEN`, `AURACOMPLY_BUFFER_DIR` and `AURACOMPLY_SEAL_POLICY`.
This repo never builds HTTP requests to the kernel itself; the SDK handles
policy, buffering, HMAC, idempotency and retry.

Two events are sealed under `product="rag-eu-ai-act"`:

| Step              | When                    | Idempotency key           |
| ----------------- | ----------------------- | ------------------------- |
| `corpus_ingested` | end of every ingest     | `corpus:<pdf sha256>`     |
| `agent_decision`  | every answered `/query` | `rag:<sha256(question)>`  |

`agent_decision/v1` payload (abridged):

```json
{
  "schema_version": "agent_decision/v1",
  "trace_id": "rag:<sha256(question)>",
  "model": {"provider": "anthropic", "id": "claude-sonnet-4-6", "...": "..."},
  "corpus_version": {"source_doc_sha256": "...", "ingested_at": "..."},
  "input_refs": {"doc_hashes": ["..."], "chunk_hashes": ["...", "..."]},
  "decision": {
    "outcome": "answered",
    "confidence": 0.9,
    "rationale_summary": "Grounded: yes (conf 0.90). Cited: 9. ..."
  },
  "human_oversight": {"required": false, "status": "not_required", "approver_ref": null},
  "data_categories": ["regulatory_text"],
  "timestamp": "...",
  "time_source": "wall_clock"
}
```

The SDK's `SealResult` is projected into the response `seal` envelope
(`line_id`, `event_hash`, `log_id`, `replayed`, `buffered`).

The Streamlit UI sends `RAG_API_TOKEN` as its bearer and reports a seal as
successful only when the kernel confirmed it (`event_hash` present). A
buffered seal is shown as "queued, not yet confirmed"; a missing or
inconsistent seal is shown as a warning.

---

## Environment variables

| Variable                     | Default                    | Purpose                              |
| ---------------------------- | -------------------------- | ------------------------------------ |
| `ANTHROPIC_API_KEY`          | —                          | Required for Claude calls            |
| `ANTHROPIC_MODEL`            | `claude-sonnet-4-6`        | Claude model id for answers and the groundedness check |
| `CHROMA_PERSIST_DIR`         | `./chroma_db`              | ChromaDB persistence directory (also holds `corpus_version.json`) |
| `COLLECTION_NAME`            | `eu_ai_act`                | Vector collection name               |
| `EMBEDDING_MODEL`            | `BAAI/bge-m3`              | HuggingFace embedding model id       |
| `RERANKER_MODEL`             | `BAAI/bge-reranker-base`   | Cross-encoder reranker model id      |
| `EU_AI_ACT_PDF`              | `./data/eu_ai_act.pdf`     | Source PDF for ingestion             |
| `RAG_API_TOKEN`              | —                          | Bearer required on `/query` and `/ingest` |
| `AURACOMPLY_DEMO_PROFILE`    | `0`                        | `1` disables bearer auth — demo only, never in production |
| `AURACOMPLY_URL`             | `http://localhost:5002`    | Kernel HTTP base URL used by the SDK |
| `AURACOMPLY_API_TOKEN`       | —                          | Kernel bearer for this tenant (tenant identity is derived from it) |
| `AURACOMPLY_BUFFER_DIR`      | `./.auracomply-buffer`     | SDK buffer directory                 |
| `AURACOMPLY_BUFFER_HMAC_KEY` | —                          | HMAC key for the SDK buffer (read by the SDK) |
| `AURACOMPLY_SEAL_POLICY`     | `buffered`                 | `buffered`, `strict` or `warn` (read by the SDK; `warn` is demo-only) |
| `API_URL`                    | `http://localhost:8001`    | API base URL used by the Streamlit UI |

`TENANT_ID` is set in `docker-compose.yml` but is not read by the code: tenant
identity comes from `AURACOMPLY_API_TOKEN`.

---

## Project layout

```
auracomply-rag/
├── api.py                  # FastAPI service (bearer auth, /query, /ingest, /health)
├── app.py                  # Streamlit UI
├── config.py               # Pydantic settings
├── ingest.py               # PDF → chunks → ChromaDB → corpus_ingested/v1 seal
├── rag_chain.py            # hybrid retrieval → rerank → Claude → groundedness → seal
├── seal_client.py          # AuracomplyClient singleton (kernel SDK wiring)
├── evals/
│   ├── golden_set.jsonl    # golden Q&A set
│   └── runner.py           # eval runner (full run opt-in via RAG_EVAL_FULL=1)
├── tests/                  # pytest suite (22 tests)
│   ├── conftest.py
│   ├── test_auth.py
│   ├── test_ingest_split.py
│   ├── test_seal_payloads.py
│   └── test_smoke_eval.py
├── .env.example
├── CLAUDE.md
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── requirements.txt
```
