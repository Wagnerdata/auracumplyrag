# auracomply-rag

![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?logo=langchain&logoColor=white)
![Claude](https://img.shields.io/badge/Anthropic-Claude-D97757)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector-FF6B6B)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

A production-style **Retrieval-Augmented Generation** pipeline for EU AI Act
compliance queries, wired into the **AuraComply** audit chain. Every query and
its answer are sealed into an immutable, tenant-scoped audit log via the
AuraComply MCP endpoint.

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
            └──────┬───────────────────────┬───────────────┬──────────┘
                   │                       │               │
                   ▼                       ▼               ▼
        ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐
        │  LCEL RAG chain  │   │  Ingest pipeline │   │  AuraComply MCP    │
        │  (rag_chain.py)  │   │   (ingest.py)    │   │ /seal_decision     │
        │                  │   │                  │   │ (auracomply_client)│
        │ retriever (k=3)  │   │ EU AI Act PDF    │   │                    │
        │   → prompt       │   │  → split by      │   │ Sealed audit chain │
        │   → Claude       │   │     Article XX   │   │  per tenant_id     │
        │   → citations    │   │  → embed         │   └────────────────────┘
        └────────┬─────────┘   │  → ChromaDB      │
                 │             └────────┬─────────┘
                 ▼                      ▼
        ┌────────────────────────────────────────────┐
        │  ChromaDB (persistent vector store)        │
        │  metadata: article_number / title / chapter│
        └────────────────────────────────────────────┘
```

---

## Stack

| Layer            | Choice                                       |
| ---------------- | -------------------------------------------- |
| Orchestration    | LangChain (LCEL Runnable composition)        |
| Vector store     | ChromaDB (persistent, local + dockerized)    |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2`     |
| LLM              | Anthropic Claude (`claude-sonnet-4-6` default; `claude-opus-4-8` available via `ANTHROPIC_MODEL`) |
| API              | FastAPI + Pydantic                           |
| UI               | Streamlit                                    |
| Audit chain      | AuraComply MCP `/seal_decision` endpoint     |
| Packaging        | Docker Compose (api, streamlit, chromadb)    |
| Tests            | pytest + respx                               |

---

## Design decisions

- **Article-boundary chunking.** `ingest.py` splits the source PDF on the
  `Article XX` regex and tags every chunk with `article_number`, `title`, and
  `chapter`. Articles longer than ~512 tokens are sub-split with
  `RecursiveCharacterTextSplitter` (size 512, overlap 50). If no Article
  pattern is found the splitter falls back to plain token chunks.
- **Citations are mandatory.** The RAG prompt forces inline `(Article XX)`
  citations and prohibits invented article numbers. The chain also returns the
  list of distinct articles retrieved, for UI badges and the audit payload.
- **Seal-or-warn.** If AuraComply's `/seal_decision` endpoint is unreachable,
  the API logs a warning and still returns the answer — query responses are
  never blocked by audit-chain outages. The seal status is returned to the UI
  so the operator sees the failure explicitly.
- **Embeddings are local.** Default embeddings use HuggingFace
  `sentence-transformers/all-MiniLM-L6-v2` so the project runs offline; swap
  via the `EMBEDDING_MODEL` env var.

---

## Quick start

### 1. Local (Python ≥ 3.11)

```bash
cp .env.example .env
# put your ANTHROPIC_API_KEY into .env

make install
# place the EU AI Act PDF at ./data/eu_ai_act.pdf
make ingest
make run-api       # http://localhost:8001
make run-ui        # http://localhost:8501
```

### 2. Docker

```bash
cp .env.example .env
docker compose up --build -d
# Streamlit: http://localhost:8501
# FastAPI:   http://localhost:8001/docs
# ChromaDB:  http://localhost:8000
```

### 3. Tests

```bash
make test
```

---

## API

`POST /query`

```json
{
  "question": "What obligations apply to high-risk AI systems?",
  "k": 3
}
```

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
      "snippet": "..."
    }
  ],
  "articles_cited": ["9", "10", "11"],
  "seal": {
    "ok": true,
    "seal": {"seal_id": "seal-...", "chain_hash": "..."}
  }
}
```

`POST /ingest`

```json
{ "pdf_path": "./data/eu_ai_act.pdf" }
```

---

## AuraComply integration

Every successful `/query` POSTs the following payload to
`${AURACOMPLY_URL}/seal_decision`:

```json
{
  "decision_type": "compliance_query",
  "content": {
    "question": "...",
    "answer": "...",
    "articles_cited": ["9", "11"]
  },
  "tenant_id": "rag-demo"
}
```

The returned seal envelope is forwarded to the client as
`response.seal.seal`. Failures degrade gracefully — the answer still ships, the
UI shows a warning, and the failure is logged for the operator.

---

## Environment variables

| Variable             | Default                                    | Purpose                              |
| -------------------- | ------------------------------------------ | ------------------------------------ |
| `ANTHROPIC_API_KEY`  | —                                          | Required for Claude calls            |
| `ANTHROPIC_MODEL`    | `claude-sonnet-4-6`                        | Override the Claude model (e.g. `claude-opus-4-8` for harder reasoning) |
| `AURACOMPLY_URL`     | `http://localhost:8000`                    | Base URL of the AuraComply MCP host  |
| `CHROMA_PERSIST_DIR` | `./chroma_db`                              | ChromaDB persistence directory       |
| `COLLECTION_NAME`    | `eu_ai_act`                                | Vector collection name               |
| `EMBEDDING_MODEL`    | `sentence-transformers/all-MiniLM-L6-v2`   | HuggingFace embedding model id       |
| `EU_AI_ACT_PDF`      | `./data/eu_ai_act.pdf`                     | Source PDF for ingestion             |
| `TENANT_ID`          | `rag-demo`                                 | Tenant attribution for the audit log |

---

## Project layout

```
auracomply-rag/
├── api.py                # FastAPI service
├── app.py                # Streamlit UI
├── auracomply_client.py  # MCP seal_decision client
├── config.py             # Pydantic settings
├── ingest.py             # PDF → ChromaDB pipeline
├── rag_chain.py          # LCEL RAG chain
├── tests/test_rag.py     # pytest suite (10 tests)
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── requirements.txt
```
