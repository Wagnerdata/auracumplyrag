# CLAUDE.md — auracomply-rag

Local protocol for the RAG showcase. The kernel's `CLAUDE.md` is the
canonical AI-agent behaviour protocol — **read it first**, then this
file. This document only records the rules that are specific to the
RAG layer; it does not duplicate the kernel's content.

**Read order on every new session:**
1. `../AuraComply/CLAUDE.md` (kernel — binding, full SDD §8 extract)
2. `../AuraComply/DECISIONS.md` (kernel decisions register)
3. `../AuraComply/docs/SEAL_CHAIN_SPEC.md` (chain spec; the RAG layer
   never reimplements seal_chain — the SDK does the sealing)
4. This file
5. The task

The kernel's §1–§8 apply transitively. Where this repo needs a tighter
or different rule, it appears below.

---

## §R1 — RAG-specific invariants

### Sealing

- **Never construct an HTTP client for `/audit/log` by hand.** Use
  `auracomply.AuracomplyClient.seal(...)`. The SDK is responsible for
  policy, buffering, HMAC, idempotency, retry, and the kernel's
  contract; reimplementing any of it in this repo is forbidden.
- **Never pass `tenant_id`, `client_id`, or any tenant-identifying
  field in a request body.** Tenant identity is the bearer token's
  job, server-side. Even body extras that the kernel silently ignores
  (D-010) are still wrong — they signal a misdesigned caller.
- **Never seal plaintext questions or answers.** The kernel's PII
  boundary rejects structurally-PII strings with 400, and the motto
  is "your data is yours; the audit is AuraComply's". The chain row
  is hashes + the bounded `decision.rationale_summary` (max 2000 chars,
  PII-screened) only. Questions are atable via `trace_id =
  sha256(question)`; answers are atable via the chunks that grounded
  them (`input_refs.chunk_hashes`) plus the rationale summary.
- **`AURACOMPLY_SEAL_POLICY=buffered` is the default.** Demo and dev
  environments should never set `strict` without an explicit reason;
  `warn` is demo-only and emits a loud log message.

### Retrieval and ingestion

- **Local-only retrieval stack.** Embeddings (bge-m3) and the reranker
  (bge-reranker-base) run in-process — no API keys for retrieval, no
  Cohere/OpenAI rerank. The only network call out of the RAG layer is
  to the configured Claude API for answer generation and groundedness.
- **Every ingest seals `corpus_ingested/v1`.** A successful ingest
  ends with one `client.seal(step="corpus_ingested", ...)` call. The
  resulting `source_doc_sha256` + `ingested_at` are then the
  `corpus_version` of every `agent_decision/v1` sealed afterwards —
  pinned by `corpus_version.json` next to the Chroma persist dir.
- **Chunk Recitals and Annexes III/IV alongside Articles.** The
  EU AI Act ingester recognises three chunk families: `Article N`
  (strict header), `Recital (N)` (numbered enumeration), and
  `ANNEX III|IV` (Roman). Each chunk carries `chunk_kind` metadata.

### API surface

- **Bearer auth on every endpoint that touches retrieval or ingest.**
  `RAG_API_TOKEN` is the secret; comparison uses `hmac.compare_digest`.
  `/health` is the only endpoint allowed to be anonymous (no tenant
  data) — same exception the kernel makes for its own `/health`.
- **`AURACOMPLY_DEMO_PROFILE=1` is the only way to skip auth.** Never
  default it to on; never document it without the loud warning that
  it disables tenant boundary protection.

### Tests

- **Hermetic by default.** Mock the SDK via its transport seam
  (`AuracomplyClient(_transport=respx.MockTransport(...))`). No real
  network, no real Claude API, no real disk side-effects beyond
  `tmp_path`.
- **Smoke eval lives in `tests/`**, runs in CI on every push.
  Full eval (`evals/runner.py` with `RAG_EVAL_FULL=1`) requires
  `ANTHROPIC_API_KEY` and is opt-in only.

---

## §R2 — Out of scope for this repo

- **The kernel.** Any change to `../AuraComply/` belongs in a kernel
  PR with its own plan + DECISIONS entry. RAG bug → fix here. Kernel
  bug → fix there.
- **Schema evolution.** The vendored schemas (`AgentDecisionV1`,
  `CorpusIngestedV1`, etc.) are byte-identical to the kernel's
  `shared/schemas/`. Don't edit them here; the kernel owns them.
- **Streamlit UI redesign.** `app.py` exists; show new response
  fields when the API returns them, do not rebuild the page.

---

## Reference

- EU AI Act Reg. 2024/1689 — Articles 6 (high-risk), 9 (risk management),
  13 (traceability), 14 (human oversight), 50 (transparency), 51–55
  (GPAI), Annexes III–IV (high-risk use cases, technical documentation).
- Source PDF lives at `data/eu_ai_act.pdf` (EUR-Lex publication;
  `source_doc_sha256` recomputed at every ingest).
