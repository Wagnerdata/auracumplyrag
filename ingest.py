"""EU AI Act PDF ingestion.

Splits the regulation by three chunk families — Article N, Recital (N),
and Annex III|IV — embeds with bge-m3 (config.settings.embedding_model),
persists to Chroma, computes the PDF's SHA-256 once, and seals a
`corpus_ingested/v1` event into the AuraComply chain so every later
`agent_decision/v1` can carry a corpus_version that ties back to a
specific knowledge state.

The seal is the LAST step; if any earlier step raises, no chain row is
written. There is no "ingest succeeded but the seal failed" half-state
to reason about — under BUFFERED policy the seal call returns
immediately and the SDK's background flusher takes over.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

import pymupdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

if TYPE_CHECKING:
    from auracomply import AuracomplyClient, SealResult

logger = logging.getLogger(__name__)


# ── Header patterns ──────────────────────────────────────────────────
# Article: "Article N\nTitle" at line start. The strict pattern avoids
# matching in-body cross-references ("Article 290 TFEU", "Article 6(2)").
ARTICLE_HEADER_PATTERN = re.compile(
    r"^Article\s+(\d+)\s*\n([^\n]{3,150})$", re.MULTILINE
)
# Recital: "(N)" at line start. EUR-Lex Recitals are short paragraphs
# numbered in parentheses; the number can have up to 4 digits in
# longer regulations. We capture the body up to the next "(M)" or to
# the first "CHAPTER" / "Article 1" boundary downstream.
RECITAL_HEADER_PATTERN = re.compile(r"^\((\d{1,4})\)\s", re.MULTILINE)
# Annex header. We only specifically extract III and IV because they
# carry the high-risk taxonomy (III) and the technical documentation
# requirements (IV); other annexes are reference appendices.
ANNEX_HEADER_PATTERN = re.compile(
    r"^ANNEX\s+(III|IV)\b\s*\n([^\n]{0,200})", re.MULTILINE
)
CHAPTER_PATTERN = re.compile(r"\bCHAPTER\s+([IVXLC]+)\b")

FALLBACK_CHUNK_SIZE = 512
FALLBACK_CHUNK_OVERLAP = 50

# Where Recitals end. Real EUR-Lex PDFs put "HAVE ADOPTED THIS
# REGULATION:" as the last line of the recitals block before Article 1.
RECITALS_END_MARKER = re.compile(
    r"HAVE\s+ADOPTED\s+THIS\s+REGULATION", re.IGNORECASE
)


@dataclass
class ArticleChunk:
    article_number: str        # "5", "rec-12", "annex-iii", or "chunk-N"
    title: str
    chapter: str
    content: str
    chunk_kind: str            # "article" | "recital" | "annex" | "fallback"


# ── PDF loading + hashing ────────────────────────────────────────────
def _load_pdf_text(pdf_path: str) -> str:
    """Extract text using PyMuPDF.

    EUR-Lex PDFs extracted via pypdf often have spaces injected inside
    words (e.g. "Ar ticle") that break the article-boundary regex.
    PyMuPDF respects the multi-column layout and returns clean tokens.
    """
    with pymupdf.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)


def _sha256_file(pdf_path: str) -> str:
    """Stable hex digest of the source PDF. Streamed so a multi-MB
    file doesn't sit in memory just to be hashed."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(64 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _clean_title(raw: str) -> str:
    """Trim trailing PDF artefacts (backticks, stray punctuation)."""
    return raw.strip().rstrip("`'\"-:. ")


# ── Splitters ────────────────────────────────────────────────────────
def _split_by_article(full_text: str) -> list[ArticleChunk]:
    chunks: list[ArticleChunk] = []
    matches = list(ARTICLE_HEADER_PATTERN.finditer(full_text))
    if not matches:
        return []

    chapter_positions = [
        (m.start(), m.group(1)) for m in CHAPTER_PATTERN.finditer(full_text)
    ]

    def chapter_at(pos: int) -> str:
        active = "Unknown"
        for cpos, cname in chapter_positions:
            if cpos <= pos:
                active = cname
            else:
                break
        return active

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        article_num = match.group(1)
        title = _clean_title(match.group(2)) or f"Article {article_num}"
        chunks.append(
            ArticleChunk(
                article_number=article_num,
                title=title[:200],
                chapter=chapter_at(start),
                content=body,
                chunk_kind="article",
            )
        )
    return chunks


def _split_recitals(full_text: str) -> list[ArticleChunk]:
    """Capture the numbered Recitals block (everything before Article 1
    that matches the "(N)" enumeration pattern, bounded by the
    "HAVE ADOPTED THIS REGULATION" marker)."""
    end_marker = RECITALS_END_MARKER.search(full_text)
    bound = end_marker.start() if end_marker else len(full_text)
    region = full_text[:bound]
    matches = list(RECITAL_HEADER_PATTERN.finditer(region))
    if not matches:
        return []

    chunks: list[ArticleChunk] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else bound
        body = region[start:end].strip()
        if len(body) < 20:
            # Number-only false positive (e.g. footnote refs); skip.
            continue
        number = match.group(1)
        chunks.append(
            ArticleChunk(
                article_number=f"rec-{number}",
                title=f"Recital ({number})",
                chapter="Recitals",
                content=body,
                chunk_kind="recital",
            )
        )
    return chunks


def _split_annexes(full_text: str) -> list[ArticleChunk]:
    """Slice ANNEX III and ANNEX IV blocks out, each as its own chunk
    family. Body runs until the next ANNEX header or EOF."""
    # Look up ANY annex boundary so we know where to stop a III/IV body.
    any_annex = re.compile(r"^ANNEX\s+[IVXLC]+\b", re.MULTILINE)
    boundaries = [m.start() for m in any_annex.finditer(full_text)] + [len(full_text)]

    chunks: list[ArticleChunk] = []
    for match in ANNEX_HEADER_PATTERN.finditer(full_text):
        start = match.start()
        end = next(b for b in boundaries if b > start)
        body = full_text[start:end].strip()
        roman = match.group(1)
        title = _clean_title(match.group(2)) or f"Annex {roman}"
        chunks.append(
            ArticleChunk(
                article_number=f"annex-{roman.lower()}",
                title=title[:200],
                chapter=f"Annex {roman}",
                content=body,
                chunk_kind="annex",
            )
        )
    return chunks


def _fallback_split(full_text: str) -> list[ArticleChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=FALLBACK_CHUNK_SIZE,
        chunk_overlap=FALLBACK_CHUNK_OVERLAP,
    )
    pieces = splitter.split_text(full_text)
    return [
        ArticleChunk(
            article_number=f"chunk-{idx}",
            title=f"Fallback chunk {idx}",
            chapter="Unknown",
            content=piece,
            chunk_kind="fallback",
        )
        for idx, piece in enumerate(pieces)
    ]


def _to_documents(chunks: Iterable[ArticleChunk]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=FALLBACK_CHUNK_SIZE,
        chunk_overlap=FALLBACK_CHUNK_OVERLAP,
    )
    docs: list[Document] = []
    for chunk in chunks:
        common_meta = {
            "article_number": chunk.article_number,
            "title": chunk.title,
            "chapter": chunk.chapter,
            "chunk_kind": chunk.chunk_kind,
        }
        if len(chunk.content) <= FALLBACK_CHUNK_SIZE:
            docs.append(Document(page_content=chunk.content, metadata=common_meta))
            continue
        for sub_idx, sub in enumerate(splitter.split_text(chunk.content)):
            docs.append(
                Document(
                    page_content=sub,
                    metadata={**common_meta, "sub_chunk": sub_idx},
                )
            )
    return docs


# ── Embeddings + store ───────────────────────────────────────────────
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_vector_store(embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


# ── Corpus version persistence ───────────────────────────────────────
def _corpus_version_path() -> Path:
    return Path(settings.chroma_persist_dir) / "corpus_version.json"


def write_corpus_version(source_doc_sha256: str, ingested_at: datetime) -> None:
    """Persist the corpus version next to the Chroma store so
    rag_chain.query() can read it without depending on the kernel."""
    payload = {
        "source_doc_sha256": source_doc_sha256,
        "ingested_at": ingested_at.isoformat(),
        "collection_name": settings.collection_name,
        "embedding_model": settings.embedding_model,
    }
    _corpus_version_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_corpus_version() -> dict | None:
    """Return the last persisted corpus_version, or None if no ingest
    has run yet (an agent_decision/v1 cannot be sealed in that case)."""
    p = _corpus_version_path()
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ── Public entry point ───────────────────────────────────────────────
def ingest(
    pdf_path: str | None = None,
    seal_client: "AuracomplyClient | None" = None,
) -> dict:
    """Run the full ingestion pipeline.

    Returns a dict with chunk counts, the persist dir, the PDF's
    sha256, and the SealResult produced by the corpus_ingested/v1 seal.
    The seal_client argument is the SDK's `AuracomplyClient` — tests
    inject one bound to a respx MockTransport.
    """
    pdf_path = pdf_path or settings.eu_ai_act_pdf
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Hashing source PDF: %s", pdf_path)
    source_doc_sha256 = _sha256_file(pdf_path)

    logger.info("Loading PDF text via PyMuPDF")
    text = _load_pdf_text(pdf_path)

    article_chunks = _split_by_article(text)
    recital_chunks = _split_recitals(text)
    annex_chunks = _split_annexes(text)
    all_chunks: list[ArticleChunk] = article_chunks + recital_chunks + annex_chunks

    if not all_chunks:
        logger.warning("No structural boundaries found; using fallback splitter")
        all_chunks = _fallback_split(text)

    logger.info(
        "Split into %d articles + %d recitals + %d annexes (%d total chunks)",
        len(article_chunks), len(recital_chunks), len(annex_chunks), len(all_chunks),
    )

    documents = _to_documents(all_chunks)
    logger.info("Embedding %d documents with %s", len(documents), settings.embedding_model)

    embeddings = get_embeddings()
    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    store.add_documents(documents)

    ingested_at = datetime.now(timezone.utc)
    write_corpus_version(source_doc_sha256, ingested_at)

    seal_result: "SealResult"
    if seal_client is None:
        # Production path imports the singleton lazily so the test path
        # (which sets seal_client via api.py) does not pay the import.
        from seal_client import get_seal_client

        seal_client = get_seal_client()

    # corpus_ingested/v1 payload — see sdk/auracomply/schemas/dora_events_v1.py.
    # Time source is wall_clock; an NTP/TSA upgrade is a Phase 5 concern.
    seal_payload = {
        "schema_version": "corpus_ingested/v1",
        "source_doc_sha256": source_doc_sha256,
        "ingested_at": ingested_at.isoformat(),
        "collection_name": settings.collection_name,
        "embedding_model": settings.embedding_model[:128],
        "timestamp": ingested_at.isoformat(),
        "time_source": "wall_clock",
    }
    seal_result = seal_client.seal(
        product="rag-eu-ai-act",
        step="corpus_ingested",
        payload=seal_payload,
        idempotency_key=f"corpus:{source_doc_sha256}",
    )

    return {
        "articles": len(article_chunks),
        "documents": len(documents),
        "persist_dir": settings.chroma_persist_dir,
        "source_doc_sha256": source_doc_sha256,
        "seal": seal_result,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = ingest()
    print(f"Ingestion complete: articles={result['articles']} "
          f"documents={result['documents']} "
          f"source_doc_sha256={result['source_doc_sha256'][:12]}…")
