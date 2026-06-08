"""EU AI Act PDF ingestion: split by article boundary, embed, persist to ChromaDB."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pymupdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)

# Article headers in the EU AI Act appear as "Article N\nTitle" at line start.
# The looser pattern also matches in-body cross-references (e.g. "Article 290 TFEU")
# and is only used to walk chapter context.
ARTICLE_HEADER_PATTERN = re.compile(
    r"^Article\s+(\d+)\s*\n([^\n]{3,150})$", re.MULTILINE
)
CHAPTER_PATTERN = re.compile(r"\bCHAPTER\s+([IVXLC]+)\b")
FALLBACK_CHUNK_SIZE = 512
FALLBACK_CHUNK_OVERLAP = 50


@dataclass
class ArticleChunk:
    article_number: str
    title: str
    chapter: str
    content: str


def _load_pdf_text(pdf_path: str) -> str:
    """Extract text using PyMuPDF (best for EUR-Lex multi-column layout).

    EUR-Lex PDFs extracted via pypdf often have spaces injected inside words
    (e.g. "Ar ticle"), which breaks article-boundary regex matching. PyMuPDF
    respects the multi-column layout and returns clean tokens.
    """
    with pymupdf.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)


def _clean_title(raw: str) -> str:
    """Trim trailing PDF artefacts (backticks, stray punctuation) from titles."""
    return raw.strip().rstrip("`'\"-:. ")


def _split_by_article(full_text: str) -> list[ArticleChunk]:
    """Split text on Article-header boundaries, capturing chapter context.

    Uses the strict header pattern "Article N\\nTitle" (line start + title on
    the next line) to avoid splitting on in-body cross-references like
    "Article 290 TFEU".
    """
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
        if len(chunk.content) <= FALLBACK_CHUNK_SIZE:
            docs.append(
                Document(
                    page_content=chunk.content,
                    metadata={
                        "article_number": chunk.article_number,
                        "title": chunk.title,
                        "chapter": chunk.chapter,
                    },
                )
            )
            continue
        for sub_idx, sub in enumerate(splitter.split_text(chunk.content)):
            docs.append(
                Document(
                    page_content=sub,
                    metadata={
                        "article_number": chunk.article_number,
                        "title": chunk.title,
                        "chapter": chunk.chapter,
                        "sub_chunk": sub_idx,
                    },
                )
            )
    return docs


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_vector_store(embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def ingest(pdf_path: str | None = None) -> dict:
    """Run the full ingestion pipeline. Returns counts for the caller."""
    pdf_path = pdf_path or settings.eu_ai_act_pdf
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Loading PDF: %s", pdf_path)
    text = _load_pdf_text(pdf_path)

    logger.info("Splitting by Article boundary")
    chunks = _split_by_article(text)
    if not chunks:
        logger.warning("No 'Article XX' boundaries found; using fallback splitter")
        chunks = _fallback_split(text)

    documents = _to_documents(chunks)
    logger.info("Embedding %d documents", len(documents))

    embeddings = get_embeddings()
    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    store.add_documents(documents)
    return {
        "articles": len(chunks),
        "documents": len(documents),
        "persist_dir": settings.chroma_persist_dir,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = ingest()
    print(f"Ingestion complete: {result}")
