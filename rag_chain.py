"""LCEL RAG pipeline: similarity search -> article-citing prompt -> Claude."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from config import settings
from ingest import get_vector_store

SYSTEM_PROMPT = """You are an EU AI Act compliance assistant.

Answer ONLY using the provided context. Every factual statement must cite the
relevant article inline using the format (Article XX). If the context does not
contain enough information to answer, say so explicitly and do not invent
articles or text. Do not hallucinate article numbers.

Context:
{context}
"""

USER_PROMPT = "Question: {question}"


@dataclass
class RagResult:
    answer: str
    sources: list[dict[str, Any]]
    articles_cited: list[str]


def _format_context(docs: list[Document]) -> str:
    parts: list[str] = []
    for doc in docs:
        article = doc.metadata.get("article_number", "?")
        title = doc.metadata.get("title", "")
        chapter = doc.metadata.get("chapter", "?")
        header = f"[Article {article} | Chapter {chapter}] {title}".strip()
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _sources_from_docs(docs: list[Document]) -> list[dict[str, Any]]:
    return [
        {
            "article_number": d.metadata.get("article_number"),
            "title": d.metadata.get("title"),
            "chapter": d.metadata.get("chapter"),
            "snippet": d.page_content[:300],
        }
        for d in docs
    ]


def _articles_from_docs(docs: list[Document]) -> list[str]:
    seen: list[str] = []
    for d in docs:
        article = str(d.metadata.get("article_number", ""))
        if article and article not in seen:
            seen.append(article)
    return seen


def build_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.anthropic_model,
        anthropic_api_key=settings.anthropic_api_key,
        max_tokens=1024,
        timeout=60,
    )


def build_rag_chain(k: int = 3, llm: ChatAnthropic | None = None):
    """Construct an LCEL chain returning a dict with answer/sources/articles."""
    store = get_vector_store()
    retriever = store.as_retriever(search_kwargs={"k": k})
    chat_llm = llm or build_llm()

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )

    answer_chain = (
        {
            "context": RunnableLambda(lambda x: _format_context(x["docs"])),
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | prompt
        | chat_llm
        | StrOutputParser()
    )

    return RunnableParallel(
        {
            "docs": RunnableLambda(lambda q: retriever.invoke(q)),
            "question": RunnablePassthrough(),
        }
    ) | RunnableLambda(
        lambda x: {
            "answer": answer_chain.invoke(x),
            "sources": _sources_from_docs(x["docs"]),
            "articles_cited": _articles_from_docs(x["docs"]),
        }
    )


def query(question: str, k: int = 3) -> RagResult:
    chain = build_rag_chain(k=k)
    result = chain.invoke(question)
    return RagResult(
        answer=result["answer"],
        sources=result["sources"],
        articles_cited=result["articles_cited"],
    )


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What are high-risk AI systems?"
    out = query(q)
    print(f"Q: {q}\n")
    print(f"A: {out.answer}\n")
    print(f"Articles cited: {out.articles_cited}")
