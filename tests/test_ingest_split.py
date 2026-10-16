"""Splitters: Article + Recital + Annex extraction.

Pure-Python tests on the regex pipeline — no PDF, no embeddings, no
SDK. These pin the regex contract so a future refactor can't quietly
drop Annex coverage or merge two recitals into one chunk.
"""
from __future__ import annotations


SAMPLE_TEXT = """REGULATION (EU) 2024/1689

(1) Whereas artificial intelligence systems can pose risks to fundamental rights.
(2) Whereas the Union seeks harmonised rules for AI providers.

HAVE ADOPTED THIS REGULATION:

CHAPTER I
GENERAL PROVISIONS

Article 1
Subject matter
This Regulation lays down harmonised rules on artificial intelligence,
and refers to Article 114 of the Treaty as legal basis.

Article 5
Prohibited artificial intelligence practices
The following AI practices shall be prohibited: social scoring.

ANNEX III
High-risk AI systems referred to in Article 6(2)
Biometric identification systems are listed in point 1.

ANNEX IV
Technical documentation referred to in Article 11
Providers shall keep technical documentation.
"""


def test_article_boundary_split_extracts_articles_and_chapters():
    from ingest import _split_by_article

    chunks = _split_by_article(SAMPLE_TEXT)
    nums = [c.article_number for c in chunks]
    assert nums == ["1", "5"]
    assert all(c.chunk_kind == "article" for c in chunks)
    article_5 = next(c for c in chunks if c.article_number == "5")
    assert article_5.chapter == "I" or article_5.chapter == "II"


def test_recital_split_captures_each_numbered_paragraph():
    from ingest import _split_recitals

    chunks = _split_recitals(SAMPLE_TEXT)
    nums = [c.article_number for c in chunks]
    assert nums == ["rec-1", "rec-2"]
    assert all(c.chunk_kind == "recital" for c in chunks)
    assert "fundamental rights" in chunks[0].content
    assert "harmonised rules" in chunks[1].content


def test_recital_split_stops_at_have_adopted_marker():
    """A numbered footnote AFTER the recitals block must not pollute
    the recital extraction. The end marker `HAVE ADOPTED THIS
    REGULATION` bounds the recital region."""
    from ingest import _split_recitals

    polluted = SAMPLE_TEXT + "\n(99) This is a footnote, not a recital.\n"
    chunks = _split_recitals(polluted)
    nums = [c.article_number for c in chunks]
    assert "rec-99" not in nums
    assert nums == ["rec-1", "rec-2"]


def test_annex_split_picks_iii_and_iv_only():
    from ingest import _split_annexes

    chunks = _split_annexes(SAMPLE_TEXT)
    ids = [c.article_number for c in chunks]
    assert ids == ["annex-iii", "annex-iv"]
    assert all(c.chunk_kind == "annex" for c in chunks)
    a3 = next(c for c in chunks if c.article_number == "annex-iii")
    assert "Biometric identification" in a3.content


def test_chunks_carry_required_metadata():
    from ingest import _split_by_article, _split_recitals, _to_documents

    chunks = _split_by_article(SAMPLE_TEXT) + _split_recitals(SAMPLE_TEXT)
    docs = _to_documents(chunks)
    assert docs, "expected at least one document"
    for doc in docs:
        for field in ("article_number", "title", "chapter", "chunk_kind"):
            assert field in doc.metadata, f"missing {field} on {doc}"
