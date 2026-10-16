"""Hermetic smoke eval — exercises the runner + a 5-question subset
of the golden set against a scripted LLM. The smoke test does NOT
guard a citation-quality threshold; the gate value belongs in the
RAG_EVAL_FULL path against a real model. This is here to verify the
scaffolding (load, score, summarise, schema) so a future change can't
break the runner silently.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document


def _stage_corpus_version(persist_dir: str) -> None:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    (Path(persist_dir) / "corpus_version.json").write_text(json.dumps({
        "source_doc_sha256": "0" * 64,
        "ingested_at": datetime(2026, 6, 13, 10, 0, tzinfo=timezone.utc).isoformat(),
        "collection_name": "eu_ai_act_test",
        "embedding_model": "BAAI/bge-m3",
    }))


def _fixed_docs():
    return [
        Document(
            page_content="Article 5 lists prohibited AI practices.",
            metadata={
                "article_number": "5",
                "title": "Prohibited practices",
                "chapter": "II",
                "chunk_kind": "article",
            },
        ),
    ]


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, q):
        return list(self._docs)


def test_smoke_subset_covers_five_macro_areas():
    """The smoke subset must contain exactly the 5 IDs we picked, and
    they must span 5 distinct categories — otherwise the smoke is
    not actually representative."""
    from evals.runner import DEFAULT_SMOKE_IDS, load_golden, smoke_subset

    items = load_golden()
    subset = smoke_subset(items)
    assert len(subset) == 5
    ids = {it.id for it in subset}
    assert ids == set(DEFAULT_SMOKE_IDS)
    categories = {it.category for it in subset}
    assert len(categories) == 5, f"smoke subset must span 5 categories, got {categories}"


def test_golden_set_has_at_least_50_items():
    from evals.runner import load_golden

    items = load_golden()
    assert len(items) >= 50, f"golden set is undersized: {len(items)}"


def test_runner_scores_perfect_match_as_f1_one():
    """When the predicted citations exactly equal the expected ones,
    the runner must score precision=recall=f1=1.0."""
    from evals.runner import EvalRecord, GoldenItem, run

    items = [GoldenItem(
        id="q-test", category="x", question="Q?",
        expected_citations=["5", "annex-iii"], notes="",
    )]

    class _Fake:
        def __init__(self, cited):
            self.articles_cited = cited
            self.groundedness = {"verdict": "yes", "confidence": 0.9, "reason": "ok"}

    records = run(items, lambda q: _Fake(["5", "annex-iii"]))
    assert len(records) == 1
    r = records[0]
    assert r.precision == 1.0
    assert r.recall == 1.0
    assert r.f1 == 1.0


def test_runner_summarises_into_stable_schema():
    """Tests pin the JSON shape of `summarise` because that schema is
    what the GitHub Actions job will surface in PR comments — drift
    here breaks downstream tooling silently."""
    from evals.runner import EvalRecord, GoldenItem, summarise

    item = GoldenItem(
        id="x", category="prohibited", question="?", expected_citations=["5"], notes="",
    )
    record = EvalRecord(
        item=item, cited=["5"], groundedness={"verdict": "yes"},
        precision=1.0, recall=1.0, f1=1.0,
    )
    summary = summarise([record])
    assert set(summary.keys()) == {"n", "citation", "groundedness", "per_category"}
    assert set(summary["citation"].keys()) == {"precision", "recall", "f1"}
    assert summary["n"] == 1


def test_smoke_eval_runs_end_to_end_against_fake_pipeline(
    seal_recorder, fake_llm, fake_reranker, monkeypatch
):
    """Pull the 5 smoke questions through the actual pipeline with a
    scripted LLM. The LLM is told to cite Article 5 in every answer
    so the runner produces non-trivial metrics; the test asserts the
    summary schema, the chain row count, and that the chain stays
    payload-clean (no plaintext questions/answers)."""
    from config import settings
    import rag_chain
    from evals.runner import load_golden, run, smoke_subset, summarise

    _stage_corpus_version(settings.chroma_persist_dir)
    monkeypatch.setattr(
        rag_chain, "_build_hybrid_retriever", lambda k: _FakeRetriever(_fixed_docs())
    )

    items = smoke_subset(load_golden())
    # Per question: one answer + one groundedness JSON = 2 LLM calls.
    fake_llm.script([s for _ in items for s in [
        "Article 5 is the relevant rule. See (Article 5).",
        '{"verdict": "yes", "confidence": 0.85, "reason": "Cited Art 5"}',
    ]])

    from rag_chain import query

    records = run(items, lambda q: query(q))
    assert len(records) == 5
    summary = summarise(records)

    # The runner's contract.
    assert summary["n"] == 5
    assert 0.0 <= summary["citation"]["precision"] <= 1.0
    assert 0.0 <= summary["citation"]["recall"] <= 1.0
    assert summary["groundedness"]["yes"] == 5

    # One sealed event per question.
    assert len(seal_recorder.posted_payloads) == 5
    for posted in seal_recorder.posted_payloads:
        # Step + schema_version pinned.
        assert posted["step"] == "agent_decision"
        assert posted["payload"]["schema_version"] == "agent_decision/v1"
        # No tenant identity, no plaintext question.
        flat = json.dumps(posted).lower()
        assert "tenant_id" not in flat
        for item in items:
            # Each question text in lowercase must NOT appear verbatim.
            assert item.question.lower() not in flat
