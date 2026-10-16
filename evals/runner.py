"""Eval runner over evals/golden_set.jsonl.

Two modes:

  Smoke (default): loads a small subset of questions (DEFAULT 5) and
  runs them against the pipeline. Used by tests/test_smoke_eval.py
  with a fake LLM scripted to a canned citation pattern so the
  hermetic suite can verify the eval scaffolding without paying for
  any model calls.

  Full: gated on RAG_EVAL_FULL=1 + ANTHROPIC_API_KEY set. Runs the
  whole golden set against the configured Claude model, prints a JSON
  report with citation precision/recall/F1 plus groundedness verdict
  distribution, and exits non-zero if the smoke gate (citation F1 >=
  RAG_EVAL_F1_GATE, default 0.5) fails.

The runner does not mutate the working tree — no chain rows are
written when called with `seal_client=None` and the caller patches
out the SDK. In CI the smoke variant runs hermetically; the full
variant runs only when explicitly opted in.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.jsonl"
DEFAULT_SMOKE_IDS = ("q01", "q06", "q20", "q25", "q39")  # one per macro-area


@dataclass(frozen=True)
class GoldenItem:
    id: str
    category: str
    question: str
    expected_citations: list[str]
    notes: str


@dataclass(frozen=True)
class EvalRecord:
    item: GoldenItem
    cited: list[str]
    groundedness: dict
    precision: float
    recall: float
    f1: float


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        items.append(GoldenItem(
            id=raw["id"],
            category=raw.get("category", ""),
            question=raw["question"],
            expected_citations=list(raw["expected_citations"]),
            notes=raw.get("notes", ""),
        ))
    return items


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _score(expected: list[str], got: list[str]) -> tuple[float, float, float]:
    if not expected:
        return 0.0, 0.0, 0.0
    if not got:
        return 0.0, 0.0, 0.0
    exp_set = set(map(_canon, expected))
    got_set = set(map(_canon, got))
    tp = len(exp_set & got_set)
    precision = tp / len(got_set)
    recall = tp / len(exp_set)
    return precision, recall, _f1(precision, recall)


def _canon(ident: str) -> str:
    """Normalise identifiers so 'Article 5' and '5' and 'annex-iii' and
    'Annex III' compare equal."""
    s = ident.strip().lower()
    s = s.replace("article ", "").replace("annex ", "annex-")
    s = s.replace("recital ", "rec-").replace("rec-(", "rec-").rstrip(")")
    s = s.replace(" ", "")
    return s


def run(
    items: Iterable[GoldenItem],
    query_fn: Callable[[str], object],
) -> list[EvalRecord]:
    """Apply `query_fn` to each item. `query_fn(question)` must return
    an object with `.articles_cited` (list[str]) and `.groundedness`
    (dict). The pipeline's `RagResult` already satisfies that contract.

    The caller is responsible for any LLM mocking / seal_client
    injection. The runner is intentionally policy-free."""
    records: list[EvalRecord] = []
    for item in items:
        result = query_fn(item.question)
        cited = list(getattr(result, "articles_cited", []) or [])
        groundedness = dict(getattr(result, "groundedness", {}) or {})
        precision, recall, f1 = _score(item.expected_citations, cited)
        records.append(EvalRecord(
            item=item,
            cited=cited,
            groundedness=groundedness,
            precision=precision,
            recall=recall,
            f1=f1,
        ))
    return records


def summarise(records: list[EvalRecord]) -> dict:
    """Aggregate metrics across one eval run. Stable JSON shape — the
    smoke test asserts the schema directly."""
    if not records:
        return {
            "n": 0,
            "citation": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "groundedness": {},
            "per_category": {},
        }
    n = len(records)
    p = sum(r.precision for r in records) / n
    r_ = sum(r.recall for r in records) / n
    f1 = sum(r.f1 for r in records) / n
    verdicts = Counter(r.groundedness.get("verdict", "missing") for r in records)
    per_cat: dict[str, dict[str, float]] = {}
    by_cat: dict[str, list[EvalRecord]] = {}
    for r in records:
        by_cat.setdefault(r.item.category, []).append(r)
    for cat, rs in by_cat.items():
        per_cat[cat] = {
            "n": len(rs),
            "precision": sum(x.precision for x in rs) / len(rs),
            "recall": sum(x.recall for x in rs) / len(rs),
            "f1": sum(x.f1 for x in rs) / len(rs),
        }
    return {
        "n": n,
        "citation": {"precision": p, "recall": r_, "f1": f1},
        "groundedness": dict(verdicts),
        "per_category": per_cat,
    }


def smoke_subset(items: list[GoldenItem]) -> list[GoldenItem]:
    by_id = {it.id: it for it in items}
    return [by_id[i] for i in DEFAULT_SMOKE_IDS if i in by_id]


def _is_full_mode() -> bool:
    return os.getenv("RAG_EVAL_FULL") == "1"


def main() -> int:
    items = load_golden()
    if not _is_full_mode():
        sys.stderr.write(
            "RAG_EVAL_FULL is not set; refusing to call the real LLM. "
            "Set RAG_EVAL_FULL=1 to opt in.\n"
        )
        return 2
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.stderr.write("ANTHROPIC_API_KEY is required in full mode.\n")
        return 2

    # The full mode imports the production query() — the smoke path
    # avoids this so it stays hermetic.
    from rag_chain import query

    records = run(items, lambda q: query(q))
    summary = summarise(records)
    print(json.dumps(summary, indent=2, sort_keys=True))
    gate = float(os.getenv("RAG_EVAL_F1_GATE", "0.5"))
    return 0 if summary["citation"]["f1"] >= gate else 1


if __name__ == "__main__":
    sys.exit(main())
