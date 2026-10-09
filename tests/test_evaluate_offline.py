"""Checks the evaluation code itself (metric maths, file output) with the fake embedder / LLM.
The numbers it produces are NOT retrieval-quality numbers for the real model."""
import json

import pytest
from fakes import FakeEmbedder, FakeLLM

import evaluate
from build_index import build_index
from src import config
from src.rag_pipeline import RAGPipeline
from src.retriever import RetrievedChunk, Retriever
from src.chunking import Chunk
from src.vector_store import VectorStore


def chunk(doc, text, cid="c"):
    return Chunk(cid, doc.split(".")[0], doc, "T", "S", "section", 1, 1, text)


def test_retrieval_metrics_rank_and_recall():
    hits = [RetrievedChunk(chunk("a.pdf", "nothing"), 0.9, 1),
            RetrievedChunk(chunk("b.pdf", "the fee is 3% here"), 0.8, 2),
            RetrievedChunk(chunk("a.pdf", "limit is 5,000"), 0.7, 3)]
    gold = [{"docs": ["b.pdf"], "evidence": ["3% here"]}, {"docs": ["a.pdf"], "evidence": ["5,000"]},
            {"docs": ["c.pdf"], "evidence": ["missing"]}]
    m = evaluate.retrieval_metrics(hits, gold)
    assert m["gold_ranks"] == [2, 3, None]
    assert m["hit@1"] == 0 and m["hit@3"] == 1 and m["mrr"] == 0.5
    assert m["recall@3"] == pytest.approx(2 / 3)


def test_wrong_document_does_not_count_as_a_hit():
    hits = [RetrievedChunk(chunk("a.pdf", "the fee is 3%"), 0.9, 1)]
    assert evaluate.retrieval_metrics(hits, [{"docs": ["b.pdf"], "evidence": ["3%"]}])["hit@1"] == 0


def test_keyword_check_normalises_rupees_and_commas():
    assert evaluate.keywords_present("The limit is ₹1,00,000 per day.", [["100000"]])
    assert not evaluate.keywords_present("The limit is ₹50,000.", [["100000"]])
    assert evaluate.keywords_present("T+2 business days", [["t+2"], ["business"]])


def test_numeric_groundedness_flags_invented_numbers_only():
    context = "[1] Source: X | Page 3 | Section 6.2\nForeclosure charge is 3% of the outstanding principal."
    ok = evaluate.numeric_groundedness("The charge is 3% [1].", context, "foreclosure after 18 months?", [])
    bad = evaluate.numeric_groundedness("The charge is 4.5% after 18 months [1].", context, "foreclosure after 18 months?", [])
    derived = evaluate.numeric_groundedness("4 x 100 = 400", "100 per day", "4 days late", ["400"])
    assert ok == [] and bad == ["4.5"] and derived == []


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("idx")
    build_index(embedder=FakeEmbedder(), index_dir=index_dir)
    retriever = Retriever(VectorStore.load(index_dir), FakeEmbedder(), top_k=5, min_score=0.30)
    return retriever


def test_run_evaluation_end_to_end_writes_reports(setup, tmp_path, monkeypatch):
    questions = json.loads(config.EVAL_FILE.read_text(encoding="utf-8"))
    retriever = setup
    # retrieval-only
    rows, summary = evaluate.run_evaluation(questions, retriever, None, verbose=False)
    assert len(rows) == len(questions) == 30
    assert 0 <= summary["overall"]["hit@5"] <= 1 and "answer_correct_answerable" not in summary["overall"]
    # full run with a fake LLM that cites passage 1
    pipe = RAGPipeline(retriever, FakeLLM(answer="Answer text [1]."))
    rows2, summary2 = evaluate.run_evaluation(questions[:6] + questions[24:28], retriever, pipe, verbose=False)
    assert "citation_hit" in summary2["overall"] and all("answer" in r for r in rows2)
    # file output goes to a temp dir
    monkeypatch.setattr(config, "EVAL_OUT_DIR", tmp_path)
    evaluate.write_outputs(rows2, summary2, full_run=True)
    assert (tmp_path / "results.csv").exists() and (tmp_path / "results_summary.md").exists()
    assert "manual_correct" in (tmp_path / "results.csv").read_text(encoding="utf-8-sig")


def test_every_gold_evidence_exists_in_the_pdfs():
    """If this fails the evaluation set is wrong (PDFs changed), not the RAG system."""
    from src.chunking import build_chunks
    from src.document_loader import load_all_pdfs
    from src.utils import norm_evidence
    chunks, _ = build_chunks(load_all_pdfs())
    for q in json.loads(config.EVAL_FILE.read_text(encoding="utf-8")):
        for g in q["gold"]:
            assert any(c.file_name in g["docs"] and all(norm_evidence(e) in norm_evidence(c.text) for e in g["evidence"])
                       for c in chunks), (q["id"], g)
