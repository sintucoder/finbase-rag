"""End-to-end plumbing tests using the fake embedder / fake LLM from fakes.py."""
import pytest
from fakes import FakeEmbedder, FakeLLM

from build_index import build_index
from src import config
from src.rag_pipeline import (NOT_FOUND_MESSAGE, RAGPipeline, cited_numbers, is_refusal)
from src.retriever import Retriever
from src.vector_store import VectorStore


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    if not list(config.DATA_DIR.glob("*.pdf")):
        pytest.skip("no PDFs in data/")
    index_dir = tmp_path_factory.mktemp("index")
    build_index(embedder=FakeEmbedder(), index_dir=index_dir)
    return VectorStore.load(index_dir)


def make_pipeline(store, llm):
    return RAGPipeline(Retriever(store, FakeEmbedder(), top_k=5, min_score=0.30), llm)


def test_index_roundtrip_keeps_chunks_and_info(store):
    assert store.info["embedding_model"] == "fake-hashing-embedder"
    assert store.index.ntotal == len(store.chunks) == store.info["num_chunks"] > 100


def test_search_returns_the_foreclosure_chunk_first(store):
    hits = Retriever(store, FakeEmbedder()).search("foreclosure charge if I close my personal loan after 18 months")
    assert hits[0].rank == 1 and hits[0].score >= hits[-1].score
    assert any("3% of the outstanding principal" in h.chunk.text and h.chunk.doc_id == "sample_5" for h in hits[:3])


def test_answer_with_citation_maps_to_the_cited_chunk(store):
    llm = FakeLLM(answer="The foreclosure charge is 3% [1].")
    pipe = make_pipeline(store, llm)
    r = pipe.ask("foreclosure charge if I close my personal loan after 18 months")
    assert r.answerable and [n for n, _ in r.sources] == [1]
    assert r.sources[0][1].chunk.doc_id == "sample_5"
    system, user = llm.calls[0][0]["content"], llm.calls[0][-1]["content"]
    assert "ONLY" in system and "Context passages" in user and "[1] Source:" in user


def test_unrelated_question_is_rejected_without_calling_the_llm(store):
    llm = FakeLLM()
    r = make_pipeline(store, llm).ask("What is the capital of France?")
    assert not r.answerable and r.refusal_reason == "below_threshold"
    assert r.answer.startswith(NOT_FOUND_MESSAGE) and r.sources == []
    assert llm.calls == []


def test_model_refusal_is_detected_and_has_no_sources(store):
    llm = FakeLLM(answer="I couldn\u2019t find enough information to answer this from the available FinBase knowledge base.")
    r = make_pipeline(store, llm).ask("foreclosure charge for personal loan")
    assert not r.answerable and r.refusal_reason == "model_declined" and r.sources == []


def test_follow_up_is_rewritten_and_history_is_sent(store):
    llm = FakeLLM(answer="It is ₹4,999 + GST [1].", rewrite="annual fee of the FinBase Metal credit card")
    history = [{"role": "user", "content": "What is the annual fee of the Luxe card?"},
               {"role": "assistant", "content": "It is ₹999 + GST."}]
    r = make_pipeline(store, llm).ask("And for the Metal one?", history)
    assert r.standalone_question == "annual fee of the FinBase Metal credit card"
    assert len(llm.calls) == 2                     # one rewrite call + one answer call
    answer_call = llm.calls[1]
    assert answer_call[1]["content"] == history[0]["content"]          # history passed as chat turns
    assert answer_call[-1]["content"].endswith("Question: annual fee of the FinBase Metal credit card")


def test_first_question_does_not_trigger_a_rewrite(store):
    llm = FakeLLM(answer="Answer [1].")
    make_pipeline(store, llm).ask("foreclosure charge personal loan")
    assert len(llm.calls) == 1


def test_citation_parser_and_refusal_detector():
    assert cited_numbers("A [1] and B [2][3], also [1, 4] and [9] and [x]", 4) == [1, 2, 3, 4]
    assert cited_numbers("no citations here", 3) == []
    assert is_refusal(NOT_FOUND_MESSAGE) and not is_refusal("The fee is 3% [1].")
