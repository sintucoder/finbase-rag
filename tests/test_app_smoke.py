"""Runs the real app.py with Streamlit's AppTest, using the fake embedder/LLM."""
import pytest
from fakes import FakeEmbedder, FakeLLM
import streamlit as st
from streamlit.testing.v1 import AppTest

from build_index import build_index
from src import config, rag_pipeline
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.utils import data_fingerprint
from src.vector_store import VectorStore


@pytest.fixture()
def app(tmp_path, monkeypatch):
    if not list(config.DATA_DIR.glob("*.pdf")):
        pytest.skip("no PDFs in data/")
    store = build_index(embedder=FakeEmbedder(), index_dir=tmp_path)
    llm = FakeLLM(answer="The foreclosure charge is 3% of the outstanding principal [1].", rewrite="foreclosure charge personal loan")
    pipeline = RAGPipeline(Retriever(store, FakeEmbedder(), top_k=5, min_score=0.30), llm)
    # never touch the user's real index/ folder
    monkeypatch.setattr(VectorStore, "exists", staticmethod(lambda directory=None: True))
    monkeypatch.setattr(VectorStore, "read_info", staticmethod(lambda directory=None: {"data_fingerprint": data_fingerprint()}))
    monkeypatch.setattr(rag_pipeline, "build_pipeline", lambda: pipeline)
    st.cache_resource.clear()      # the app caches its pipeline; every test needs its own
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    return at, llm


def test_answer_is_shown_with_sources(app):
    at, _ = app
    at.chat_input[0].set_value("foreclosure charge if I close my personal loan after 18 months").run()
    assert not at.exception
    texts = [m.markdown[0].value for m in at.chat_message]
    assert "3% of the outstanding principal [1]" in texts[1]
    assert any(e.label.startswith("Sources (") for e in at.expander)
    assert any("Personal Loans Master Policy" in md.value for md in at.markdown)


def test_off_topic_question_is_declined_without_sources(app):
    at, llm = app
    at.chat_input[0].set_value("What is the capital of France?").run()
    assert not at.exception
    assert "couldn't find enough information" in at.chat_message[1].markdown[0].value
    assert not any(e.label.startswith("Sources (") for e in at.expander)
    assert llm.calls == []


def test_follow_up_uses_history_and_new_conversation_clears_it(app):
    at, llm = app
    at.chat_input[0].set_value("foreclosure charge for a personal loan").run()
    at.chat_input[0].set_value("and what if I close it after 30 months?").run()
    assert not at.exception and len(at.chat_message) == 4
    assert any("Rewrite the user's last question" in c[0]["content"] for c in llm.calls)   # rewrite step ran
    new_chat = next(b for b in at.sidebar.button if b.label == "New conversation")
    new_chat.click().run()
    assert len(at.chat_message) == 0


def test_example_button_asks_the_question(app):
    at, _ = app
    next(b for b in at.sidebar.button if b.label.startswith("What is the foreclosure charge")).click().run()
    assert not at.exception and len(at.chat_message) == 2
