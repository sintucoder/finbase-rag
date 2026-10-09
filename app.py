"""FinBase customer support assistant (Streamlit UI).

    streamlit run app.py
"""
import streamlit as st

from src import config
from src.rag_pipeline import RAGResponse, build_pipeline
from src.utils import data_fingerprint
from src.vector_store import VectorStore

st.set_page_config(page_title="FinBase Support Assistant", page_icon="💬", layout="centered")

EXAMPLE_QUESTIONS = [
    "What is the foreclosure charge on a personal loan?",
    "What happens if my UPI payment fails but money is debited?",
    "What is the interest rate for senior citizens on a 1-year FD?",
    "How much is the annual fee of the Luxe credit card?",
    "What are the Video KYC timings?",
    "What is the interest rate on a FinBase home loan?",
]


@st.cache_resource(show_spinner="Loading the embedding model and the search index...")
def load_pipeline():
    return build_pipeline()


def response_to_message(response: RAGResponse, question: str) -> dict:
    """Keep only plain data in the session state."""
    sources = []
    for number, hit in response.sources:
        chunk = hit.chunk
        sources.append({
            "number": number, "title": chunk.doc_title, "pages": chunk.pages_label, "section": chunk.section,
            "score": round(hit.score, 2), "text": chunk.text,
            "also_on_pages": chunk.also_on_pages,
        })
    note = f"Best match similarity {response.top_score:.2f} · {response.latency_s}s"
    if response.standalone_question != question:   # a follow-up that was rewritten before searching
        note += f" · searched for: \"{response.standalone_question}\""
    return {"role": "assistant", "content": response.answer, "sources": sources, "note": note,
            "answerable": response.answerable}


def render_message(message: dict) -> None:
    st.markdown(message["content"])
    if message["role"] != "assistant":
        return
    sources = message.get("sources", [])
    if sources:
        with st.expander(f"Sources ({len(sources)})", expanded=True):
            for s in sources:
                st.markdown(f"**[{s['number']}] {s['title']}**  \n{s['pages']} · {s['section']}")
                extra = ""
                if s["also_on_pages"]:
                    pages = ", ".join(str(p) for p in s["also_on_pages"])
                    extra = f" · same content also on pages {pages}"
                st.caption(f"similarity {s['score']}{extra}")
                st.text(s["text"])
    elif not message.get("answerable", True):
        st.caption("No supporting passage found, so no sources are shown.")
    st.caption(message.get("note", ""))


def reset_conversation() -> None:
    st.session_state.messages = []


# ---- sidebar ---------------------------------------------------------------------------
with st.sidebar:
    st.header("FinBase Support")
    st.write("Ask about credit cards, personal loans, savings accounts, UPI and payments, "
             "fixed deposits and wealth products, and KYC and security.")
    st.button("New conversation", on_click=reset_conversation)
    st.subheader("Try an example")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, key=f"ex_{example}"):
            st.session_state.pending_question = example
    st.divider()
    st.caption(f"LLM: `{config.LLM_MODEL}`  \nEmbeddings: `{config.EMBEDDING_MODEL}`  \n"
               f"Top-k: {config.TOP_K} · min similarity: {config.MIN_SIMILARITY}")

# ---- index / pipeline loading ------------------------------------------------------------
st.title("FinBase Support Assistant")
st.caption("Answers come only from the FinBase policy documents, with the source of every answer shown.")

if not VectorStore.exists():
    st.warning("The search index has not been built yet.")
    if st.button("Build index now (takes about a minute the first time)"):
        from build_index import build_index
        with st.spinner("Reading PDFs, chunking and embedding..."):
            build_index()
        st.cache_resource.clear()
        st.rerun()
    st.stop()

if VectorStore.read_info().get("data_fingerprint") != data_fingerprint():
    st.info("The PDFs in data/ changed after the index was built. Run `python build_index.py` to update it.")

try:
    pipeline = load_pipeline()
except Exception as exc:   # missing API key, model/index mismatch, ...
    st.error(str(exc))
    st.stop()

# ---- chat ------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_message(message)

typed = st.chat_input("Ask a question about FinBase products...")
question = st.session_state.pop("pending_question", None) or typed

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the FinBase documents..."):
            try:
                response = pipeline.ask(question, history)
            except Exception as exc:
                st.error(f"Something went wrong while answering: {exc}")
                st.stop()
        assistant_message = response_to_message(response, question)
        render_message(assistant_message)
    st.session_state.messages.append(assistant_message)
