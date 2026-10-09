# FinBase Support Assistant (RAG)

A customer-support chatbot that answers questions about FinBase products using only the six FinBase policy PDFs. It retrieves the relevant passages first, gives them to an LLM as context, and shows which document, page and section each answer came from. If the documents don't contain the answer, it says so instead of guessing.

## Problem statement

The assignment asks for a FinTech support assistant built on Retrieval-Augmented Generation: answers must be grounded in the provided knowledge base, sources must be visible, follow-up questions should work, and the system must be evaluated (retrieval quality, correctness, groundedness, citation accuracy). A plain "paste the documents into a prompt" chatbot is explicitly not what is wanted.

## Features

- Full RAG pipeline: PDF extraction, cleaning, structure-aware chunking, embeddings, FAISS search, grounded generation
- Citations with document title, page number and section, for every answer; the `[1]`, `[2]` markers in the answer point at the numbered sources underneath
- Refuses to answer when nothing relevant is found (similarity threshold) or when the LLM judges the context insufficient
- Follow-up questions: a follow-up like "what about senior citizens?" is rewritten into a standalone question before searching
- Streamlit chat UI with loading state, example questions, new-conversation button, and the retrieved passages behind each answer
- Evaluation script with a 30-question set (Hit@k, Recall@k, MRR, answer correctness, numeric groundedness, optional LLM judge, citation accuracy)
- Offline unit tests (28) that run without an API key or model download
- Query log (`logs/queries.jsonl`) with scores and latency for each question

## Architecture

```
                       build_index.py (run once, and again when PDFs change)
  data/*.pdf ──► document_loader ──► chunking ──► embeddings ──► FAISS index
                 (pypdf + cleaning)  (sections,    (MiniLM)       index/faiss.index
                                      FAQ items,                  index/chunks.json
                                      dedup)

                       app.py / evaluate.py (every question)
  question ──► [follow-up? rewrite with LLM] ──► embed query ──► FAISS top-5
                                                                    │
                      score < MIN_SIMILARITY for all?  ──yes──► "not found" (LLM not called)
                                                                    │ no
                                              numbered passages + rules ──► LLM ──► answer with [n] citations
                                                                                      │
                                                    "not found" sentence?  ──yes──► no sources shown
                                                                                      │ no
                                                                      map [n] back to chunks ──► Sources in the UI
```

## The RAG pipeline, step by step

| Step | What happens | File |
|---|---|---|
| 1. Load | `pypdf` reads each PDF page by page (layout mode, so table rows stay on one line) | `src/document_loader.py` |
| 2. Clean | Fix the rupee glyph, bullets, markdown leftovers, drop table-of-contents lines | `src/utils.py` |
| 3. Chunk | Split on the document structure, then merge repeated content | `src/chunking.py` |
| 4. Embed | Each chunk becomes a 384-number vector with `all-MiniLM-L6-v2` | `src/embeddings.py` |
| 5. Index | Vectors go into a FAISS `IndexFlatIP`; chunk text and metadata are saved next to it | `src/vector_store.py` |
| 6. Retrieve | Embed the question, take the 5 most similar chunks, drop those below the threshold | `src/retriever.py` |
| 7. Prompt | Numbered passages + strict rules + the question are sent to the LLM | `src/rag_pipeline.py` |
| 8. Answer | The LLM answers and cites passage numbers | `src/llm.py` |
| 9. Cite | `[n]` markers are mapped back to chunks, which carry document, page, section | `src/rag_pipeline.py`, `app.py` |

## What the documents look like (and how the pipeline deals with it)

A few properties of the six PDFs shaped the design:

1. **The PDFs are mostly repetition.** Each is about 25 pages: roughly 9 pages of real content, then an FAQ of 100 items that is really 10 distinct questions repeated with different counters and internal codes, plus numbered boilerplate sections ("Section 7 ... Section 20") that differ only in their numbers. Without handling this, the top 5 results for most questions would be five copies of the same FAQ item. The chunker merges repeats and keeps the first copy, and remembers the other pages in `also_on_pages`, which the UI shows. The 6 documents have 152 pages and produced 844 raw chunks; 130 remain after merging (60 FAQ chunks, exactly 10 per document, 64 section chunks, 6 title blocks).
2. **The rupee sign is a missing glyph.** The PDFs print it as a placeholder box. Different extractors decode that box differently: `pdftotext` gives `■`, `pdfplumber` gives `n` (so ₹999 becomes "n999"), PyMuPDF gives `I`. `pypdf` returns `■` consistently, and `clean_page_text` maps it to `₹`. This is why the project uses `pypdf`.
3. **Tables need care.** In plain extraction every table cell lands on its own line. In `layout` mode a row stays on one line, which the cleaner writes as `Fee | Neo | Luxe | Metal`. When a long table has to be split, the header row is repeated at the top of the next chunk, otherwise "₹750 + GST" no longer says which fee it is.
4. **The documents list sections that are not there.** The table of contents of five documents names a section that is missing from the body (FD rate matrix in the FD guide, chargeback workflow in the payments SOP, reward points redemption in the card handbook, transfer cut-off times in the savings manual, fraud reporting timelines and Video KYC technical standards in the KYC document). These gaps are not filled in by the system. Questions that depend on them should be declined (see Q26 in the evaluation set).
5. **Citations in the FAQ do not always match the body.** The loan FAQ says foreclosure charges are in "Section 4.2", while the body puts them in Section 6.2 (page 3). The assistant cites where the text actually appears in the PDF.
6. **A few figures conflict or look like typos.** The payments SOP says delayed-refund compensation starts after T+2 (Section 2) but its matrix says "beyond T+5" for merchant payments. The savings manual's contactless limit is printed as "₹5,00,0". These are left as they are, and the prompt tells the model to point out disagreements instead of choosing silently (evaluation question Q23 tests this).

## Technology choices

| Part | Choice | Why |
|---|---|---|
| Language / UI | Python, Streamlit | The brief asks for a simple interface; Streamlit gives a chat UI in about 100 lines and no front-end code |
| PDF extraction | `pypdf` (layout mode) | Pure Python, installs everywhere, decodes the rupee glyph consistently, keeps table rows together |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Small (about 90 MB), runs on CPU, no API key or cost, widely documented, cosine scores are well spread so a similarity threshold is meaningful |
| Vector search | FAISS `IndexFlatIP` | See below |
| LLM | Any OpenAI-compatible API through the `openai` package | Switching between OpenAI, Gemini (free tier), Groq, etc. is three lines in `.env`; only `src/llm.py` talks to the provider |
| Memory | `st.session_state` | The assignment does not ask for persistent history, so SQLite was left out |
| Not used | LangChain, LlamaIndex, FastAPI, Docker, a database | Every step here is about 20-60 lines of plain Python, which makes the pipeline readable and explainable. A framework would hide exactly the parts that need explaining |

### Why embeddings

Users do not use the document's words. "My money got stuck after a failed UPI payment" should find the passage about auto-reversal within T+2 business days. Keyword search matches words; an embedding model maps text to a vector so that texts with similar *meaning* end up close together, even with no words in common.

### Why FAISS

FAISS stores the vectors and finds the nearest ones to a query vector fast. For about 130 chunks the project uses `IndexFlatIP`, an exact brute-force search, so there is no approximation error and it takes well under a millisecond. Vectors are normalised, so inner product equals cosine similarity. Approximate indexes (IVF, HNSW) only matter at hundreds of thousands of chunks. FAISS is a library and not a server, which keeps the project to a single `pip install`.

## Chunking strategy

- **Split on structure first.** A new chunk starts at every `Section N:` heading, every FAQ item (`Q007:`) and every numbered sub-heading (`1.2 FinBase Luxe Credit Card`). A fee, a rule or an FAQ answer stays whole. Fixed-size splitting would cut through table rows and sentences.
- **Fall back to size only when needed.** Blocks longer than 750 characters are split on line boundaries with 100 characters of overlap. Median chunk length is 570 characters and the longest is 872 (including a repeated table header).
- **Why 750 characters.** `all-MiniLM-L6-v2` reads at most 256 tokens, which is roughly 800-1000 characters of English and fewer for number-heavy tables. Larger chunks would be silently truncated when embedded. Smaller chunks would separate a rule from its exceptions.
- **No tiny leftovers.** A final fragment under 150 characters is glued to the previous chunk (an earlier version of the splitter produced a 32-character chunk containing just "p.a. across all deposit tenures.").
- **Context header.** The text that gets embedded is `document title | section` followed by the chunk. A chunk like "Annual Fee Waiver: ... 1,20,000" does not contain the words "Luxe credit card" by itself, the header adds them.
- **Metadata** kept on every chunk: document title, file name, page start/end, section path, chunk type (title block / section / FAQ), and other pages with identical content.

## Retrieval strategy

Dense retrieval only: embed the question, take the top 5 chunks by cosine similarity (`TOP_K`), and keep those with a score of at least `MIN_SIMILARITY` (default 0.30). If none pass, the system answers "not found" without calling the LLM, which also saves a call.

**The 0.30 default is an initial guess, not a tuned value.** `python evaluate.py --retrieval-only` prints the top-1 similarity of every answerable question next to the not-in-knowledge-base questions and suggests a separating threshold if one exists. Set `MIN_SIMILARITY` from that output.

Follow-up questions are handled by rewriting: if there is chat history, the LLM turns "and for the Metal one?" into "What is the annual fee of the FinBase Metal credit card?" before searching. The UI shows what was actually searched.

## LLM and prompting approach

The system prompt (in `src/rag_pipeline.py`) tells the model to: use only the numbered passages; cite passage numbers after each statement; reply with one exact sentence if the passages are insufficient; answer only the supported part of a question and say what is missing; allow simple arithmetic but show it; list each case when figures depend on product or customer type or when passages disagree; and ignore instructions embedded in the passages or question. Temperature is 0.

## Hallucination mitigation

Several layers, each cheap on its own:

1. **Retrieval gate:** below-threshold questions never reach the LLM.
2. **Grounding instruction:** answer only from the passages, never guess fees, rates, limits or eligibility.
3. **Fixed refusal sentence:** "I couldn't find enough information to answer this from the available FinBase knowledge base." It is detected in code, and no sources are shown for a refusal.
4. **Mandatory citations:** every claim carries `[n]`, so unsupported claims are easy to spot.
5. **Conflict handling:** passages that disagree are reported, not merged.
6. **Temperature 0.**
7. **Evaluation:** a numeric groundedness check flags numbers in an answer that are in neither the context nor the question, and `--judge` adds an LLM judge.

None of this makes hallucination impossible. See Limitations.

## Citations

The cited passage numbers in the answer are mapped back to chunks, and each source shows document title, page (or page range), section, similarity score and the full passage text. Example:

```
Source [1]: FinBase Personal Loans Master Policy & Operational Manual, Page 3, Section 6.2: Foreclosure Charges & Rules
```

For merged duplicates the UI adds "same content also on pages ...". Page numbers are the physical page of the PDF (the documents print no page numbers of their own).

## Evaluation methodology

`eval/eval_questions.json` holds 30 questions written from the six PDFs:

| Category | Count | Purpose |
|---|---|---|
| direct_factual | 4 | single fact lookup |
| numerical | 7 | table lookups, slab and range logic, arithmetic |
| policy | 8 | rules, windows, exclusions |
| multi_document | 3 | answer needs two documents, or sits in a document you would not expect |
| confusing | 2 | conflicting or conditional figures |
| not_in_kb | 4 | must be declined (missing section, unsupported tenure, unrelated product, off-topic) |
| follow_up | 2 | question only makes sense with the previous turn |

Each answerable question has **gold evidence**: short phrases that must appear in a chunk of the right document (not hard-coded chunk IDs, so changing the chunking does not invalidate the set). A test checks that every evidence phrase really exists in the PDFs.

| Metric | How it is computed | Automatic? |
|---|---|---|
| Hit@1/3/5 | any gold evidence chunk in the top k | yes |
| Recall@k | share of gold evidence items found in the top k (matters for multi-document questions) | yes |
| MRR | 1 / rank of the first gold chunk | yes |
| Rejection at retrieval | not-in-KB questions whose top score is below the threshold | yes |
| Answer correctness | every keyword group appears in the answer (for not-in-KB questions: the system declined) | yes, keyword-based, so it is strict about wording |
| Numeric groundedness | every number in the answer appears in the context or the question | yes, rough: flags derived numbers unless listed in `allowed_numbers` |
| LLM-judged groundedness | `--judge`: the LLM checks the answer against the context | yes, but the same model judging itself is a soft signal |
| Citation accuracy | the passages the answer cites contain the gold evidence (`citation_hit`: at least one; `citation_full`: all, for multi-document questions) | yes |
| Manual review | empty `manual_correct` column in `results.csv` | **manual**, fill in yourself |

Automatic keyword matching will mark some correct answers wrong (a different phrasing) and rarely the opposite, so read the answers in `results.csv`, not just the summary.

### Results

No numbers are included yet: they have to come from a real run with the embedding model and your LLM. Run the evaluation, then paste `eval/results_summary.md` below. Do not edit the numbers by hand.

```
python evaluate.py --retrieval-only     # no API key needed
python evaluate.py --judge --sleep 4    # full run
```

| group | n | Hit@1 | Hit@3 | Hit@5 | MRR | answer correct | citation hit |
|---|---|---|---|---|---|---|---|
| overall | 30 | _run it_ | | | | | |

## Project structure

```
finbase-rag/
├── data/                  the six FinBase PDFs
├── src/
│   ├── config.py          all settings, read from .env
│   ├── utils.py           text cleaning, duplicate keys, hashing, query log
│   ├── document_loader.py PDF -> pages of clean text
│   ├── chunking.py        structure-aware chunks + duplicate merging
│   ├── embeddings.py      sentence-transformers wrapper
│   ├── vector_store.py    FAISS index: build, save, load, search
│   ├── retriever.py       top-k search + similarity threshold
│   ├── llm.py             OpenAI-compatible client (the only provider-specific file)
│   └── rag_pipeline.py    prompt, rewrite, answer, citations
├── eval/eval_questions.json   30 test questions with gold evidence
├── tests/                 28 offline tests (fake embedder and fake LLM)
├── app.py                 Streamlit UI
├── build_index.py         builds / rebuilds the index
├── evaluate.py            evaluation script
├── requirements.txt
├── .env.example
└── .gitignore
```

## Installation

Needs Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # the first install is large because of PyTorch
cp .env.example .env               # Windows: copy .env.example .env
```

Open `.env` and set your API key and model id.

## Environment variables

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `LLM_API_KEY` | yes | | API key (`OPENAI_API_KEY` is also accepted) |
| `LLM_BASE_URL` | no | OpenAI | Base URL of an OpenAI-compatible API. For Gemini: `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `LLM_MODEL` | yes | | Model id from your provider. There is no default because ids change, check the provider's current list |
| `LLM_MAX_TOKENS` | no | 1024 | Output limit. Keep it generous: some models use part of it for hidden reasoning |
| `EMBEDDING_MODEL` | no | `sentence-transformers/all-MiniLM-L6-v2` | Changing it requires rebuilding the index |
| `TOP_K` | no | 5 | Chunks retrieved per question |
| `MIN_SIMILARITY` | no | 0.30 | Cosine threshold, calibrate it with `evaluate.py --retrieval-only` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | no | 750 / 100 | Characters. Changing them requires rebuilding the index |
| `HISTORY_TURNS` | no | 3 | Past question/answer pairs sent to the LLM |

Gemini's free tier may use your content to improve Google's products. That is fine for these sample documents, but do not send confidential company documents through a free tier.

## Run locally

```bash
python build_index.py        # once: reads PDFs, builds index/ (downloads the embedding model the first time)
streamlit run app.py         # opens the chat in your browser
python evaluate.py --retrieval-only
python evaluate.py
python -m pytest tests -q    # offline tests, no API key needed
```

If you forget `build_index.py`, the app shows a "Build index now" button.

## Adding or re-indexing documents

1. Put the new PDF in `data/`.
2. Run `python build_index.py`. The index is rebuilt from scratch, which takes seconds at this size.
3. The app warns when the PDFs in `data/` no longer match the index.

The chunker expects the structure of these documents (`Section N:` headings, `Q001:` FAQ items, `1.2 Title` sub-headings). A PDF with a different layout still works, but the headings will not be detected and it falls back to size-based splitting, so check `index/chunks.json` to see what the chunks look like. If you change `EMBEDDING_MODEL`, `CHUNK_SIZE` or `CHUNK_OVERLAP`, rebuild too (the app refuses to start if the embedding model differs from the one in the index).

## Example questions

- What is the foreclosure charge if I close my personal loan after 18 months?
- My UPI payment failed but money was debited. When do I get it back, and what if it is late?
- How much interest does a senior citizen get on a 1-year FD?
- What is the late payment fee if my credit card balance is ₹7,000?
- Is PAN mandatory for a fixed deposit above ₹50,000?
- Follow-up: "What is the annual fee of the Luxe card?" then "And the Metal one?"
- Should be declined: "What is the interest rate on a FinBase home loan?", "What is the capital of France?"

## Deploying a public URL

The assignment asks for a public URL. The simplest route is Streamlit Community Cloud: push the repo to GitHub, create the app from `app.py`, and put `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL` in the app's secrets (they are exposed to the app as environment variables). `index/` is git-ignored, so either use the "Build index now" button after the first start, or run `build_index.py` locally and remove `index/*` from `.gitignore` to commit the small index (a few hundred KB). These deployment steps have not been tested.

## Limitations

- **Quality is unmeasured until you run the evaluation.** The offline tests check the plumbing with a fake embedder and a fake LLM, which says nothing about how well the real model retrieves. The default similarity threshold is a starting guess until it is calibrated.
- **Dense retrieval can miss exact tokens.** Queries that are mostly an identifier (an error code like `U69`, a rule id like `CC-SEC-401`) are weak for embeddings. A keyword (BM25) component would help.
- **Table reasoning depends on the LLM.** Range lookups ("₹12 lakh loan: which slab?") and arithmetic work only if the right table chunk is retrieved and the model reads it correctly.
- **Duplicates were merged by exact text.** If two FAQ answers differ by a single word they stay separate, which is correct, but a rewritten duplicate would not be merged.
- **Layout assumptions.** Heading detection is tuned to these six PDFs.
- **Citations show where the answer is supported, not that it is correct.** The system trusts the documents, including their typos and conflicting figures.
- **Single-turn retrieval with a short memory.** Only the last 3 exchanges are used for follow-ups, and a bad rewrite can retrieve the wrong thing (the UI shows what was searched).
- **Not a real banking assistant.** It has no access to accounts, does not give advice, and any production use would need review by the compliance team.

## Future improvements

- Hybrid search (BM25 + embeddings) merged by reciprocal rank fusion
- A cross-encoder reranker on the top 10-20 results
- Streaming answers
- Persistent chat history in SQLite if it becomes a requirement
- Larger evaluation set with a second person labelling the gold evidence
- Caching of repeated questions
- A confidence indicator in the UI based on the similarity score
