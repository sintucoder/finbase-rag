import pytest

from src import config
from src.chunking import build_chunks, chunk_document
from src.document_loader import PageText, load_all_pdfs
from src.utils import clean_page_text, dedup_key


def page(text, number=1, doc="d1"):
    return PageText(doc, f"{doc}.pdf", "Test Doc Title", number, text)


def test_clean_page_text_fixes_pdf_artifacts():
    raw = ("Table of Contents\n"
           "\x7f   [Section 1: Clause 1](#section-1)\n"
           "\x7f   Joining fee: \u25a0999 + 18% GST.\n"
           "FinBase does **NOT** offer `crypto`.\n"
           "Fee Item          \u25a00 (Free)          \u25a0999 + GST\n")
    cleaned = clean_page_text(raw)
    assert "\u25a0" not in cleaned and "₹999 + 18% GST" in cleaned
    assert "Table of Contents" not in cleaned and "#section-1" not in cleaned
    assert "does NOT offer crypto" in cleaned
    assert "• Joining fee" in cleaned
    assert "Fee Item | ₹0 (Free) | ₹999 + GST" in cleaned


def test_blocks_follow_headings_faq_items_and_subsections():
    text = ("Test Doc Title\nDocument Code: X\n"
            "Section 1: Intro\nIntro text.\n"
            "1.1 Neo Card\nNeo details.\n"
            "1.2 Luxe Card\nLuxe details.\n"
            "Section 2: FAQ Directory (Items 1 - 2)\n"
            "Q001: Question one? (Case 1)\nAnswer one. [Reference X: Clause A-001].\n"
            "Q002: Question two? (Case 2)\nAnswer two. [Reference X: Clause A-002].")
    chunks, _ = chunk_document([page(text)])
    sections = [c.section for c in chunks]
    assert sections[0] == "Document information"
    assert any(s.endswith("1.1 Neo Card") for s in sections)
    assert any(s.endswith("1.2 Luxe Card") for s in sections)
    assert sum(c.block_type == "faq" for c in chunks) == 2
    assert not any(c.text.strip().startswith("Section 2: FAQ") and len(c.text.splitlines()) == 1 for c in chunks)


def test_repeated_faq_items_are_merged_and_pages_remembered():
    faq = "Q00{n}: Same question? (Wealth query {n})\nSame answer. [Reference W: Clause W-00{n}].\n• Code: CAT-0{n}"
    pages = [page("Section 9: FAQ\n" + faq.format(n=1), 1),
             page(faq.format(n=2), 2), page(faq.format(n=3), 3)]
    chunks, before = chunk_document(pages)
    faq_chunks = [c for c in chunks if c.block_type == "faq"]
    assert before - len(chunks) == 2 and len(faq_chunks) == 1
    assert faq_chunks[0].page_start == 1 and faq_chunks[0].also_on_pages == [2, 3]


def test_different_answers_are_not_merged():
    a = "Q001: Same question? (Case 1)\nAnswer A. [Reference X]."
    b = "Q002: Same question? (Case 2)\nAnswer B. [Reference X]."
    assert dedup_key(a, "faq") != dedup_key(b, "faq")


def test_long_table_split_repeats_header_row_and_has_no_orphans():
    rows = "\n".join(f"Fee number {i} | ₹{i}00 + GST | ₹{i}50 + GST | ₹{i}99 + GST" for i in range(1, 30))
    text = "Section 5: Fees\nFee Item | Neo | Luxe | Metal\n" + rows
    chunks, _ = chunk_document([page(text)], chunk_size=500, overlap=80)
    assert len(chunks) > 1
    for c in chunks[1:]:
        assert c.text.splitlines()[0] == "Fee Item | Neo | Luxe | Metal"
    assert min(len(c.text) for c in chunks) > 150


def test_page_range_is_tracked_across_a_page_break():
    chunks, _ = chunk_document([page("Section 1: A\nline on page one", 1), page("line on page two", 2)])
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)
    assert chunks[0].pages_label == "Pages 1-2"


# ---- checks against the real PDFs (skipped if data/ is empty) -------------------
@pytest.fixture(scope="module")
def real_chunks():
    if not list(config.DATA_DIR.glob("*.pdf")):
        pytest.skip("no PDFs in data/")
    chunks, stats = build_chunks(load_all_pdfs())
    return chunks, stats


def test_real_pdfs_all_documents_loaded_with_titles(real_chunks):
    _, stats = real_chunks
    assert len(stats) == 6
    assert all(s["title"].startswith("FinBase") for s in stats.values())


def test_real_pdfs_faq_dedup_leaves_ten_distinct_items_per_document(real_chunks):
    chunks, stats = real_chunks
    for doc_id in stats:
        assert sum(1 for c in chunks if c.doc_id == doc_id and c.block_type == "faq") == 10


def test_real_pdfs_text_is_clean(real_chunks):
    chunks, _ = real_chunks
    joined = "\n".join(c.text for c in chunks)
    assert "\u25a0" not in joined and "](#section-" not in joined and "**" not in joined
    assert "₹999 + GST" in joined


def test_real_pdfs_foreclosure_rule_is_on_page_3_of_the_loan_policy(real_chunks):
    chunks, _ = real_chunks
    hit = [c for c in chunks if c.doc_id == "sample_5" and "3% of the outstanding principal" in c.text
           and c.block_type == "section"]
    assert hit and hit[0].page_start == 3 and "6.2" in hit[0].section
