"""Small helpers: text cleaning, duplicate detection keys, hashing, logging."""
import hashlib
import json
import logging
import re
import time
from pathlib import Path

from src import config

logger = logging.getLogger("finbase_rag")

# The PDFs use a missing glyph for the rupee sign; extractors turn it into U+25A0.
RUPEE_GLYPH = "\u25a0"
BULLET_CHARS = ("\x7f", "\uf0b7")


def clean_page_text(raw: str) -> str:
    """Turn raw extractor output for one page into clean, line-based text."""
    text = raw.replace(RUPEE_GLYPH, "₹")
    for ch in BULLET_CHARS:
        text = text.replace(ch, "•")
    text = text.replace("**", "").replace("`", "")      # markdown leftovers inside the PDFs

    cleaned = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # drop table-of-contents entries such as "• [Section 4: ...](#section-4)"
        if "](#section-" in line or line.lower() == "table of contents":
            continue
        line = re.sub(r"^•\s*", "• ", line)
        # table cells are separated by wide gaps in layout mode -> keep them visible
        line = re.sub(r"\s{3,}", " | ", line)
        line = re.sub(r"[ \t]+", " ", line)
        cleaned.append(line)
    return "\n".join(cleaned)


def collapse(text: str) -> str:
    return " ".join(text.split())


_FAQ_PARSE = re.compile(r"^Q\d{3}:\s*(.*?)\s*\([^()]*?\d+\s*\)\s*(.*?)\[Reference", re.S)


def dedup_key(text: str, block_type: str) -> str:
    """Key under which two chunks count as 'the same content'.

    FAQ items repeat the same question/answer with only counters and internal
    codes changed, so for them we compare question + answer text. Other repeated
    blocks (the numbered boilerplate sections) differ only by their numbers, so
    we mask digits before comparing.
    """
    flat = collapse(text)
    if block_type == "faq":
        m = _FAQ_PARSE.match(flat)
        if m:
            return f"faq::{m.group(1).lower()}::{m.group(2).lower()}"
    return "blk::" + re.sub(r"\d+", "#", flat.lower())


def data_fingerprint(data_dir: Path = config.DATA_DIR) -> str:
    """Hash of all PDFs, used to detect that the saved index is out of date."""
    h = hashlib.sha256()
    for pdf in sorted(Path(data_dir).glob("*.pdf")):
        h.update(pdf.name.encode())
        h.update(hashlib.sha256(pdf.read_bytes()).digest())
    return h.hexdigest()


def format_pages(start: int, end: int) -> str:
    return f"Page {start}" if start == end else f"Pages {start}-{end}"


def log_query(record: dict) -> None:
    """Append one JSON line per question to logs/queries.jsonl (best effort)."""
    try:
        config.LOG_DIR.mkdir(exist_ok=True)
        record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
        with open(config.LOG_DIR / "queries.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:  # logging must never break the app
        logger.warning("Could not write query log: %s", exc)


def norm_evidence(text: str) -> str:
    """Normalisation used when checking that a chunk contains a piece of evidence."""
    return collapse(text).lower()


def norm_answer(text: str) -> str:
    """Normalisation used for keyword checks on answers: '₹1,00,000' -> '100000'."""
    return collapse(text.lower().replace("₹", "").replace(",", ""))
