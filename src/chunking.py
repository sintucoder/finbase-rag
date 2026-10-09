"""Step 3: split pages into chunks that follow the structure of the documents.

Strategy
1. Walk through the document line by line and start a new block at every
   "Section ...:" heading, every FAQ item ("Q007:") and every numbered
   sub-heading ("1.2 FinBase Luxe Credit Card"). A fee, a rule or an FAQ answer
   therefore stays in one piece.
2. If a block is longer than CHUNK_SIZE characters, split it into windows with
   CHUNK_OVERLAP characters of overlap.
3. Merge chunks whose content is repeated (the PDFs repeat the same FAQ items and
   boilerplate sections many times). The first copy is kept and the other pages
   are remembered in `also_on_pages`, so citations stay honest.
"""
import re
import textwrap
from dataclasses import dataclass, field

from src import config
from src.document_loader import PageText
from src.utils import dedup_key, format_pages

SECTION_RE = re.compile(r"^Section\s+[\w.\-]+:\s+\S")
FAQ_RE = re.compile(r"^Q\d{3}:\s")
SUBSECTION_RE = re.compile(r"^\d+\.\d+\s+[A-Z][A-Za-z]")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    file_name: str
    doc_title: str
    section: str          # heading the chunk sits under, e.g. "Section 6: ... > 6.2 ..."
    block_type: str       # "front_matter" | "section" | "faq"
    page_start: int
    page_end: int
    text: str
    also_on_pages: list[int] = field(default_factory=list)   # pages holding identical content

    @property
    def embed_text(self) -> str:
        """What gets embedded: document title + section give each chunk its context."""
        body = self.text
        first_line, _, rest = body.partition("\n")
        if first_line == self.section:             # the heading is already in the header
            body = rest or body
        return f"{self.doc_title} | {self.section}\n{body}"

    @property
    def pages_label(self) -> str:
        return format_pages(self.page_start, self.page_end)

    @property
    def citation(self) -> str:
        return f"{self.doc_title}, {self.pages_label}, {self.section}"


def _split_into_blocks(lines: list[tuple[int, str]]) -> list[tuple[str, str, list[tuple[int, str]]]]:
    """Group (page, line) tuples into (block_type, label, lines) blocks."""
    blocks = []
    section = "Document information"
    block_type, label, buffer = "front_matter", "Document information", []

    def flush():
        # a block holding only a heading carries no information (e.g. "Section 23: FAQ Directory")
        only_heading = len(buffer) == 1 and SECTION_RE.match(buffer[0][1])
        if buffer and not only_heading:
            blocks.append((block_type, label, list(buffer)))
        buffer.clear()

    for page, line in lines:
        if SECTION_RE.match(line):
            flush()
            section, block_type, label = line, "section", line
        elif FAQ_RE.match(line):
            flush()
            block_type, label = "faq", f"{section} > {line[:4]}"
        elif block_type != "faq" and SUBSECTION_RE.match(line):
            flush()
            block_type, label = "section", f"{section} > {line}"
        buffer.append((page, line))
    flush()
    return blocks


MIN_TAIL = 150   # a final window with less new text than this is merged into the previous one


def _split_long_block(lines: list[tuple[int, str]], chunk_size: int, overlap: int):
    """Cut a long block into windows of about chunk_size characters, with overlap.

    Two details matter for these documents:
    * a table that gets cut keeps its header row at the top of every later window,
      otherwise a row like "₹750 + GST" no longer says which fee it belongs to;
    * a tiny leftover at the end is glued to the previous window instead of
      becoming a meaningless chunk of its own.
    """
    expanded = []
    for page, line in lines:                       # very long single lines are wrapped first
        for piece in textwrap.wrap(line, chunk_size) or [line]:
            expanded.append((page, piece))
    if sum(len(line) + 1 for _, line in expanded) <= chunk_size:
        return [expanded]

    table_header = next((item for i, item in enumerate(expanded)
                         if " | " in item[1] and (i == 0 or " | " not in expanded[i - 1][1])), None)

    windows: list[list[tuple[int, str]]] = []      # each window: list of (page, line)
    new_start: list[int] = []                      # index in `expanded` where each window's new text begins
    current, size = [], 0
    for index, item in enumerate(expanded):
        item_len = len(item[1]) + 1
        if current and size + item_len > chunk_size:
            windows.append(current)
            carry, carried = [], 0                 # repeat the last few lines at the start of the next window
            for prev in reversed(current):
                if carried + len(prev[1]) + 1 > overlap:
                    break
                carry.insert(0, prev)
                carried += len(prev[1]) + 1
            current, size = carry, carried
            new_start.append(index)
        current.append(item)
        size += item_len
    windows.append(current)
    new_start.append(len(expanded))

    # glue a too-small final window onto the one before it
    if len(windows) > 1:
        tail_new = expanded[new_start[-2]:]
        if sum(len(line) + 1 for _, line in tail_new) < MIN_TAIL:
            windows[-2].extend(tail_new)
            windows.pop()

    # repeat the table header row in windows that start in the middle of a table
    for number, window in enumerate(windows):
        if number > 0 and table_header and table_header not in window and any(" | " in l for _, l in window):
            windows[number] = [(window[0][0], table_header[1])] + window
    return windows


def deduplicate(chunks: list[Chunk]) -> list[Chunk]:
    """Keep the first copy of repeated content and remember where the copies were."""
    kept, first_seen = [], {}
    for chunk in chunks:
        key = dedup_key(chunk.text, chunk.block_type)
        original = first_seen.get(key)
        if original is None:
            first_seen[key] = chunk
            kept.append(chunk)
            continue
        own_pages = set(range(original.page_start, original.page_end + 1))
        new_pages = set(range(chunk.page_start, chunk.page_end + 1)) - own_pages
        original.also_on_pages = sorted(set(original.also_on_pages) | new_pages)
    return kept


def chunk_document(pages: list[PageText], chunk_size: int = config.CHUNK_SIZE,
                   overlap: int = config.CHUNK_OVERLAP) -> tuple[list[Chunk], int]:
    """Chunk one document. Returns (chunks, number_of_chunks_before_dedup)."""
    first = pages[0]
    lines = [(p.page_number, line) for p in pages for line in p.text.splitlines()]

    chunks = []
    for block_type, label, block_lines in _split_into_blocks(lines):
        for window in _split_long_block(block_lines, chunk_size, overlap):
            chunks.append(Chunk(
                chunk_id="", doc_id=first.doc_id, file_name=first.file_name,
                doc_title=first.doc_title, section=label, block_type=block_type,
                page_start=window[0][0], page_end=window[-1][0],
                text="\n".join(line for _, line in window),
            ))
    before = len(chunks)
    chunks = deduplicate(chunks)
    for number, chunk in enumerate(chunks, start=1):
        chunk.chunk_id = f"{first.doc_id}-{number:03d}"
    return chunks, before


def build_chunks(pages: list[PageText], chunk_size: int = config.CHUNK_SIZE,
                 overlap: int = config.CHUNK_OVERLAP) -> tuple[list[Chunk], dict]:
    """Chunk every document. Returns (all_chunks, per-document statistics)."""
    by_doc: dict[str, list[PageText]] = {}
    for page in pages:
        by_doc.setdefault(page.doc_id, []).append(page)

    all_chunks, stats = [], {}
    for doc_id, doc_pages in by_doc.items():
        chunks, before = chunk_document(doc_pages, chunk_size, overlap)
        stats[doc_id] = {"title": doc_pages[0].doc_title, "pages": len(doc_pages),
                         "chunks_before_dedup": before, "chunks_kept": len(chunks)}
        all_chunks.extend(chunks)
    return all_chunks, stats
