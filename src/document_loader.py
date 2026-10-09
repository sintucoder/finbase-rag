"""Steps 1-2 of the pipeline: read the PDFs and extract clean text page by page."""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from src import config
from src.utils import clean_page_text


@dataclass
class PageText:
    doc_id: str        # file name without extension, e.g. "sample_3"
    file_name: str     # e.g. "sample_3.pdf"
    doc_title: str     # e.g. "FinBase Credit Cards Comprehensive Handbook & Cardholder Agreement"
    page_number: int   # 1-based physical page number in the PDF
    text: str


def extract_title(first_page_text: str, fallback: str) -> str:
    """The title is every line before 'Document Code:' on page 1 (it can wrap)."""
    title_lines = []
    for line in first_page_text.splitlines():
        if line.startswith("Document Code:"):
            break
        title_lines.append(line)
    title = " ".join(title_lines).strip()
    return title or fallback


def load_pdf(path: Path) -> list[PageText]:
    reader = PdfReader(str(path))
    # layout mode keeps each table row on one line; plain mode splits every cell onto its own line
    cleaned = [clean_page_text(p.extract_text(extraction_mode="layout") or "") for p in reader.pages]
    title = extract_title(cleaned[0], fallback=path.stem) if cleaned else path.stem
    return [
        PageText(path.stem, path.name, title, number, text)
        for number, text in enumerate(cleaned, start=1)
        if text.strip()
    ]


def load_all_pdfs(data_dir: Path = config.DATA_DIR) -> list[PageText]:
    pdfs = sorted(Path(data_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")
    pages: list[PageText] = []
    for pdf in pdfs:
        pages.extend(load_pdf(pdf))
    return pages
