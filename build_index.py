"""Build (or rebuild) the search index from the PDFs in data/.

    python build_index.py

Run it once at the start and again whenever a PDF in data/ is added or changed.
"""
import logging

from src import config
from src.chunking import build_chunks
from src.document_loader import load_all_pdfs
from src.utils import data_fingerprint
from src.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_index")


def build_index(embedder=None, data_dir=config.DATA_DIR, index_dir=config.INDEX_DIR) -> VectorStore:
    log.info("1/4 Loading and cleaning PDFs from %s", data_dir)
    pages = load_all_pdfs(data_dir)

    log.info("2/4 Chunking %d pages", len(pages))
    chunks, stats = build_chunks(pages)
    for doc_id, s in stats.items():
        log.info("    %-9s %2d pages  %3d chunks before dedup -> %2d kept   %s",
                 doc_id, s["pages"], s["chunks_before_dedup"], s["chunks_kept"], s["title"])

    log.info("3/4 Embedding %d chunks with %s", len(chunks), config.EMBEDDING_MODEL)
    if embedder is None:
        from src.embeddings import Embedder
        embedder = Embedder()
    vectors = embedder.embed_documents([c.embed_text for c in chunks])

    log.info("4/4 Building FAISS index and saving to %s", index_dir)
    store = VectorStore.build(chunks, vectors, embedder.model_name, data_fingerprint(data_dir))
    store.save(index_dir)
    log.info("Done: %d chunks indexed (%d dimensions).", len(chunks), vectors.shape[1])
    return store


if __name__ == "__main__":
    build_index()
