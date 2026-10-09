"""Step 5: store the chunk vectors in a FAISS index and search them.

We use IndexFlatIP: an exact (brute-force) inner-product search. With a few
hundred normalised vectors this is instant, and exact search means no recall
lost to approximation. Approximate indexes (IVF, HNSW) only pay off at
hundreds of thousands of vectors.

On disk the index consists of three files in INDEX_DIR:
  faiss.index   the vectors
  chunks.json   the text + metadata of each chunk, in the same order as the vectors
  info.json     which embedding model / PDFs the index was built from
"""
import json
import time
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from src import config
from src.chunking import Chunk


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], info: dict):
        self.index = index
        self.chunks = chunks
        self.info = info

    @classmethod
    def build(cls, chunks: list[Chunk], vectors: np.ndarray, model_name: str, fingerprint: str = "") -> "VectorStore":
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        info = {
            "embedding_model": model_name,
            "num_chunks": len(chunks),
            "dimension": int(vectors.shape[1]),
            "data_fingerprint": fingerprint,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return cls(index, chunks, info)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        k = min(k, len(self.chunks))
        scores, ids = self.index.search(query_vector, k)
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

    # ---- persistence -------------------------------------------------------
    def save(self, directory: Path = config.INDEX_DIR) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "faiss.index"))
        (directory / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=1), encoding="utf-8")
        (directory / "info.json").write_text(json.dumps(self.info, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path = config.INDEX_DIR) -> "VectorStore":
        directory = Path(directory)
        if not (directory / "faiss.index").exists():
            raise FileNotFoundError("No index found. Run `python build_index.py` first.")
        index = faiss.read_index(str(directory / "faiss.index"))
        chunks = [Chunk(**d) for d in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
        info = json.loads((directory / "info.json").read_text(encoding="utf-8"))
        return cls(index, chunks, info)

    @staticmethod
    def exists(directory: Path = config.INDEX_DIR) -> bool:
        return (Path(directory) / "faiss.index").exists()

    @staticmethod
    def read_info(directory: Path = config.INDEX_DIR) -> dict:
        """Index metadata only (cheap: does not load the vectors)."""
        return json.loads((Path(directory) / "info.json").read_text(encoding="utf-8"))
