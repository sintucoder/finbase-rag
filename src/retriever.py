"""Step 6: given a question, find the most similar chunks."""
from dataclasses import dataclass

from src import config
from src.chunking import Chunk
from src.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float   # cosine similarity, higher = more similar
    rank: int      # 1 = best match


class Retriever:
    def __init__(self, store: VectorStore, embedder, top_k: int = config.TOP_K,
                 min_score: float = config.MIN_SIMILARITY):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score

    def search(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Raw top-k results, best first, with no threshold applied."""
        query_vector = self.embedder.embed_query(query)
        hits = self.store.search(query_vector, k or self.top_k)
        return [RetrievedChunk(chunk, score, rank) for rank, (chunk, score) in enumerate(hits, start=1)]

    def retrieve(self, query: str) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        """Returns (all_hits, relevant_hits). Relevant = score >= min_score."""
        hits = self.search(query)
        return hits, [h for h in hits if h.score >= self.min_score]
