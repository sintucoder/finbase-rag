"""Step 4: turn text into vectors with a sentence-transformers model.

The vectors are L2-normalised, so the inner product of two vectors equals their
cosine similarity. That is what the FAISS index in vector_store.py relies on.
"""
import numpy as np

from src import config


class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL, query_prefix: str = config.QUERY_PREFIX):
        # imported here so that modules which only need the interface (tests, eval
        # helpers) do not pay the cost of loading PyTorch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=len(texts) > 100,
        )
        return vectors.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Returns an array of shape (1, dim)."""
        vector = self.model.encode(
            [self.query_prefix + query], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return vector.astype("float32")
