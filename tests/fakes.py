"""Stand-ins used ONLY by the tests, so they run offline without downloading a model.

They are good enough to check that the plumbing works (chunks -> index -> search ->
prompt -> citations). They say nothing about the retrieval quality of the real
sentence-transformers model, which is measured by evaluate.py.
"""
import re
import zlib

import numpy as np

STOPWORDS = set("a an the is are of to in on for and or my i do does what how can if it be at by with from as this that".split())


class FakeEmbedder:
    """Hashed bag-of-words vectors, L2-normalised (cosine similarity = word overlap)."""
    model_name = "fake-hashing-embedder"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for word in re.findall(r"[a-z0-9₹%+.]+", text.lower()):
            if word not in STOPWORDS:
                v[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts]).astype("float32")

    def embed_query(self, query):
        return self._vec(query)[None, :].astype("float32")


class FakeLLM:
    """Returns scripted replies and remembers every call."""

    def __init__(self, answer="Placeholder answer [1].", rewrite="REWRITTEN QUESTION"):
        self.answer, self.rewrite, self.calls = answer, rewrite, []

    def chat(self, messages, temperature=0.0, max_tokens=0):
        self.calls.append(messages)
        if "Rewrite the user's last question" in messages[0]["content"]:
            return self.rewrite
        return self.answer
