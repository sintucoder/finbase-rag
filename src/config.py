"""Project settings in one place.

Every value can be overridden from a .env file or a real environment variable,
so nothing below needs editing for a normal run.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ---- paths ---------------------------------------------------------------
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
LOG_DIR = ROOT / "logs"
EVAL_FILE = ROOT / "eval" / "eval_questions.json"
EVAL_OUT_DIR = ROOT / "eval"


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


# ---- chunking ------------------------------------------------------------
# Sizes are in characters. all-MiniLM-L6-v2 reads at most 256 tokens (~1000
# characters of English), so chunks plus the small context header stay below that.
CHUNK_SIZE = _int("CHUNK_SIZE", 750)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 100)

# ---- embeddings ----------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# BGE models want a prefix on the *query* side only. MiniLM needs none.
QUERY_PREFIX = os.getenv(
    "QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: "
    if "bge" in EMBEDDING_MODEL.lower()
    else "",
)

# ---- retrieval -----------------------------------------------------------
TOP_K = _int("TOP_K", 5)
# Cosine similarity below this means "nothing relevant found". 0.30 is a
# starting point: calibrate it with `python evaluate.py --retrieval-only`.
MIN_SIMILARITY = _float("MIN_SIMILARITY", 0.30)

# ---- LLM (any OpenAI-compatible API) --------------------------------------
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None   # None -> official OpenAI endpoint
LLM_MODEL = os.getenv("LLM_MODEL", "")             # no default on purpose: model ids change, set it in .env
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 1024)      # generous: some models spend tokens on hidden reasoning

# ---- conversation --------------------------------------------------------
HISTORY_TURNS = _int("HISTORY_TURNS", 3)           # past question/answer pairs sent to the LLM
