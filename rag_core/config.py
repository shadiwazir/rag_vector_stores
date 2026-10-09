"""Central config: paths, chunking, backend selection. Override via .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # read .env if present

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = Path(os.getenv("KB_DIR", BASE_DIR / "knowledge_base"))   # drop .txt/.pdf here
STORES_DIR = Path(os.getenv("STORES_DIR", BASE_DIR / "stores"))  # persisted indexes

# --- Chunking / retrieval ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))

# --- Backend selection: auto | local | openai | fake (fake = offline test) ---
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto")
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")

# --- Model names ---
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# decoder-only instruct model: works with the text-generation pipeline on CPU
LOCAL_CHAT_MODEL = os.getenv("LOCAL_CHAT_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")

# --- Store-specific settings ---
PGVECTOR_URL = os.getenv("PGVECTOR_URL", "postgresql+psycopg://rag:rag@localhost:5432/ragdb")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-kb")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_kb")


def resolve_backend(kind: str) -> str:
    """'auto' -> openai when a key exists, otherwise local."""
    requested = EMBEDDING_BACKEND if kind == "embed" else LLM_BACKEND
    if requested != "auto":
        return requested
    return "openai" if os.getenv("OPENAI_API_KEY") else "local"
