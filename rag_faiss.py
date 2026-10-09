"""FAISS variation: local persistent index, no server needed.

Unlike the 06-RAG demo (in-memory only), this index is saved to disk at
build time and reloaded for every query.

Usage:
    python rag_faiss.py build
    python rag_faiss.py ask "your question"
    python rag_faiss.py info
"""
import shutil

from langchain_community.vectorstores import FAISS

from rag_core import config
from rag_core.pipeline import run_cli

STORE_DIR = config.STORES_DIR / "faiss"


def build_store(chunks, embeddings) -> None:
    if STORE_DIR.exists():
        shutil.rmtree(STORE_DIR)  # rebuild from scratch -> no stale chunks
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(STORE_DIR))  # writes index.faiss + index.pkl


def load_store(embeddings) -> FAISS:
    return FAISS.load_local(
        str(STORE_DIR), embeddings, allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    run_cli("FAISS", build_store, load_store)
