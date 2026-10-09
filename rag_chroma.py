"""Chroma variation: embedded persistent vector DB (SQLite-backed), no server.

Usage:
    python rag_chroma.py build
    python rag_chroma.py ask "your question"
    python rag_chroma.py info
"""
import shutil

from langchain_chroma import Chroma

from rag_core import config
from rag_core.pipeline import run_cli

STORE_DIR = config.STORES_DIR / "chroma"


def build_store(chunks, embeddings) -> None:
    if STORE_DIR.exists():
        shutil.rmtree(STORE_DIR)  # rebuild from scratch -> no stale chunks
    Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(STORE_DIR),  # auto-persisted to disk
    )


def load_store(embeddings) -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(STORE_DIR),
        embedding_function=embeddings,
    )


if __name__ == "__main__":
    run_cli("Chroma", build_store, load_store)
