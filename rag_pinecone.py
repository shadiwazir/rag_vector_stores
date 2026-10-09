"""Pinecone variation: fully managed cloud vector store.

Requires PINECONE_API_KEY in .env (free tier works). The serverless index
is created automatically on first build.

Usage:
    python rag_pinecone.py build
    python rag_pinecone.py ask "your question"
    python rag_pinecone.py info
"""
import os
import time

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from rag_core import config
from rag_core.pipeline import run_cli


def _client() -> Pinecone:
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        raise RuntimeError("Set PINECONE_API_KEY in .env to use this variation.")
    return Pinecone(api_key=key)


def _ensure_index(pc: Pinecone, embeddings) -> None:
    """Create the serverless index if missing; dimension is probed live."""
    if pc.has_index(config.PINECONE_INDEX):
        return
    dim = len(embeddings.embed_query("dimension probe"))
    pc.create_index(
        name=config.PINECONE_INDEX,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(config.PINECONE_INDEX).status["ready"]:
        time.sleep(1)  # wait until the index can accept writes


def build_store(chunks, embeddings) -> None:
    pc = _client()
    _ensure_index(pc, embeddings)
    index = pc.Index(config.PINECONE_INDEX)
    # idempotent rebuild: clear old vectors (empty index has no namespace -> skip)
    if index.describe_index_stats().total_vector_count > 0:
        index.delete(delete_all=True)
    PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=config.PINECONE_INDEX
    )


def load_store(embeddings) -> PineconeVectorStore:
    pc = _client()
    _ensure_index(pc, embeddings)
    return PineconeVectorStore(
        index=pc.Index(config.PINECONE_INDEX), embedding=embeddings
    )


if __name__ == "__main__":
    run_cli("Pinecone", build_store, load_store)
