"""pgvector variation: vectors stored in PostgreSQL (server-based store).

Start the database first:
    docker compose up -d          # postgres:16 + pgvector extension image

Usage:
    python rag_pgvector.py build
    python rag_pgvector.py ask "your question"
    python rag_pgvector.py info
"""
from langchain_postgres import PGVector

from rag_core import config
from rag_core.pipeline import run_cli


def build_store(chunks, embeddings) -> None:
    PGVector.from_documents(
        chunks,
        embeddings,
        collection_name=config.COLLECTION_NAME,
        connection=config.PGVECTOR_URL,
        pre_delete_collection=True,  # wipe old collection -> idempotent rebuild
        use_jsonb=True,
    )


def load_store(embeddings) -> PGVector:
    return PGVector(
        embeddings=embeddings,
        collection_name=config.COLLECTION_NAME,
        connection=config.PGVECTOR_URL,
        use_jsonb=True,
    )


if __name__ == "__main__":
    run_cli("pgvector", build_store, load_store)
