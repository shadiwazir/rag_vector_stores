"""Shared RAG flow: retrieve -> build grounded prompt -> answer with citations."""
import argparse

from langchain_core.vectorstores import VectorStore

from . import config
from .backends import get_embeddings, get_llm
from .documents import load_documents, split_documents

PROMPT = """You are a document assistant. Answer ONLY from the context below.
If the answer is not in the context, say "Not found in the knowledge base."
Cite supporting chunks with their labels, e.g. [S1], [S2].

Context:
{context}

Question: {question}
Answer:"""


def format_context(results) -> str:
    """Turn (doc, score) pairs into [S1]-labelled blocks with source metadata."""
    blocks = []
    for rank, (doc, score) in enumerate(results, start=1):
        blocks.append(
            f"[S{rank}] {doc.metadata.get('source_file', '?')} "
            f"p.{doc.metadata.get('page_number', '?')} "
            f"chunk {doc.metadata.get('chunk_id', '?')} (score {float(score):.4f})\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def answer(store: VectorStore, question: str, top_k: int = config.TOP_K) -> str:
    """Full RAG cycle for one question: retrieve, prompt, generate, cite."""
    results = store.similarity_search_with_score(question, k=top_k)
    context = format_context(results)
    response = get_llm().invoke(PROMPT.format(context=context, question=question))
    text = response.content if hasattr(response, "content") else str(response)

    print(f"\nQ: {question}\n\nA: {text.strip()}\n\nSources:")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"  [S{rank}] {doc.metadata.get('source_file', '?')} "
              f"p.{doc.metadata.get('page_number', '?')} score={float(score):.4f}")
    return text


def run_cli(description: str, build_store, load_store) -> None:
    """argparse front-end shared by every store variation.

    build_store(docs, embeddings) -> persists a new index from knowledge_base/
    load_store(embeddings)        -> opens the persisted index
    """
    ap = argparse.ArgumentParser(description=description)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="index knowledge_base/ and persist the store")
    p_ask = sub.add_parser("ask", help="query the persisted store")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", "--top-k", type=int, default=config.TOP_K)
    sub.add_parser("info", help="print store stats")
    args = ap.parse_args()

    embeddings = get_embeddings()
    if args.cmd == "build":
        chunks = split_documents(load_documents())
        build_store(chunks, embeddings)
        print(f"[{description}] persisted {len(chunks)} chunks "
              f"(backend: {config.resolve_backend('embed')})")
        return

    try:
        store = load_store(embeddings)
    except Exception as exc:
        raise SystemExit(f"No usable store found - run `build` first. ({exc})")

    if args.cmd == "info":
        print(store)
    else:
        answer(store, args.question, args.top_k)
