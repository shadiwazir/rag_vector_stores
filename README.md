# RAG Vector Stores — build-grade RAG with 4 interchangeable vector stores

Standalone RAG project: same pipeline, four different vector stores, each in
its own file. Indexes are **persisted to disk/server at build time** and
reloaded for every query — no rebuild-per-session coursework pattern.

| File | Store | Type | Needs |
|---|---|---|---|
| `rag_faiss.py` | FAISS | local index files | nothing |
| `rag_chroma.py` | Chroma | local embedded DB | nothing |
| `rag_pgvector.py` | pgvector | PostgreSQL extension | `docker compose up -d` |
| `rag_pinecone.py` | Pinecone | managed cloud | `PINECONE_API_KEY` |

## Project structure

```
rag_vector_stores/
├── knowledge_base/        # <-- copy your .txt / .pdf files here
├── stores/                # persisted indexes (created by `build`)
├── rag_core/              # shared code used by every variation
│   ├── config.py          # paths, models, tuning (overridable via .env)
│   ├── documents.py       # load .txt/.pdf -> chunk with metadata
│   ├── backends.py        # pluggable embeddings + LLM (openai/local/fake)
│   └── pipeline.py        # retrieve -> grounded prompt -> cited answer
├── rag_faiss.py           # variation 1: FAISS
├── rag_chroma.py          # variation 2: Chroma
├── rag_pgvector.py        # variation 3: pgvector
├── rag_pinecone.py        # variation 4: Pinecone
├── Dockerfile             # container image for the UI and RAG backends
├── docker-compose.yml     # UI and postgres+pgvector services
├── requirements.txt
└── .env.example
```

## Setup

```bash
cd my_work/rag_vector_stores
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # optional; defaults work out of the box
```

## Quickstart

1. Copy your documents (`.txt` / `.pdf`) into `knowledge_base/`
   (a sample `state_of_union.txt` is already included).
2. Build an index, then query it — repeat for any store:

```bash
python rag_faiss.py build
python rag_faiss.py ask "How is the US supporting Ukraine?"
python rag_faiss.py info
```

Same commands work for `rag_chroma.py`, `rag_pgvector.py`, `rag_pinecone.py`.

### pgvector (needs the database)

```bash
docker compose up -d        # starts postgres:16 + pgvector
python rag_pgvector.py build
python rag_pgvector.py ask "your question"
```

When using the CLI directly on the host with the copied `.env`, change the
hostname in `PGVECTOR_URL` from `pgvector` to `localhost`. The Dockerized app
uses the `pgvector` service hostname automatically.

### Pinecone (needs an API key)

Add `PINECONE_API_KEY=...` to `.env` (free tier works), then `build` / `ask`.
The serverless index is created automatically on first build.

## Backends (embeddings + LLM)

Selected via `.env`: `EMBEDDING_BACKEND` / `LLM_BACKEND`

| Value | What runs | Cost |
|---|---|---|
| `auto` (default) | OpenAI if `OPENAI_API_KEY` is set, else local | — |
| `local` | sentence-transformers + SmolLM2 on CPU | free |
| `openai` | OpenAI embeddings + chat model | paid API |
| `fake` | deterministic stub, no models at all | free (smoke tests only) |

## Notes

- `build` is idempotent: each store wipes its previous data before re-indexing.
- Answers cite sources as `[S1]`, `[S2]` with file name, page, and score.
- If you only use the OpenAI backend, you can skip the 4 heavy local-model
  lines in `requirements.txt` (torch, transformers, ...).

## Web UI

A small FastAPI app in `ui/` drives `build` / `ask` for every store from the browser
(Build tab with knowledge-base upload, Ask tab with conversations). It runs the
`rag_<store>.py` scripts as subprocesses in the same environment, so no second venv is needed.

Run the UI and pgvector database in Docker:

```bash
cp .env.example .env       # edit secrets and set a strong POSTGRES_PASSWORD
docker compose up --build
```

Open http://localhost:8000. Compose loads `.env` into the app container's
environment (`env_file`) and passes the PostgreSQL settings to the database; it
is not copied into the Docker image or stored in `docker-compose.yml`. After
editing `.env`, run `docker compose up -d` again to recreate the containers. The `knowledge_base/` and
`stores/` directories are mounted from the host, so uploaded documents and
built indexes persist across container restarts.

```bash
pip install -r requirements.txt
uvicorn ui.app:app --port 8000      # then open http://localhost:8000
```

Settings exposed in the UI: store, `EMBEDDING_BACKEND`, `LLM_BACKEND` (leave at
"(.env default)" to use `.env`), `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`.
Optional env vars: `RAG_BACKEND_DIR`, `RAG_BACKEND_PYTHON`, `RAG_TIMEOUT` (seconds, default 900).
