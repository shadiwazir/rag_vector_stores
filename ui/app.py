"""FastAPI UI/API driving the rag_vector_stores CLI scripts (backend is untouched)."""
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

UI_DIR = Path(__file__).resolve().parent
BACKEND_DIR = Path(os.getenv("RAG_BACKEND_DIR", UI_DIR.parent)).resolve()
KB_DIR = BACKEND_DIR / "knowledge_base"
# Same environment as the UI: the backend scripts run with the interpreter serving this app
BACKEND_PYTHON = os.getenv("RAG_BACKEND_PYTHON", sys.executable)
TIMEOUT = int(os.getenv("RAG_TIMEOUT", "900"))
ALLOWED_EXT = {".pdf", ".txt"}

Store = Literal["faiss", "chroma", "pgvector", "pinecone"]
Backend = Literal["auto", "local", "openai", "fake"]

app = FastAPI(title="RAG Vector Store UI")


class RunConfig(BaseModel):
    store: Store
    # None -> not passed to the backend, so its .env value applies
    embedding_backend: Optional[Backend] = None
    llm_backend: Optional[Backend] = None
    chunk_size: int = Field(1000, ge=1)
    chunk_overlap: int = Field(200, ge=0)
    top_k: int = Field(4, ge=1)

    @model_validator(mode="after")
    def _check_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class AskRequest(RunConfig):
    question: str = Field(min_length=1)


def _run(cfg: RunConfig, args: list[str]) -> dict:
    env = os.environ.copy()
    env.update(CHUNK_SIZE=str(cfg.chunk_size), CHUNK_OVERLAP=str(cfg.chunk_overlap),
               TOP_K=str(cfg.top_k), PYTHONUNBUFFERED="1")
    if cfg.embedding_backend:
        env["EMBEDDING_BACKEND"] = cfg.embedding_backend
    if cfg.llm_backend:
        env["LLM_BACKEND"] = cfg.llm_backend
    try:
        p = subprocess.run([BACKEND_PYTHON, f"rag_{cfg.store}.py", *args], cwd=BACKEND_DIR,
                           env=env, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, f"Timed out after {TIMEOUT}s")
    return {"ok": p.returncode == 0, "returncode": p.returncode,
            "stdout": p.stdout, "stderr": p.stderr}


@app.post("/api/build")
def build(cfg: RunConfig):
    return _run(cfg, ["build"])


@app.post("/api/ask")
def ask(req: AskRequest):
    res = _run(req, ["ask", req.question, "-k", str(req.top_k)])
    if res["ok"]:
        # The backend echoes the question first; with a transcript it contains "A:" lines,
        # so strip that echo before splitting answer from sources.
        out = res["stdout"]
        echo = f"\nQ: {req.question}\n\nA: "
        body = out[len(echo):] if out.startswith(echo) else out
        answer, sep, sources = body.rpartition("\n\nSources:\n")
        res["answer"] = (answer if sep else body).strip()
        res["sources"] = sources.strip() if sep else ""
    return res


@app.get("/api/files")
def list_files():
    KB_DIR.mkdir(exist_ok=True)
    return [{"name": f.name, "size": f.stat().st_size}
            for f in sorted(KB_DIR.iterdir())
            if f.is_file() and f.suffix.lower() in ALLOWED_EXT]


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    KB_DIR.mkdir(exist_ok=True)
    names = [Path(f.filename or "").name for f in files]
    bad = [n for n in names if Path(n).suffix.lower() not in ALLOWED_EXT]
    if bad:  # validate everything first so nothing is saved on a bad batch
        raise HTTPException(400, f"Only .pdf and .txt files are allowed: {', '.join(bad)}")
    for name, f in zip(names, files):
        (KB_DIR / name).write_bytes(await f.read())
    return {"saved": names}


@app.delete("/api/files/{name}")
def delete_file(name: str):
    target = KB_DIR / Path(name).name
    if not target.is_file():
        raise HTTPException(404, "Not found")
    target.unlink()
    return {"deleted": target.name}


@app.get("/")
def index():
    return FileResponse(UI_DIR / "static" / "index.html")
