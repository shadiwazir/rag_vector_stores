"""Load .txt/.pdf files from the knowledge base and split them into chunks."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def load_documents(kb_dir: Path = config.KB_DIR) -> list[Document]:
    """Load every .txt and .pdf in kb_dir; tag metadata with source_file/page."""
    docs: list[Document] = []
    for path in sorted(kb_dir.glob("**/*")):
        if path.suffix.lower() == ".txt":
            loaded = TextLoader(str(path), autodetect_encoding=True).load()
        elif path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            continue  # ignore anything else
        for d in loaded:
            d.metadata["source_file"] = path.name
            if "page" in d.metadata:  # pdf pages are 0-based -> make 1-based
                d.metadata["page_number"] = d.metadata["page"] + 1
        docs.extend(loaded)
    if not docs:
        raise FileNotFoundError(f"No .txt/.pdf files found in {kb_dir}. Add files and retry.")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split into overlapping chunks and tag each with a stable chunk_id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
