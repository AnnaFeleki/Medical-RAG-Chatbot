"""Ingest PDF files into a Chroma vector store for retrieval.

Key behaviour: by default `ingest()` REPLACES the index (reset=True), so the
vector store always reflects exactly the documents you pass in. This prevents
stale documents from a previous session leaking into answers.
"""

import argparse
import os
import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import get_embeddings

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# A page with fewer characters than this is treated as "no extractable text"
# (usually a scanned image that needs OCR).
MIN_CHARS_PER_PAGE = 40


def clear_index(persist_dir: str = "chroma_db") -> None:
    """Drop any existing index.

    Uses Chroma's API so it works even while a client handle is alive in the
    same process (e.g. a running Streamlit app). Deleting the folder instead
    can raise "readonly database" in that situation, so that is only a fallback.
    """
    try:
        store = Chroma(persist_directory=persist_dir, embedding_function=get_embeddings())
        store.delete_collection()
    except Exception:
        shutil.rmtree(persist_dir, ignore_errors=True)


def _pdf_paths(source) -> list:
    """Accept a directory path OR an explicit list of file paths; return PDF paths."""
    if isinstance(source, (list, tuple)):
        return [p for p in source if str(p).lower().endswith(".pdf") and os.path.isfile(p)]
    if isinstance(source, str) and os.path.isdir(source):
        return [
            os.path.join(source, name)
            for name in sorted(os.listdir(source))
            if name.lower().endswith(".pdf")
        ]
    return []


def load_pdfs(source):
    """Load PDFs (from a dir or list of paths) as page-level documents.

    Returns (docs, warnings). `warnings` names any PDF that yielded almost no
    text, so the caller can tell the user it probably needs OCR.
    """
    docs, warnings = [], []
    for path in _pdf_paths(source):
        name = os.path.basename(path)
        pages = PyPDFLoader(path).load()
        text_chars = sum(len(p.page_content.strip()) for p in pages)
        if text_chars < MIN_CHARS_PER_PAGE * max(len(pages), 1):
            warnings.append(name)
        docs.extend(pages)
        print(f"Loaded {name} ({len(pages)} pages, {text_chars} chars)")
    return docs, warnings


def ingest(source="data/pdfs", persist_dir="chroma_db", reset=True):
    """Chunk, embed, and store PDFs.

    Args:
        source: a directory path or an explicit list of PDF file paths.
        persist_dir: Chroma persistence folder.
        reset: if True (default), delete any existing index first so the store
            contains ONLY the documents passed in this call.

    Returns:
        (n_chunks, warnings) — number of chunks indexed and a list of filenames
        that had no extractable text.
    """
    docs, warnings = load_pdfs(source)

    if reset:
        clear_index(persist_dir)

    if not docs:
        print(f"No readable PDF content found in {source}.")
        return 0, warnings

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
    )
    print(f"Indexed {len(chunks)} chunks from {len(docs)} pages into {persist_dir}")
    return len(chunks), warnings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into a Chroma vector store")
    parser.add_argument("--pdf-dir", default="data/pdfs", help="Folder containing PDF files")
    parser.add_argument("--persist-dir", default="chroma_db", help="Chroma persistence folder")
    parser.add_argument(
        "--no-reset", action="store_true",
        help="Add to the existing index instead of replacing it",
    )
    args = parser.parse_args()
    n, warns = ingest(args.pdf_dir, args.persist_dir, reset=not args.no_reset)
    if warns:
        print("WARNING: no extractable text (may be scanned, needs OCR): " + ", ".join(warns))
