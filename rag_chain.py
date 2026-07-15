"""Retrieval chain that answers questions with page-level citations."""

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from models import get_chat_model, get_embeddings

load_dotenv()

TOP_K = 4
# Optional soft filter: drop retrieved chunks whose relevance (0-1) is below this.
# Default 0.0 = keep all top-k (safe). Raise via MIN_RELEVANCE in .env to be stricter.
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.0"))

SYSTEM_PROMPT = """You are a careful assistant that answers questions about medical \
guidelines and research papers.

Rules:
1. Use ONLY the context below. Never use outside knowledge.
2. If the answer is not in the context, say exactly: "I could not find this in the provided documents."
3. Cite every claim with its source tag in square brackets, for example [guideline.pdf, p. 12].
4. Do not give medical advice, diagnosis, or treatment recommendations. \
Answers are for information only.

Context:
{context}"""


def get_vectorstore(persist_dir: str = "chroma_db") -> Chroma:
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),
    )


def format_docs(docs: list) -> str:
    """Format retrieved chunks with [file, page] tags the model can cite."""
    parts = []
    for doc in docs:
        name = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", 0) + 1
        parts.append(f"[{name}, p. {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer(question: str, persist_dir: str = "chroma_db") -> dict:
    """Answer a question from the indexed documents.

    Returns a dict with keys: answer (str) and sources (list of dicts).
    """
    if not os.path.isdir(persist_dir):
        return {
            "answer": "No documents are indexed yet. Upload one or more PDFs and click "
                      "**Index uploaded PDFs** in the sidebar, then ask again.",
            "sources": [],
        }

    vectorstore = get_vectorstore(persist_dir)

    # Retrieve with relevance scores (0-1, higher = closer) for transparency.
    try:
        scored = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
    except Exception:
        scored = [(d, None) for d in vectorstore.similarity_search(question, k=TOP_K)]

    if not scored:
        return {
            "answer": "No documents are indexed yet. Upload PDFs and index them first.",
            "sources": [],
        }

    # Soft relevance filter — OPT-IN. With MIN_RELEVANCE=0 (default) keep all
    # top-k, because some embedding/distance combos return negative relevance
    # for perfectly valid chunks, which would otherwise reject every answer.
    if MIN_RELEVANCE > 0:
        kept = [(d, s) for d, s in scored if s is None or s >= MIN_RELEVANCE]
    else:
        kept = scored
    if not kept:
        return {
            "answer": "I could not find this in the provided documents.",
            "sources": [],
        }

    docs = [d for d, _ in kept]

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )
    llm = get_chat_model()
    chain = prompt | llm
    response = chain.invoke({"context": format_docs(docs), "question": question})

    sources = [
        {
            "file": os.path.basename(d.metadata.get("source", "unknown")),
            "page": d.metadata.get("page", 0) + 1,
            "score": round(s, 3) if isinstance(s, (int, float)) else None,
            "excerpt": d.page_content[:300],
        }
        for d, s in kept
    ]
    return {"answer": response.content, "sources": sources}


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What are the main topics covered in the documents?"
    result = answer(question)
    print(result["answer"])
    print("\nSources used:")
    for s in result["sources"]:
        tag = f" (score {s['score']})" if s["score"] is not None else ""
        print(f"  {s['file']}, p. {s['page']}{tag}")
