"""Streamlit chat interface for the medical RAG chatbot."""

import os
import shutil

import streamlit as st

from ingest import clear_index, ingest
from models import PROVIDER
from rag_chain import answer

UPLOAD_DIR = "data/uploads"   # only the user's uploaded PDFs live here
PERSIST_DIR = "chroma_db"

st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🩺", layout="wide")

# ---------- light styling for a cleaner, portfolio-ready look ----------
st.markdown(
    """
    <style>
      .block-container {max-width: 900px; padding-top: 2.2rem;}
      h1 {font-weight: 800; letter-spacing: -0.5px;}
      .subtitle {color: #6b7280; font-size: 1.02rem; margin-top: -0.4rem;}
      .doc-chip {display:inline-block; background:#eef2ff; color:#3730a3;
        border-radius:999px; padding:3px 12px; margin:3px 4px 3px 0; font-size:0.82rem;}
      .empty-card {border:1px solid #e5e7eb; border-radius:14px; padding:26px 28px;
        background:#fafafa; margin-top:8px;}
      .empty-card h3 {margin-top:0;}
      div.stButton > button {border-radius:10px; font-weight:600;}
      [data-testid="stChatInput"] {border-radius:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- session state ----------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("indexed_files", [])
st.session_state.setdefault("scan_warnings", [])

indexed = os.path.isdir(PERSIST_DIR) and st.session_state.indexed_files

# ---------- sidebar ----------
with st.sidebar:
    st.header("📄 Documents")
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Index uploaded PDFs", type="primary", disabled=not uploaded, use_container_width=True):
        # Start from a clean upload folder so ONLY the current files are indexed.
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        paths = []
        for file in uploaded:
            path = os.path.join(UPLOAD_DIR, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            paths.append(path)
        with st.spinner("Indexing documents…"):
            n_chunks, warnings = ingest(paths, PERSIST_DIR, reset=True)
        st.session_state.indexed_files = [os.path.basename(p) for p in paths]
        st.session_state.scan_warnings = warnings
        st.session_state.messages = []  # old chat referred to old docs
        if n_chunks:
            st.success(f"Indexed {n_chunks} chunks from {len(paths)} file(s).")
        else:
            st.error("No extractable text found — see the warning below.")
        st.rerun()

    if indexed:
        st.markdown("**Active documents**", help="Answers are drawn only from these.")
        st.markdown(
            "".join(f"<span class='doc-chip'>{f}</span>" for f in st.session_state.indexed_files),
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Clear index", use_container_width=True):
            clear_index(PERSIST_DIR)
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.session_state.scan_warnings = []
            st.rerun()

    if st.session_state.scan_warnings:
        st.warning(
            "No readable text in: " + ", ".join(st.session_state.scan_warnings)
            + ". These look scanned — run OCR (e.g. `ocrmypdf`) before uploading."
        )

    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. PDFs are chunked and embedded into Chroma\n"
        "2. The top matching chunks are retrieved per question\n"
        "3. The LLM answers only from those chunks, with citations"
    )
    st.caption(f"Model provider: `{PROVIDER}`")

# ---------- main ----------
st.title("Medical RAG Chatbot")
st.markdown(
    "<p class='subtitle'>Ask questions about your medical guidelines and research papers. "
    "Every answer is cited. For information only — not medical advice.</p>",
    unsafe_allow_html=True,
)

if not indexed:
    st.markdown(
        "<div class='empty-card'><h3>👋 Get started</h3>"
        "<p>1. Upload one or more PDFs in the sidebar.<br>"
        "2. Click <b>Index uploaded PDFs</b>.<br>"
        "3. Ask questions here — answers come only from your documents, with page citations.</p>"
        "<p style='color:#6b7280'>Your answers are grounded strictly in the "
        "files you index, so nothing from other documents can leak in.</p>"
        "<p style='color:#6b7280;margin-bottom:0'>No PDFs handy? Two synthetic sample papers are in "
        "the <code>examples/</code> folder of the repo.</p></div>",
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                result = answer(question, PERSIST_DIR)
            st.markdown(result["answer"])

            if result["sources"]:
                with st.expander(f"Sources used ({len(result['sources'])})"):
                    for s in result["sources"]:
                        score = f" · match {s['score']}" if s.get("score") is not None else ""
                        st.markdown(f"**{s['file']}, p. {s['page']}**{score}")
                        st.caption(s["excerpt"] + "…")

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
