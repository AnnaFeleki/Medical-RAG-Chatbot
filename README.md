# Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about medical guidelines and research papers, with **page-level citations for every claim**. Built with LangChain, Chroma, and Streamlit, with a **FastAPI** service for programmatic access — runs fully local by default, or on OpenAI with a one-line change.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![LangChain](https://img.shields.io/badge/LangChain-0.3-green) ![Streamlit](https://img.shields.io/badge/Streamlit-app-ff4b4b) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

![Architecture](docs/architecture.png)

<!-- Add a short screen recording of the app here — the single biggest credibility boost:
     ![Demo](docs/demo.gif)
     Record ~10s (upload a PDF → Index → ask a question → show the cited answer).
     Tools: Peek (Linux), LICEcap / ScreenToGif (Windows), or macOS screen-record → convert to GIF. -->

> 📹 **Add a `docs/demo.gif`** (a 10-second capture of a question → cited answer) at the top of this README — it's the fastest way to make the project land. See the HTML comment above for how.

## Why this matters

LLMs hallucinate. In healthcare that is unacceptable. This chatbot answers **only** from the documents you give it, refuses to answer when the context does not contain the answer, and cites the exact file and page for every statement so users can verify the source.

## Features

- **Grounded, cited answers** — every claim carries a `[filename.pdf, p. 12]` citation, and the model refuses to answer outside the provided documents.
- **Document isolation** — each time you index, the store is rebuilt from exactly the PDFs you upload, so content from other documents can never leak into an answer.
- **Runs fully local** with open models via Ollama by default: free, no API key, and no document ever leaves your machine — ideal for sensitive clinical data.
- **Optional OpenAI mode** with a one-line `.env` change.
- **Scanned-PDF detection** — warns when an uploaded PDF has no extractable text and needs OCR.
- **Clean Streamlit chat interface** with a source-inspection panel and retrieval match scores.

## Quickstart (local open models, default)

1. Install [Ollama](https://ollama.com) and pull the models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

2. Install and run:

```bash
git clone https://github.com/AnnaFeleki/medical-rag-chatbot.git
cd medical-rag-chatbot
pip install -r requirements.txt
cp .env.example .env

streamlit run app.py
```

Then upload PDFs in the sidebar and click **Index uploaded PDFs** — or try the two synthetic sample papers in [`examples/`](examples) to see it work straight away. (You can also index from the CLI: `python ingest.py --pdf-dir examples`.)

To use OpenAI instead, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` in `.env`. Note: if you switch provider after indexing, click **Clear index** (or delete `chroma_db/`) and re-index, because embeddings from different providers are not compatible.

## REST API (FastAPI)

The same ingest + retrieval core powers both the Streamlit app and an HTTP API, so any frontend or service can consume it:

```bash
uvicorn api:app --reload      # interactive Swagger docs at http://localhost:8000/docs
```

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/status` | Whether an index exists, and which files |
| `POST` | `/index` | Upload PDFs (multipart) and rebuild the index |
| `POST` | `/ask` | Ask a question → grounded, cited answer |
| `DELETE` | `/index` | Clear the index |

```bash
# index a PDF, then ask a question
curl -F "files=@guideline.pdf" http://localhost:8000/index
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the first-line treatment?"}'
```

## How it works

1. PDFs are split into 1000-character chunks with overlap (`RecursiveCharacterTextSplitter`).
2. Chunks are embedded — locally with **nomic-embed-text** (Ollama) by default, or **OpenAI text-embedding-3-small** in OpenAI mode — and stored in Chroma.
3. Each time you index, the Chroma store is **reset** so it holds only your current documents.
4. At question time the top-k chunks are retrieved and passed to the LLM.
5. A strict system prompt forces grounded answers with `[file, p. N]` citations, and refuses when the answer is not in the retrieved context.

## Evaluation

Retrieval quality is measurable, not vibes. `eval.py` reports **Hit-rate@k** and **MRR@k** over a small labelled question set. Index the sample papers, then run it:

```bash
python ingest.py --pdf-dir examples
python eval.py --questions eval/questions.example.json --k 4
```

Each item maps a question to the file that should answer it (the included examples are pre-labelled against the two sample papers). Add your own Q→source pairs to track retrieval quality as you tune chunking or embeddings.

## Project structure

```
medical-rag-chatbot/
  app.py          Streamlit chat interface
  api.py          FastAPI service (same core as the Streamlit app)
  ingest.py       PDF loading, chunking, embedding into Chroma (index reset on each run)
  rag_chain.py    Retrieval chain with citation prompt
  models.py       Provider switch (Ollama / OpenAI)
  eval.py         Retrieval evaluation (Hit-rate@k, MRR@k)
  examples/       Two synthetic sample papers to try immediately
  data/uploads/   PDFs uploaded via the app (git-ignored)
  data/pdfs/      Optional: PDFs for CLI ingest
  chroma_db/      Vector store (created on first index; git-ignored)
  docs/           Architecture diagram and screenshots
```

## Disclaimer

This tool is for information retrieval only. It does not provide medical advice, diagnosis, or treatment recommendations.

## License

MIT. Built by [Anna Feleki](https://github.com/AnnaFeleki).
