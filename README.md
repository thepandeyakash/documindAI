# Production-Grade RAG Python App (Gemini edition)

A PDF question-answering app built with **FastAPI + Inngest + Qdrant + Streamlit**.
This version runs on Google's Gemini API free tier: embeddings and answer generation
both use a single free `GEMINI_API_KEY`.

- **Embeddings:** `gemini-embedding-001` via the official `google-genai` SDK
  (768 dimensions, L2-normalized; the same dimension is used for documents and queries)
- **Answer generation:** `gemini-3.8-flash` through Gemini's OpenAI-compatible endpoint
  (proxied by the Inngest dev server via `step.ai.infer`)
- **Vector store:** Qdrant collection `docs_gemini_768` (new, so it does not clash with
  the older 3072-dim OpenAI `docs` or 384-dim `docs_minilm_384` collections)

## Prerequisites (Windows)

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- Node.js (the Inngest dev server runs via `npx`)
- Docker Desktop (for Qdrant)
- A free Gemini API key: create one at https://aistudio.google.com/apikey

## Setup

```powershell
cd ProductionGradeRAGPythonApp-main
uv sync
copy .env.example .env
notepad .env   # set: GEMINI_API_KEY=AIza...
```

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | yes | Google Gemini API key used for embeddings and answer generation |
| `INNGEST_API_BASE` | no | Inngest dev-server API base polled by Streamlit (default `http://127.0.0.1:8288/v1`) |

## Run (4 terminals, all from the project folder)

1. Qdrant (persists into the local `qdrant_storage` folder):

```powershell
docker run --rm -p 6333:6333 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

2. Inngest dev server (runs the workflow steps, including the AI call proxy):

```powershell
npx inngest-cli@latest dev
```

3. FastAPI app (registers the Inngest functions on `http://127.0.0.1:8000`):

```powershell
uv run uvicorn main:app --reload --port 8000
```

4. Streamlit UI:

```powershell
uv run streamlit run streamlit_app.py
```

Open http://localhost:8501, upload a PDF to ingest it, then ask questions about it.

## How it works

Event names and UI behavior are unchanged from the original app:

- `rag/ingest_pdf` → chunk the PDF → embed chunks with Gemini → upsert into Qdrant `docs_gemini_768`
- `rag/query_pdf_ai` → embed the question → search Qdrant (filtered by the event's
  `source_id`, i.e. the PDF's file name, so only that document's chunks are retrieved)
  → answer with the Gemini chat model (the chat request is proxied by the Inngest dev
  server via `step.ai.infer`)

## Tests

Offline unit tests (mocked SDK, no key needed):

```powershell
uv run python -m unittest discover -s tests -v
```

Live smoke test (needs `GEMINI_API_KEY` in `.env` and Qdrant running):

```powershell
uv run python smoke_test.py                # generated sample PDF
uv run python smoke_test.py path\to\doc.pdf
```

## Known limitations

- `GEMINI_API_KEY` is required; embeddings and answer generation call Google's API over
  the network. Nothing runs locally and no models are downloaded.
- The free tier has low requests-per-minute limits (HTTP 429 responses are surfaced as
  clear rate-limit errors); large PDFs are embedded in 100-text batches.
- The old 3072-dim `docs` and 384-dim `docs_minilm_384` Qdrant collections are
  intentionally not reused — re-upload PDFs into the new `docs_gemini_768` collection.
- Model defaults (`gemini-embedding-001`, `gemini-3.8-flash`) are constants at the top of
  `data_loader.py` / `main.py`; change them there if a model is retired. Note
  `gemini-2.5-flash` is no longer available to new API users.
- The UI and API have no authentication — local use only.
- The original Inngest throttling/rate limiting (e.g. 1 ingest per `source_id` per 4 hours)
  is unchanged.
