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
| `QDRANT_URL` | no | Qdrant URL (default `http://localhost:6333`; set the Qdrant Cloud cluster URL in production) |
| `QDRANT_API_KEY` | no | Qdrant Cloud API key (empty for local Qdrant) |
| `INNGEST_DEV` | local | `INNGEST_DEV=1` routes Inngest to the local dev server; **leave unset in production** |
| `INNGEST_EVENT_KEY` | production | Inngest event key; lets the Streamlit app send events to Inngest Cloud |
| `INNGEST_SIGNING_KEY` | production | Inngest signing key; backend verifies Inngest requests, Streamlit authenticates REST polling |
| `INNGEST_API_BASE` | no | Inngest REST API base polled by Streamlit (default `http://127.0.0.1:8288/v1`; production: `https://api.inngest.com/v1`) |
| `INNGEST_ENV` | production | Inngest environment name; must match on the FastAPI and Streamlit apps (e.g. `production`) |
| `BACKEND_URL` | production | Public URL of the deployed FastAPI backend; when set, Streamlit uploads PDFs to `<BACKEND_URL>/upload` |
| `HOST` / `PORT` | no | Only used by `python main.py`; Railway supplies `PORT` automatically |
| `UPLOAD_DIR` | no | Backend upload directory (default `uploads/`); point at a volume for durable storage |

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

## Production deployment

The same code runs locally and in production; only configuration differs:
`INNGEST_DEV=1` locally (dev server, no keys) vs. `INNGEST_DEV` unset plus
Inngest Cloud keys in production. Suggested order:

### 1. Qdrant Cloud

Create a cluster at https://cloud.qdrant.io and note its URL and API key.

### 2. Railway (FastAPI backend)

1. Create a Railway service from this folder. Nixpacks detects `uv.lock`,
   `.python-version` (3.13) and the `Procfile` (`web: uv run python main.py`);
   the app binds `0.0.0.0` and listens on Railway's `$PORT`.
2. Set service variables: `GEMINI_API_KEY`, `QDRANT_URL` (Qdrant Cloud URL),
   `QDRANT_API_KEY`, `INNGEST_ENV=production`, and `INNGEST_SIGNING_KEY`
   (from step 3).
3. Generate a public domain (Settings → Networking) so Inngest Cloud can reach
   `https://<your-railway-domain>/api/inngest`.

### 3. Inngest Cloud

1. Create an app whose environment name matches `INNGEST_ENV` (e.g.
   `production`).
2. Sync the app by pointing Inngest at `https://<your-railway-domain>/api/inngest`.
   The backend verifies these requests with the signing key, so
   `INNGEST_SIGNING_KEY` must be set on Railway.
3. Copy that environment's **event key** and **signing key** into Railway
   (`INNGEST_SIGNING_KEY`) and into the Streamlit secrets (step 4).

### 4. Streamlit Community Cloud

1. Deploy the same repo; in App settings set the main file path to
   `streamlit_app.py` and choose Python 3.13 in Advanced settings. Dependencies
   install from the pinned `requirements.txt` — regenerate it after dependency
   changes: `uv export --frozen --no-hashes --no-emit-project -o requirements.txt`.
2. Add secrets (App settings → Secrets); the app mirrors them into environment
   variables automatically:

   ```toml
   INNGEST_EVENT_KEY = "..."        # event key from step 3
   INNGEST_SIGNING_KEY = "..."      # signing key from step 3
   INNGEST_API_BASE = "https://api.inngest.com/v1"
   INNGEST_ENV = "production"       # must match the backend
   BACKEND_URL = "https://<your-railway-domain>"
   ```

3. Redeploy after adding secrets.

### Production data flow

- **Ingest:** Streamlit uploads the PDF to `POST /upload` on the backend (the two
  apps do not share a filesystem), then sends `rag/ingest_pdf`; the ingest
  function reads the file stored on the backend host and upserts chunks into the
  Qdrant Cloud collection `docs_gemini_768`.
- **Query:** Streamlit sends `rag/query_pdf_ai` (text-only event) and polls
  `https://api.inngest.com/v1/...` with the signing key as bearer token for the
  run's output.

Notes:

- Railway's filesystem is ephemeral; uploaded PDFs are reproducible from Qdrant.
  Set `UPLOAD_DIR` to a mounted volume if uploads must survive deploys.
- Events only trigger runs if `INNGEST_ENV` matches between the two apps and the
  keys belong to that same Inngest environment — this is the most common
  misconfiguration.

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
- The UI and API have no authentication. The `/upload` endpoint accepts PDFs
  from anyone who can reach it (filenames are sanitized and non-PDF/empty/
  >100 MB uploads are rejected), so add authentication and HTTPS-scoped access
  control before exposing the backend to untrusted traffic.
- The original Inngest throttling/rate limiting (e.g. 1 ingest per `source_id` per 4 hours)
  is unchanged.
