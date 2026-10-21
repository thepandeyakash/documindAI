import asyncio
import json
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

# Deployment platforms supply config as environment variables, but Streamlit
# Community Cloud stores secrets in .streamlit/secrets.toml (st.secrets). Mirror
# them into os.environ (without overriding real env vars) before any client is
# built, so the Inngest SDK and the helpers below see one config surface.
try:
    for _key, _value in dict(st.secrets).items():
        os.environ.setdefault(str(_key), str(_value))
except Exception:
    pass  # No secrets.toml locally; .env / process env only.


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    # Environment-driven mode (the SDK's standard switch): INNGEST_DEV=1 in the
    # local .env routes events to the dev server on localhost:8288; with
    # INNGEST_DEV unset (production) events go to Inngest Cloud and require
    # INNGEST_EVENT_KEY.
    return inngest.Inngest(app_id="rag_app")


# Set in production (Streamlit Community Cloud) to the deployed FastAPI backend's
# public URL, e.g. https://your-service.up.railway.app. When set, uploaded PDFs
# are handed to the backend via POST /upload (the ingest function runs there and
# can only read files on that host). Unset = local dev: save the file here.
BACKEND_URL = (os.getenv("BACKEND_URL") or "").rstrip("/")

# 100 MB backend upload limit + generous transfer time for large PDFs.
UPLOAD_TIMEOUT_S = 300


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


def upload_pdf_to_backend(file) -> dict:
    """Store the PDF on the deployed backend; returns {pdf_path, source_id}."""
    resp = requests.post(
        f"{BACKEND_URL}/upload",
        files={"file": (file.name, file.getvalue(), "application/pdf")},
        timeout=UPLOAD_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


async def send_rag_ingest_event(pdf_path: str, source_id: str) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": pdf_path,
                "source_id": source_id,
            },
        )
    )
    return result[0]


def _inngest_api_base() -> str:
    # Local dev server default. Production: Inngest Cloud's REST API
    # (INNGEST_API_BASE=https://api.inngest.com/v1).
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def _inngest_api_headers() -> dict:
    # The local dev server needs no auth; Inngest Cloud's REST API expects the
    # environment's signing key as a bearer token (per Inngest's docs on fetching
    # run status/output). Unset locally.
    signing_key = (os.getenv("INNGEST_SIGNING_KEY") or "").strip()
    return {"Authorization": f"Bearer {signing_key}"} if signing_key else {}


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url, headers=_inngest_api_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def fetch_run_output(run_id: str) -> dict | None:
    # The runs-list response is cached (~15s) by the dev server and can lag
    # behind completion; the single-run endpoint is authoritative.
    resp = requests.get(
        f"{_inngest_api_base()}/runs/{run_id}",
        headers=_inngest_api_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json().get("data") or {}
    return data.get("output")


# The dev server can flip a run to "Completed" shortly before its output is
# readable (and caches the runs-list response ~15s), so an empty output at the
# first sight of "Completed" is retried for this long before giving up.
OUTPUT_EMPTY_GRACE_S = 45.0


def _coerce_output(raw) -> dict:
    """Normalize a run output value to a dict.

    The dev server may deliver `output` either as a JSON object or as a
    JSON-encoded string; both must end up as the same dict for the UI.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def wait_for_run_output(event_id: str, timeout_s: float = 420.0, poll_interval_s: float = 0.5) -> dict:
    # Gemini on the free tier takes ~1-3 min per answer (measured), so the
    # previous 120s budget timed out before the run finished.
    start = time.time()
    last_status = None
    last_run_id = None
    empty_completed_since = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_run_id = run.get("run_id") or last_run_id
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                output = _coerce_output(run.get("output"))
                if not output:
                    # The single-run endpoint is authoritative; consult it
                    # before concluding the run produced nothing.
                    output = _coerce_output(fetch_run_output(last_run_id) if last_run_id else None)
                if output:
                    return output
                # "Completed" but the output is not readable yet: keep polling
                # within the grace window instead of returning an empty dict.
                if empty_completed_since is None:
                    empty_completed_since = time.time()
                elif time.time() - empty_completed_since > OUTPUT_EMPTY_GRACE_S:
                    return {}
            elif status in ("Failed", "Cancelled"):
                # Surface the step's real error (e.g. Gemini 429 quota, Qdrant
                # connection refused) instead of a bare "Function run Failed".
                out = _coerce_output(run.get("output"))
                if not out:
                    out = _coerce_output(fetch_run_output(last_run_id) if last_run_id else None)
                cause = ""
                if out:
                    cause = str(out.get("message") or out.get("error") or "")[:400]
                    if not cause:
                        cause = str(out)[:400]
                raise RuntimeError(
                    f"Function run {status} (event {event_id}, run {last_run_id})"
                    + (f": {cause}" if cause else "")
                )
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out after {timeout_s:.0f}s waiting for run output "
                f"(event {event_id}, last status: {last_status})"
            )
        time.sleep(poll_interval_s)


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None and st.button("Ingest PDF"):
    with st.spinner("Saving file and triggering ingestion..."):
        pdf_path: str | None = None
        source_id = uploaded.name
        if BACKEND_URL:
            # Production: the ingest function runs on the backend host, so the
            # file must be stored there (Streamlit CC's disk is not shared).
            try:
                info = upload_pdf_to_backend(uploaded)
                pdf_path = info.get("pdf_path")
                source_id = info.get("source_id") or source_id
            except requests.RequestException as exc:
                st.error(f"Upload to the backend failed ({BACKEND_URL}/upload): {exc}")
        else:
            # Local dev: same machine, store next to this app and use its path.
            pdf_path = str(save_uploaded_pdf(uploaded).resolve())
        if pdf_path:
            event_id = asyncio.run(send_rag_ingest_event(pdf_path, source_id))
            # Phase 1: the 4h-per-filename rate limit blocks run creation entirely,
            # so first confirm a run actually exists (dev-server list caches ~15s).
            deadline = time.time() + 45
            run_created = bool(fetch_runs(event_id))
            while not run_created and time.time() < deadline:
                time.sleep(0.5)
                run_created = bool(fetch_runs(event_id))
            if not run_created:
                st.warning(
                    f"No ingest run was created for {source_id} (event {event_id}). "
                    "This happens when the same file name was already ingested within "
                    "the last 4 hours (rate limit) — check the Inngest dashboard."
                )
                outcome = None
            else:
                # Phase 2: wait for the outcome; failing runs retry for minutes, so
                # allow long enough to report the real error instead of a timeout.
                try:
                    outcome = wait_for_run_output(event_id, timeout_s=420.0)
                except (TimeoutError, RuntimeError, requests.RequestException) as exc:
                    st.error(f"Ingestion failed: {exc}")
                    outcome = None
        else:
            outcome = None
    if outcome is not None:
        st.success(f"Ingested {outcome.get('ingested', 0)} chunk(s) from: {source_id}")

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, source_id: str, top_k: int) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                # Must match the source_id used at ingest time (the PDF's file name)
                # so retrieval only searches that document's chunks.
                "source_id": source_id,
                "top_k": top_k,
            },
        )
    )

    return result[0]


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    source_id = st.text_input(
        "PDF file name to query",
        help="Must match the ingested PDF's file name (e.g. report.pdf); "
        "retrieval only searches that document's chunks.",
    )
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip() and source_id.strip():
        with st.spinner("Sending event and generating answer..."):
            # Fire-and-forget event to Inngest for observability/workflow
            event_id = asyncio.run(
                send_rag_query_event(question.strip(), source_id.strip(), int(top_k))
            )
            # Poll the local Inngest API for the run's output
            query_error = None
            try:
                output = wait_for_run_output(event_id)
            except (TimeoutError, RuntimeError, requests.RequestException) as exc:
                query_error = str(exc)
                st.error(f"Query failed: {exc}")
                output = {}
        answer = output.get("answer", "")
        sources = output.get("sources", [])
        num_contexts = output.get("num_contexts")

        if answer:
            st.subheader("Answer")
            st.write(answer)
            if num_contexts is not None:
                st.caption(f"Based on {num_contexts} retrieved context chunk(s).")
        elif query_error:
            # st.error above already reports the real cause (timeout, failed
            # run, ...); don't pile a misleading "no output" warning on top.
            pass
        elif not output:
            st.warning("The run produced no output. Check the Inngest dashboard for this event's run.")
        else:
            st.warning("The run completed but returned no answer field.")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")

