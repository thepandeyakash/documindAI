import math
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

load_dotenv()

# Embeddings via Google's Gemini API (free-tier eligible, only GEMINI_API_KEY needed).
# gemini-embedding-001 returns 3072-dim by default; we request 768 dims explicitly and
# use the same dimension for documents and queries.
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
# The Gemini embedding endpoint accepts at most 100 texts per request.
EMBED_BATCH_SIZE = 100

TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_TYPE_QUERY = "RETRIEVAL_QUERY"

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def _require_gemini_key() -> str:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set, so Gemini embeddings cannot be requested. "
            "Create a free key at https://aistudio.google.com/apikey and add "
            "GEMINI_API_KEY=... to your .env file (see README.md)."
        )
    return key


def _raise_for_api_error(exc: Exception, action: str) -> None:
    code = getattr(exc, "code", None)
    if callable(code):
        try:
            code = code()
        except Exception:
            code = None
    detail = (str(getattr(exc, "message", None) or exc))[:300]

    if code in (401, 403):
        raise RuntimeError(
            f"Gemini rejected the API key while {action} (HTTP {code}): {detail}. "
            "Check that GEMINI_API_KEY is valid, not revoked, and that the Gemini "
            "API is enabled for its Google Cloud project."
        ) from exc
    if code == 429:
        raise RuntimeError(
            f"Gemini rate-limited the request while {action} (HTTP 429): {detail}. "
            "Free-tier keys have per-minute AND daily request limits; a daily-quota "
            "message ('exceeded your current quota') only clears after the quota "
            "resets (midnight Pacific) — wait, or use a key with available quota."
        ) from exc
    if code == 404:
        raise RuntimeError(
            f"Gemini returned 404 while {action}: {detail}. "
            f"Model '{EMBED_MODEL}' may have been retired; check "
            "https://ai.google.dev/gemini-api/docs/models"
        ) from exc
    raise RuntimeError(
        f"Gemini API error while {action} (HTTP {code}): {detail}"
    ) from exc


def _parse_embeddings(resp, expected_count: int) -> list[list[float]]:
    """Validate and normalize a google-genai embed_content response to [n][768] floats."""
    embeddings = getattr(resp, "embeddings", None)
    if not embeddings:
        raise RuntimeError(
            "Unexpected Gemini embeddings response (no embeddings returned): "
            f"{str(resp)[:200]}"
        )
    if len(embeddings) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} embeddings from Gemini but received "
            f"{len(embeddings)}."
        )

    vectors: list[list[float]] = []
    for emb in embeddings:
        values = getattr(emb, "values", None)
        if not values:
            raise RuntimeError(f"Unexpected Gemini embedding entry: {str(emb)[:200]}")
        try:
            vector = [float(x) for x in values]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Gemini embedding contains non-numeric values: {str(values)[:100]}"
            ) from exc
        if len(vector) != EMBED_DIM:
            raise RuntimeError(
                f"Expected {EMBED_DIM}-dimensional embeddings for '{EMBED_MODEL}' "
                f"(output_dimensionality={EMBED_DIM}) but received {len(vector)} "
                "dimensions. The vector store configuration and embedding model "
                "are out of sync."
            )
        # Gemini's docs recommend L2-normalizing when a reduced output
        # dimensionality is requested; do it so vectors are safe for cosine/dot use.
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            raise RuntimeError("Gemini returned an all-zero embedding.")
        vectors.append([x / norm for x in vector])
    return vectors


def embed_texts(
    texts: list[str], task_type: str = TASK_TYPE_DOCUMENT
) -> list[list[float]]:
    if not texts:
        return []
    key = _require_gemini_key()
    client = genai.Client(api_key=key)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBED_DIM,
                ),
            )
        except Exception as exc:
            _raise_for_api_error(exc, "generating embeddings")
        vectors.extend(_parse_embeddings(resp, expected_count=len(batch)))
    return vectors
