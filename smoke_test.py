"""Minimal end-to-end smoke test for the Gemini-backed RAG pipeline.

Checks, in order (no Inngest dev server required):
  1. PDF loading + chunking
  2. Live embeddings via Google Gemini (must be 768-dim)
  3. Upsert + search in a local Qdrant instance (http://localhost:6333)
  4. Live answer generation via Google Gemini

Usage:
    uv run python smoke_test.py                # uses a generated one-page sample PDF
    uv run python smoke_test.py path\\to\\doc.pdf

GEMINI_API_KEY is read from the environment or .env. The key is never printed.
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SAMPLE_TEXT = (
    "The capital of France is Paris. "
    "Python is a popular programming language. "
    "Retrieval augmented generation retrieves relevant document chunks to answer questions."
)


def _write_sample_pdf(path: Path) -> None:
    stream = f"BT /F1 12 Tf 50 750 Td ({SAMPLE_TEXT}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(pdf))
        pdf += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(pdf)
    pdf += f"xref\n0 {len(objs) + 1}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n".encode()
    pdf += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    )
    path.write_bytes(pdf)


def main() -> int:
    if not (os.getenv("GEMINI_API_KEY") or "").strip():
        print("BLOCKED: GEMINI_API_KEY is not set. Add GEMINI_API_KEY=... to .env (see README.md).")
        return 1

    # Imported after the key check so a missing key fails fast with a clear message.
    from google import genai

    from data_loader import (
        EMBED_DIM,
        EMBED_MODEL,
        TASK_TYPE_QUERY,
        embed_texts,
        load_and_chunk_pdf,
    )
    from main import GEMINI_CHAT_MODEL
    from vector_db import QdrantStorage

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = Path(tempfile.gettempdir()) / "rag_smoke_test_sample.pdf"
        _write_sample_pdf(pdf_path)
    print(f"[1/4] chunking PDF: {pdf_path}")
    chunks = load_and_chunk_pdf(str(pdf_path))
    if not chunks:
        print("BLOCKED: no text chunks extracted from the PDF.")
        return 1
    print(f"      -> {len(chunks)} chunk(s)")

    print(f"[2/4] embedding via Gemini (live call, model={EMBED_MODEL})")
    vectors = embed_texts(chunks)  # raises a descriptive error on any provider problem
    dims = {len(v) for v in vectors}
    if dims != {EMBED_DIM}:
        print(f"BLOCKED: expected {EMBED_DIM}-dim vectors, got dims {dims}")
        return 1
    print(f"      -> {len(vectors)} vector(s) x {EMBED_DIM} dims")

    print("[3/4] upsert + search in Qdrant (http://localhost:6333)")
    store = QdrantStorage()
    source = pdf_path.name
    # Qdrant point IDs must be integers or UUIDs; mirror the app's UUID5 scheme.
    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"smoke:{source}:{i}"))
        for i in range(len(chunks))
    ]
    store.upsert(
        ids,
        vectors,
        [{"source": source, "text": c} for c in chunks],
    )
    question = "What is the capital of France?"
    query_vec = embed_texts([question], task_type=TASK_TYPE_QUERY)[0]
    found = store.search(query_vec, top_k=3, source_id=source)
    if not found["contexts"]:
        print("BLOCKED: Qdrant search returned no contexts after upsert.")
        return 1
    print(f"      -> retrieved {len(found['contexts'])} context(s) from {found['sources']}")

    print(f"[4/4] answer generation via Gemini (live call, model={GEMINI_CHAT_MODEL})")
    context_block = "\n\n".join(f"- {c}" for c in found["contexts"])
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )
    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    try:
        resp = client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=user_content,
            config=genai.types.GenerateContentConfig(
                system_instruction="You answer questions using only the provided context.",
                temperature=0.2,
                max_output_tokens=256,
            ),
        )
    except Exception as exc:
        print(f"BLOCKED: answer generation failed: {str(exc)[:300]}")
        return 1
    answer = (resp.text or "").strip()
    if not answer:
        print("BLOCKED: Gemini returned an empty answer.")
        return 1
    print(f"      -> answer: {answer[:300]}")

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
