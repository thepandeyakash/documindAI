import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import TASK_TYPE_QUERY, load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage, normalize_source_id
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

# Answer generation via Google Gemini's OpenAI-compatible endpoint (free-tier eligible).
# The inngest experimental ai.openai.Adapter speaks the OpenAI chat schema, and Gemini
# natively serves that schema at .../v1beta/openai/ — only GEMINI_API_KEY is needed.
# Verified live: gemini-2.5-flash is closed to new users; gemini-3.8-flash is the
# currently available free-tier text model (see Google's pricing page).
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_CHAT_MODEL = "gemini-3.8-flash"

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
  ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        # Canonical form (whitespace-normalized) so the stored payload, the
        # deterministic point IDs, and the query filter all agree.
        source_id = normalize_source_id(ctx.event.data.get("source_id", pdf_path))
        chunks = load_and_chunk_pdf(pdf_path)
        if not chunks:
            # A scanned/image-only PDF parses to nothing; fail loudly rather
            # than reporting a misleading "Ingested 0 chunk(s)".
            raise RuntimeError(
                f"No extractable text found in '{pdf_path}'. The PDF is likely "
                "scanned or image-only, so there is nothing to ingest — run OCR "
                "on it first."
            )
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        store = QdrantStorage()
        store.upsert(ids, vecs, payloads)
        # Report points confirmed in Qdrant for this source, not chunks processed.
        return RAGUpsertResult(ingested=store.count_by_source(source_id))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, source_id: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question], task_type=TASK_TYPE_QUERY)[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k, source_id=source_id)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    # Same ID the ingest function stored in each chunk's "source" payload field
    # (the uploaded PDF's file name), so retrieval only matches that document.
    source_id = normalize_source_id(ctx.event.data.get("source_id") or "")

    found = await ctx.step.run("embed-and-search", lambda: _search(question, source_id, top_k), output_type=RAGSearchResult)

    if not found.contexts:
        # No chunks for this document: skip the LLM call and say so clearly.
        return {
            "answer": (
                f"No ingested chunks found for source '{source_id}'. "
                "Upload and ingest this PDF first, then ask again."
            ),
            "sources": [],
            "num_contexts": 0,
        }

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not gemini_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set, so the Gemini answer generation cannot be "
            "requested. Create a free key at https://aistudio.google.com/apikey and "
            "add GEMINI_API_KEY=... to your .env file (see README.md)."
        )

    adapter = ai.openai.Adapter(
        auth_key=gemini_key,
        base_url=GEMINI_OPENAI_BASE_URL,
        model=GEMINI_CHAT_MODEL
    )

    try:
        res = await ctx.step.ai.infer(
            "llm-answer",
            adapter=adapter,
            body={
                "max_tokens": 1024,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "You answer questions using only the provided context."},
                    {"role": "user", "content": user_content}
                ]
            }
        )
        answer = res["choices"][0]["message"]["content"].strip()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Gemini answer generation failed (model={GEMINI_CHAT_MODEL}). Check that "
            "GEMINI_API_KEY is valid and that the model is available for your key "
            f"(https://ai.google.dev/gemini-api/docs/models). Original error: {exc}"
        ) from exc
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])