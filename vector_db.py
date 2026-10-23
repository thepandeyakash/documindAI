import os

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, FieldCondition, Filter, MatchValue


def normalize_source_id(value) -> str:
    """Canonicalize a source id so ingest and query always agree.

    Filenames can contain double spaces (e.g. 'Unit 3  x.pdf'); Qdrant's
    MatchValue filter is an exact string match, so both the stored payload and
    the search filter must use the same whitespace-normalized form.
    """
    return " ".join(str(value or "").split())


class QdrantStorage:
    # "docs_gemini_768" (768-dim, gemini-embedding-001) is a new collection so it never
    # collides with the older 3072-dim OpenAI "docs" or 384-dim "docs_minilm_384"
    # collections.
    # Local Docker Qdrant by default; QDRANT_URL / QDRANT_API_KEY (from the
    # process environment or .env, loaded by the caller) point this at a
    # Qdrant Cloud cluster instead. An explicit url argument wins over env.
    def __init__(self, url=None, collection="docs_gemini_768", dim=768):
        url = (
            (url or "").strip()
            or (os.getenv("QDRANT_URL") or "").strip()
            or "http://localhost:6333"
        )
        api_key = (os.getenv("QDRANT_API_KEY") or "").strip()
        client_kwargs = {"url": url, "timeout": 30}
        if api_key:
            # Local Qdrant needs no key; pass one only when configured (Cloud).
            client_kwargs["api_key"] = api_key
        self.client = QdrantClient(**client_kwargs)
        self.collection = collection
        try:
            exists = self.client.collection_exists(self.collection)
        except Exception as exc:
            # Connection refused here means the Qdrant server itself is down —
            # say so explicitly instead of surfacing a raw transport error.
            hint = (
                "Start the container with: docker start rag-qdrant"
                if url.startswith(("http://localhost", "http://127.0.0.1"))
                else "Check QDRANT_URL and QDRANT_API_KEY in the backend environment (or .env locally)."
            )
            raise RuntimeError(
                f"Qdrant is not reachable at {url} ({exc.__class__.__name__}). {hint}"
            ) from exc
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        # Normalize the stored source so the exact-match filter (which
        # normalizes identically) always finds these points, regardless of
        # irregular whitespace in the original filename.
        payloads = [{**p, "source": normalize_source_id(p.get("source"))} for p in payloads]
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def count_by_source(self, source_id: str) -> int:
        """Points actually stored in Qdrant for this source (not chunks processed)."""
        res = self.client.count(
            collection_name=self.collection,
            count_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=normalize_source_id(source_id)))]
            ),
            exact=True,
        )
        return res.count

    def search(self, query_vector, top_k: int = 5, source_id: str | None = None):
        # Restrict retrieval to one document's chunks when source_id is given, so a
        # query about an uploaded PDF can never match chunks from other PDFs.
        query_filter = None
        if source_id:
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=normalize_source_id(source_id)))]
            )
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            query_filter=query_filter,
            with_payload=True,
            limit=top_k
        )
        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}