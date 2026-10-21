"""Regression tests for source_id normalization and Qdrant round-trip.

Covers the bug where a PDF filename containing a double space
('Unit 3  Social Media_sk.pdf') was stored verbatim, so a query typed with
normal spacing matched nothing and reported "No ingested chunks found".
No Gemini calls: vectors are synthetic.
"""
import unittest
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from vector_db import QdrantStorage, normalize_source_id

QDRANT_URL = "http://localhost:6333"
DIM = 768


def _qdrant_available() -> bool:
    try:
        QdrantClient(url=QDRANT_URL, timeout=3).get_collections()
        return True
    except Exception:
        return False


class NormalizeSourceIdTest(unittest.TestCase):
    def test_double_space_collapsed(self):
        self.assertEqual(
            normalize_source_id("Unit 3  Social Media_sk.pdf"),
            "Unit 3 Social Media_sk.pdf",
        )

    def test_whitespace_and_none(self):
        self.assertEqual(normalize_source_id("  a b.pdf "), "a b.pdf")
        self.assertEqual(normalize_source_id(None), "")
        self.assertEqual(normalize_source_id(""), "")


@unittest.skipUnless(_qdrant_available(), "Qdrant not reachable on localhost:6333")
class SourceRoundTripTest(unittest.TestCase):
    """Ingest then query the same PDF through the real QdrantStorage."""

    def setUp(self):
        # Unique collection per test so nothing touches docs_gemini_768.
        self.collection = f"docs_test_{uuid.uuid4().hex[:12]}"
        self.client = QdrantClient(url=QDRANT_URL, timeout=10)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )
        self.store = QdrantStorage(collection=self.collection, dim=DIM)

    def tearDown(self):
        self.client.delete_collection(self.collection)

    def _vec(self, seed: int) -> list[float]:
        v = [((seed * 31 + i * 7) % 100) / 100.0 for i in range(DIM)]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v]

    def test_double_space_filename_ingest_then_query(self):
        # Ingest side: filename as uploaded (double space), as main.py would
        # receive it in the event payload before normalization.
        raw_name = "Unit 3  Social Media_sk.pdf"
        chunks = [f"chunk {i} about social media" for i in range(4)]
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{raw_name}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": raw_name, "text": c} for c in chunks]
        self.store.upsert(ids, [self._vec(i) for i in range(len(chunks))], payloads)

        # The success count must reflect points confirmed in Qdrant for the
        # canonical name — the query side uses single-space input.
        self.assertEqual(self.store.count_by_source("Unit 3 Social Media_sk.pdf"), 4)
        # Old irregular form must not linger as a separate identity.
        self.assertEqual(self.store.count_by_source(raw_name), 4)

        # Query side: same PDF typed with normal spacing finds the chunks.
        found = self.store.search(self._vec(0), top_k=3, source_id="Unit 3 Social Media_sk.pdf")
        self.assertEqual(len(found["contexts"]), 3)
        self.assertEqual(found["sources"], ["Unit 3 Social Media_sk.pdf"])

    def test_other_sources_not_leaked(self):
        self.store.upsert(
            [str(uuid.uuid4()) for _ in range(2)],
            [self._vec(1), self._vec(2)],
            [{"source": "a.pdf", "text": "aaa"}, {"source": "b.pdf", "text": "bbb"}],
        )
        found = self.store.search(self._vec(1), top_k=5, source_id="a.pdf")
        self.assertEqual(found["sources"], ["a.pdf"])
        self.assertEqual(found["contexts"], ["aaa"])


if __name__ == "__main__":
    unittest.main()
