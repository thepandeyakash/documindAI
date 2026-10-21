"""Regression tests for source_id normalization and Qdrant round-trip.

Covers the bug where a PDF filename containing a double space
('Unit 3  Social Media_sk.pdf') was stored verbatim, so a query typed with
normal spacing matched nothing and reported "No ingested chunks found".
No Gemini calls: vectors are synthetic.
"""
import os
import unittest
import uuid
from unittest import mock

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
        # Hermetic: earlier test modules run load_dotenv(), so the real .env may
        # point QDRANT_URL/QDRANT_API_KEY at Qdrant Cloud. This round-trip must
        # exercise the local server above (and only it), whatever the developer
        # has configured, so blank both vars while building the store.
        with mock.patch.dict(os.environ, {"QDRANT_URL": "", "QDRANT_API_KEY": ""}):
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


class QdrantConnectionConfigTest(unittest.TestCase):
    """QDRANT_URL / QDRANT_API_KEY handling: local Docker vs Qdrant Cloud.

    QdrantClient is replaced by a fake that records the kwargs it was called
    with, so these tests make no network calls and never read a real key.
    """

    def _build(self, set_env=None, unset_env=()):
        captured = {}

        class _FakeClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def collection_exists(self, name):
                return True

        with mock.patch.dict(os.environ, set_env or {}, clear=False):
            for key in unset_env:
                os.environ.pop(key, None)
            with mock.patch("vector_db.QdrantClient", _FakeClient):
                store = QdrantStorage()
        return captured, store

    def test_unset_env_defaults_to_local_without_api_key(self):
        kwargs, store = self._build(unset_env=("QDRANT_URL", "QDRANT_API_KEY"))
        self.assertEqual(kwargs["url"], "http://localhost:6333")
        self.assertNotIn("api_key", kwargs)
        # Requirement: collection name and dimensions stay exactly as before.
        self.assertEqual(store.collection, "docs_gemini_768")

    def test_qdrant_cloud_url_and_api_key_from_env(self):
        kwargs, _ = self._build(
            set_env={
                "QDRANT_URL": "https://demo-cluster.qdrant.io:6333",
                "QDRANT_API_KEY": "placeholder-key-for-test",  # not a real credential
            },
            unset_env=(),  # replace the two keys above, don't pop them
        )
        self.assertEqual(kwargs["url"], "https://demo-cluster.qdrant.io:6333")
        self.assertEqual(kwargs["api_key"], "placeholder-key-for-test")

    def test_blank_or_whitespace_values_fall_back_to_local_no_key(self):
        kwargs, _ = self._build(
            set_env={"QDRANT_URL": "   ", "QDRANT_API_KEY": "  "},
            unset_env=(),
        )
        self.assertEqual(kwargs["url"], "http://localhost:6333")
        self.assertNotIn("api_key", kwargs)


if __name__ == "__main__":
    unittest.main()
