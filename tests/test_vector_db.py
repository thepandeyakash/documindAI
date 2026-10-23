"""Regression tests for source_id normalization and Qdrant round-trip.

Covers the bug where a PDF filename containing a double space
('Unit 3  Social Media_sk.pdf') was stored verbatim, so a query typed with
normal spacing matched nothing and reported "No ingested chunks found".
No Gemini calls: vectors are synthetic.
"""
import os
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

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

            def get_collection(self, name):
                # Report the source index as already present: this test only
                # pins the constructor's connection kwargs.
                return SimpleNamespace(payload_schema={"source": object()})

            def create_payload_index(self, **kwargs):
                raise AssertionError("index exists; it must not be recreated")

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


class SourcePayloadIndexTest(unittest.TestCase):
    """The 'source' keyword index must exist before filtered count/search.

    Cloud Qdrant rejects filtered requests on unindexed fields
    ("Index required, not found for 'source'"), so QdrantStorage ensures the
    index exists for collections it creates and for pre-existing ones, without
    deleting or recreating the collection or its points.
    """

    def _build(self, exists: bool, schema=None, race: bool = False):
        calls = {"create_payload_index": [], "create_collection": 0}
        state = {"schema": dict(schema or {})}

        class _Client:
            def __init__(self, **kwargs):
                pass

            def collection_exists(self, name):
                return exists

            def create_collection(self, **kwargs):
                calls["create_collection"] += 1

            def get_collection(self, name):
                return SimpleNamespace(payload_schema=dict(state["schema"]))

            def create_payload_index(self, **kwargs):
                calls["create_payload_index"].append(kwargs)
                if race:
                    # Another writer wins the race between check and create.
                    state["schema"]["source"] = object()
                    raise RuntimeError("already exists")

        with mock.patch("vector_db.QdrantClient", _Client):
            store = QdrantStorage()
        return store, calls

    def test_existing_collection_without_index_gets_keyword_index(self):
        store, calls = self._build(exists=True, schema={})
        self.assertEqual(store.collection, "docs_gemini_768")
        self.assertEqual(len(calls["create_payload_index"]), 1)
        kwargs = calls["create_payload_index"][0]
        self.assertEqual(kwargs["collection_name"], "docs_gemini_768")
        self.assertEqual(kwargs["field_name"], "source")
        self.assertEqual(kwargs["field_schema"], PayloadSchemaType.KEYWORD)
        self.assertIs(kwargs["wait"], True)
        # Existing collection: nothing is recreated.
        self.assertEqual(calls["create_collection"], 0)

    def test_existing_index_is_not_rebuilt(self):
        _, calls = self._build(exists=True, schema={"source": object()})
        self.assertEqual(calls["create_payload_index"], [])

    def test_new_collection_also_gets_the_index(self):
        _, calls = self._build(exists=False, schema={})
        self.assertEqual(calls["create_collection"], 1)
        self.assertEqual(len(calls["create_payload_index"]), 1)

    def test_concurrent_creation_race_is_tolerated(self):
        """An 'already exists' error is fine once the index is visible."""
        self._build(exists=True, schema={}, race=True)  # must not raise


if __name__ == "__main__":
    unittest.main()
