"""Regression: Inngest functions must use the backend's Qdrant configuration.

Production incident: the deployed build still had QdrantStorage defaulting to
http://localhost:6333 and ignoring QDRANT_URL/QDRANT_API_KEY, so ingestion and
query runs (which execute inside the deployed web service) tried to reach
localhost instead of Qdrant Cloud.

These tests execute the real rag_ingest_pdf / rag_query_pdf_ai handler bodies
with an inline fake step runner, and assert the Qdrant client is constructed at
run time from the process environment: the cloud URL + API key when configured,
and the localhost default (no key) for local development. No network calls, no
real credentials, no Gemini usage.
"""

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest import mock

# Dev mode must be on before main is imported, or Inngest's serve setup demands
# a production signing key at startup.
os.environ.setdefault("INNGEST_DEV", "1")

import main as main_module  # noqa: E402
import vector_db  # noqa: E402

CLOUD_URL = "https://demo-cluster.cloud.qdrant.io:6333"
CLOUD_KEY = "placeholder-cloud-key-for-test"  # not a real credential


class _FakeStep:
    """Runs each step inline, like a successful first attempt (no memoization)."""

    async def run(self, name, fn, output_type=None):
        return fn()


class _FakeContext:
    def __init__(self, data):
        self.event = SimpleNamespace(data=data)
        self.step = _FakeStep()


class _RecordingQdrantClient:
    """Stands in for QdrantClient; records constructor kwargs per instance."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).instances.append(self)

    def collection_exists(self, name):
        return True

    def get_collection(self, name):
        # Index already present so these tests stay focused on connection config.
        return SimpleNamespace(payload_schema={"source": object()})

    def create_payload_index(self, **kwargs):
        raise AssertionError("index exists; it must not be recreated")

    def upsert(self, *args, **kwargs):
        return None

    def count(self, *args, **kwargs):
        return SimpleNamespace(count=3)

    def search(self, *args, **kwargs):
        return []


class InngestFunctionsQdrantConfigTest(unittest.TestCase):
    """The function bodies (not import time) must resolve Qdrant from env."""

    def setUp(self):
        _RecordingQdrantClient.instances.clear()
        patcher = mock.patch.object(vector_db, "QdrantClient", _RecordingQdrantClient)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run_ingest(self):
        ctx = _FakeContext({"pdf_path": "unused.pdf", "source_id": "report.pdf"})
        with mock.patch.object(
            main_module, "load_and_chunk_pdf", lambda path: ["chunk one"]
        ):
            with mock.patch.object(
                main_module,
                "embed_texts",
                lambda texts, **kwargs: [[0.01] * 768 for _ in texts],
            ):
                # The SDK stores the user handler on the Function object.
                return asyncio.run(main_module.rag_ingest_pdf._handler(ctx))

    def _run_query(self):
        ctx = _FakeContext(
            {"question": "what is it about?", "source_id": "report.pdf", "top_k": 3}
        )
        with mock.patch.object(
            main_module,
            "embed_texts",
            lambda texts, **kwargs: [[0.01] * 768 for _ in texts],
        ):
            return asyncio.run(main_module.rag_query_pdf_ai._handler(ctx))

    def test_ingest_uses_qdrant_cloud_env(self):
        with mock.patch.dict(
            os.environ, {"QDRANT_URL": CLOUD_URL, "QDRANT_API_KEY": CLOUD_KEY}
        ):
            result = self._run_ingest()
        self.assertEqual(len(_RecordingQdrantClient.instances), 1)
        kwargs = _RecordingQdrantClient.instances[0].kwargs
        self.assertEqual(kwargs["url"], CLOUD_URL)
        self.assertEqual(kwargs["api_key"], CLOUD_KEY)
        self.assertEqual(result, {"ingested": 3})

    def test_query_uses_qdrant_cloud_env(self):
        with mock.patch.dict(
            os.environ, {"QDRANT_URL": CLOUD_URL, "QDRANT_API_KEY": CLOUD_KEY}
        ):
            result = self._run_query()
        self.assertEqual(len(_RecordingQdrantClient.instances), 1)
        kwargs = _RecordingQdrantClient.instances[0].kwargs
        self.assertEqual(kwargs["url"], CLOUD_URL)
        self.assertEqual(kwargs["api_key"], CLOUD_KEY)
        # No chunks -> clear empty-result answer, and no LLM call is attempted.
        self.assertIn("No ingested chunks found", result["answer"])

    def test_local_dev_still_defaults_to_localhost_without_key(self):
        with mock.patch.dict(os.environ, {"QDRANT_URL": "", "QDRANT_API_KEY": ""}):
            result = self._run_ingest()
        kwargs = _RecordingQdrantClient.instances[0].kwargs
        self.assertEqual(kwargs["url"], "http://localhost:6333")
        self.assertNotIn("api_key", kwargs)
        self.assertEqual(result, {"ingested": 3})


if __name__ == "__main__":
    unittest.main()
