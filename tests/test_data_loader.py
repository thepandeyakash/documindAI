"""Unit tests for the Gemini embedding helpers in data_loader.py.

These tests mock the google-genai SDK client, so no network access or
GEMINI_API_KEY is required.
Run with:  uv run python -m unittest discover -s tests -v
"""

import os
import unittest
from unittest import mock

import data_loader


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeEmbedResponse:
    def __init__(self, vectors):
        self.embeddings = [_FakeEmbedding(v) for v in vectors]


class _FakeAPIError(Exception):
    """Mimics google.genai.errors.APIError (.code / .message attributes)."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def _fake_client(response=None, error=None):
    client = mock.Mock()
    if error is not None:
        client.models.embed_content.side_effect = error
    else:
        client.models.embed_content.return_value = response
    return client


class EmbedTextsTests(unittest.TestCase):
    def test_empty_input_returns_empty_list_without_key(self):
        self.assertEqual(data_loader.embed_texts([]), [])

    def test_missing_key_raises_helpful_error(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY is not set"):
                data_loader.embed_texts(["hello"])

    def test_happy_path_returns_768_dim_unit_vectors(self):
        payload = [[0.3] * 768, [0.4] * 768]
        client = _fake_client(response=_FakeEmbedResponse(payload))
        with mock.patch.object(data_loader.genai, "Client", return_value=client):
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                out = data_loader.embed_texts(["chunk one", "chunk two"])

        self.assertEqual(len(out), 2)
        self.assertTrue(all(len(v) == 768 for v in out))
        # Vectors must be L2-normalized (0.3 * sqrt(768) == norm before normalization).
        norm = sum(x * x for x in out[0]) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

        _, kwargs = client.models.embed_content.call_args
        self.assertEqual(kwargs["model"], "gemini-embedding-001")
        self.assertEqual(kwargs["contents"], ["chunk one", "chunk two"])
        self.assertEqual(kwargs["config"].output_dimensionality, 768)
        self.assertEqual(kwargs["config"].task_type, "RETRIEVAL_DOCUMENT")

    def test_query_task_type_is_forwarded(self):
        client = _fake_client(response=_FakeEmbedResponse([[0.5] * 768]))
        with mock.patch.object(data_loader.genai, "Client", return_value=client):
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                data_loader.embed_texts(["q"], task_type=data_loader.TASK_TYPE_QUERY)

        _, kwargs = client.models.embed_content.call_args
        self.assertEqual(kwargs["config"].task_type, "RETRIEVAL_QUERY")

    def test_large_input_is_batched_at_100(self):
        client = mock.Mock()

        def _embed(*, model, contents, config):
            return _FakeEmbedResponse([[0.1] * 768] * len(contents))

        client.models.embed_content.side_effect = _embed
        with mock.patch.object(data_loader.genai, "Client", return_value=client):
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                out = data_loader.embed_texts([f"t{i}" for i in range(250)])

        self.assertEqual(len(out), 250)
        self.assertEqual(client.models.embed_content.call_count, 3)
        sizes = [
            len(c.kwargs["contents"]) for c in client.models.embed_content.call_args_list
        ]
        self.assertEqual(sizes, [100, 100, 50])

    def test_wrong_dimension_raises(self):
        client = _fake_client(response=_FakeEmbedResponse([[0.0] * 3072]))
        with mock.patch.object(data_loader.genai, "Client", return_value=client):
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                with self.assertRaisesRegex(RuntimeError, "768-dimensional"):
                    data_loader.embed_texts(["hello"])

    def test_count_mismatch_raises(self):
        client = _fake_client(response=_FakeEmbedResponse([[0.0] * 768]))
        with mock.patch.object(data_loader.genai, "Client", return_value=client):
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                with self.assertRaisesRegex(RuntimeError, "Expected 2 embeddings"):
                    data_loader.embed_texts(["a", "b"])

    def test_empty_embedding_entry_raises(self):
        client = _fake_client(response=_FakeEmbedResponse([[]]))
        with mock.patch.object(data_loader.genai, "Client", return_value=client):
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                with self.assertRaisesRegex(RuntimeError, "Unexpected Gemini embedding"):
                    data_loader.embed_texts(["hello"])

    def test_zero_vector_raises(self):
        client = _fake_client(response=_FakeEmbedResponse([[0.0] * 768]))
        with mock.patch.object(data_loader.genai, "Client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "all-zero"):
                with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                    data_loader.embed_texts(["hello"])

    def test_http_401_mentions_key_problem(self):
        client = _fake_client(error=_FakeAPIError(401, "API key not valid"))
        with mock.patch.object(data_loader.genai, "Client", return_value=client):
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "bad"}):
                with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                    data_loader.embed_texts(["hello"])

    def test_http_429_surfaces_rate_limit(self):
        client = _fake_client(error=_FakeAPIError(429, "RESOURCE_EXHAUSTED"))
        with mock.patch.object(data_loader.genai, "Client", return_value=client):
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                with self.assertRaisesRegex(RuntimeError, "rate-limited"):
                    data_loader.embed_texts(["hello"])

    def test_http_404_mentions_model_retirement(self):
        client = _fake_client(error=_FakeAPIError(404, "models/gemini-embedding-001 is not found"))
        with mock.patch.object(data_loader.genai, "Client", return_value=client):
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                with self.assertRaisesRegex(RuntimeError, "retired"):
                    data_loader.embed_texts(["hello"])

    def test_generic_error_includes_detail(self):
        client = _fake_client(error=_FakeAPIError(500, "backend boom"))
        with mock.patch.object(data_loader.genai, "Client", return_value=client):
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                with self.assertRaisesRegex(RuntimeError, "backend boom"):
                    data_loader.embed_texts(["hello"])


if __name__ == "__main__":
    unittest.main()
