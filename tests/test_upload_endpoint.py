"""Tests for the FastAPI /upload endpoint (production ingest hand-off).

Streamlit Community Cloud and the Railway backend do not share a filesystem, so
in production the Streamlit app uploads PDFs to the backend (POST /upload) and
the ingest function reads the returned backend-local path. These tests pin the
endpoint's contract: store a PDF under UPLOAD_DIR, sanitize the stored name, and
reject anything that is not a non-empty PDF under the size cap.

``main`` is imported with INNGEST_DEV=1 so the Inngest client starts in dev mode
(no signing key required); no Inngest/Gemini/Qdrant network calls are made.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Dev mode must be on BEFORE main is imported, or the Inngest serve setup in
# production mode demands a signing key at startup.
os.environ.setdefault("INNGEST_DEV", "1")

from fastapi.testclient import TestClient  # noqa: E402

import main as main_module  # noqa: E402

PDF_BYTES = b"%PDF-1.7 fake content for tests"


class UploadEndpointTest(unittest.TestCase):
	def setUp(self):
		self._tmp = tempfile.TemporaryDirectory()
		self.addCleanup(self._tmp.cleanup)
		patcher = mock.patch.object(main_module, "UPLOAD_DIR", Path(self._tmp.name))
		patcher.start()
		self.addCleanup(patcher.stop)
		self.client = TestClient(main_module.app)

	def _upload(self, filename, content=PDF_BYTES):
		return self.client.post(
			"/upload",
			files={"file": (filename, content, "application/pdf")},
		)

	def test_pdf_is_stored_and_paths_returned(self):
		"""Happy path: the file lands in UPLOAD_DIR and both ids come back."""
		resp = self._upload("report.pdf")
		self.assertEqual(resp.status_code, 200)
		body = resp.json()
		self.assertEqual(body["source_id"], "report.pdf")
		stored = Path(body["pdf_path"])
		self.assertTrue(stored.is_file())
		self.assertEqual(stored.read_bytes(), PDF_BYTES)
		self.assertEqual(stored.parent, Path(self._tmp.name).resolve())

	def test_directory_components_are_stripped(self):
		"""Client-supplied paths must not escape UPLOAD_DIR (path traversal)."""
		resp = self._upload("../../evil.pdf")
		self.assertEqual(resp.status_code, 200)
		self.assertEqual(resp.json()["source_id"], "evil.pdf")
		stored = Path(resp.json()["pdf_path"])
		self.assertEqual(stored.parent, Path(self._tmp.name).resolve())

	def test_non_pdf_is_rejected(self):
		resp = self._upload("notes.txt", b"plain text")
		self.assertEqual(resp.status_code, 400)

	def test_empty_file_is_rejected(self):
		resp = self._upload("empty.pdf", b"")
		self.assertEqual(resp.status_code, 400)

	def test_oversize_file_is_rejected(self):
		with mock.patch.object(main_module, "MAX_UPLOAD_BYTES", 8):
			resp = self._upload("big.pdf", b"0123456789")
		self.assertEqual(resp.status_code, 413)
		self.assertFalse((Path(self._tmp.name) / "big.pdf").exists())


if __name__ == "__main__":
	unittest.main()
