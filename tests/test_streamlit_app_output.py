"""Focused tests for streamlit_app's Inngest run-output handling.

All Inngest HTTP responses are mocked -- no real API calls, no Gemini usage.
They encode the response shapes observed from the local Inngest dev server:

* ``/v1/events/{id}/runs`` -> ``{"data": [{..., "output": {...}}]}``
* ``/v1/runs/{id}``        -> ``{"data": {..., "output": {...}}}``

and the two mismatch modes that made Streamlit print "The run produced no
output" even though the run had a valid ``answer`` payload:

* ``output`` arriving as a JSON-encoded *string* instead of an object, and
* ``status: "Completed"`` becoming visible before ``output`` is readable
  (the dev server caches the runs-list response for ~15s).
"""

import json
import os
import sys
import types
import unittest
from unittest import mock


def _install_streamlit_stub() -> None:
    """Replace streamlit with a no-op stub before importing streamlit_app.

    Importing streamlit_app executes its top-level page-building code; the
    stub keeps that harmless and makes the tests independent of a running
    Streamlit runtime.
    """
    if "streamlit" in sys.modules:
        return

    st_stub = types.ModuleType("streamlit")

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    st_stub.set_page_config = lambda **kwargs: None
    st_stub.cache_resource = lambda fn: fn
    for name in (
        "title",
        "divider",
        "success",
        "warning",
        "error",
        "write",
        "caption",
        "subheader",
    ):
        setattr(st_stub, name, lambda *args, **kwargs: None)
    st_stub.file_uploader = lambda *args, **kwargs: None
    st_stub.button = lambda *args, **kwargs: False
    st_stub.form = lambda *args, **kwargs: _Ctx()
    st_stub.text_input = lambda *args, **kwargs: ""
    st_stub.number_input = lambda *args, **kwargs: 5
    st_stub.form_submit_button = lambda *args, **kwargs: False
    st_stub.spinner = lambda *args, **kwargs: _Ctx()
    sys.modules["streamlit"] = st_stub


_install_streamlit_stub()

import streamlit_app as sa  # noqa: E402  (must import after the stub)

EVENT_ID = "evt_test_1"
RUN_ID = "run_test_1"
FINAL_OUTPUT = {
    "answer": "Snowball sampling reaches hard-to-reach populations.",
    "num_contexts": 5,
    "sources": ["Unit 3 Social Media.pdf"],
}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _runs_list(status, output, run_id=RUN_ID):
    return _FakeResponse(
        {"data": [{"run_id": run_id, "event_id": EVENT_ID, "status": status, "output": output}]}
    )


def _run_detail(status, output, run_id=RUN_ID):
    return _FakeResponse(
        {"data": {"run_id": run_id, "event_id": EVENT_ID, "status": status, "output": output}}
    )


def _routes(list_payload, detail_payload):
    """Build a requests.get replacement that routes on the Inngest URL shape.

    List endpoint: /v1/events/{id}/runs   Detail endpoint: /v1/runs/{id}
    """

    def fake_get(url, *args, **kwargs):
        if "/events/" in url:
            return list_payload
        return detail_payload

    return fake_get


class CoerceOutputTest(unittest.TestCase):
    """_coerce_output normalizes every shape the dev server can deliver."""

    def test_dict_passthrough(self):
        self.assertEqual(sa._coerce_output(FINAL_OUTPUT), FINAL_OUTPUT)

    def test_json_string_is_parsed(self):
        raw = json.dumps(FINAL_OUTPUT)
        self.assertEqual(sa._coerce_output(raw), FINAL_OUTPUT)

    def test_garbage_string_returns_empty_dict(self):
        self.assertEqual(sa._coerce_output("not json"), {})

    def test_none_and_empty_return_empty_dict(self):
        self.assertEqual(sa._coerce_output(None), {})
        self.assertEqual(sa._coerce_output(""), {})


class WaitForRunOutputTest(unittest.TestCase):
    def test_completed_run_with_dict_output_is_returned(self):
        """Baseline regression: the successful response shape works as-is."""
        with mock.patch.object(
            sa.requests,
            "get",
            side_effect=_routes(
                _runs_list("Completed", FINAL_OUTPUT),
                _run_detail("Completed", FINAL_OUTPUT),
            ),
        ):
            out = sa.wait_for_run_output(EVENT_ID, timeout_s=5, poll_interval_s=0.01)
        self.assertEqual(out, FINAL_OUTPUT)
        self.assertEqual(out["answer"], FINAL_OUTPUT["answer"])
        self.assertEqual(out["num_contexts"], 5)
        self.assertEqual(out["sources"], FINAL_OUTPUT["sources"])

    def test_completed_run_with_json_string_output_is_parsed(self):
        """The output field may arrive as a JSON-encoded string, not an object.

        Previously a str failed `isinstance(output, dict)` on BOTH endpoints
        and the function returned {} -> "The run produced no output."
        """
        encoded = json.dumps(FINAL_OUTPUT)
        with mock.patch.object(
            sa.requests,
            "get",
            side_effect=_routes(
                _runs_list("Completed", encoded),
                _run_detail("Completed", encoded),
            ),
        ):
            out = sa.wait_for_run_output(EVENT_ID, timeout_s=5, poll_interval_s=0.01)
        self.assertEqual(out, FINAL_OUTPUT)

    def test_empty_output_is_retried_until_it_lands(self):
        """Completed status can become visible before the output is readable.

        Previously the function returned {} on the first empty observation
        instead of polling; now it keeps polling and returns the real output.
        """
        state = {"list_calls": 0}

        def fake_get(url, *args, **kwargs):
            if "/events/" in url:
                state["list_calls"] += 1
                if state["list_calls"] < 3:
                    return _runs_list("Completed", None)
                return _runs_list("Completed", FINAL_OUTPUT)
            return _run_detail("Completed", None)

        with mock.patch.object(sa.requests, "get", side_effect=fake_get):
            out = sa.wait_for_run_output(EVENT_ID, timeout_s=10, poll_interval_s=0.01)

        self.assertEqual(out, FINAL_OUTPUT)
        self.assertGreaterEqual(state["list_calls"], 3, "expected the poller to keep polling")

    def test_output_string_lands_on_detail_endpoint_only(self):
        """List output null (stale cache) + detail output as JSON string."""
        with mock.patch.object(
            sa.requests,
            "get",
            side_effect=_routes(
                _runs_list("Completed", None),
                _run_detail("Completed", json.dumps(FINAL_OUTPUT)),
            ),
        ):
            out = sa.wait_for_run_output(EVENT_ID, timeout_s=5, poll_interval_s=0.01)
        self.assertEqual(out, FINAL_OUTPUT)

    def test_failed_run_raises_with_the_real_cause(self):
        """Failed runs must surface the underlying error message."""
        cause = {"message": "Gemini API error 429: quota exceeded"}
        with mock.patch.object(
            sa.requests,
            "get",
            side_effect=_routes(_runs_list("Failed", cause), _run_detail("Failed", cause)),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                sa.wait_for_run_output(EVENT_ID, timeout_s=5, poll_interval_s=0.01)
        self.assertIn("429", str(ctx.exception))

    def test_failed_run_with_string_output_still_surfaces_cause(self):
        cause = json.dumps({"message": "Qdrant connection refused"})
        with mock.patch.object(
            sa.requests,
            "get",
            side_effect=_routes(_runs_list("Failed", cause), _run_detail("Failed", cause)),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                sa.wait_for_run_output(EVENT_ID, timeout_s=5, poll_interval_s=0.01)
        self.assertIn("Qdrant connection refused", str(ctx.exception))

    def test_timeout_when_output_never_appears(self):
        """A permanently empty output must raise TimeoutError, not hang."""
        with mock.patch.object(
            sa.requests,
            "get",
            side_effect=_routes(
                _runs_list("Completed", None),
                _run_detail("Completed", None),
            ),
        ):
            with self.assertRaises(TimeoutError):
                sa.wait_for_run_output(EVENT_ID, timeout_s=0.3, poll_interval_s=0.01)

    def test_grace_window_expires_to_empty_dict(self):
        """After the grace window an empty-but-completed run returns {}.

        That is the only path that legitimately yields {} so the UI can
        honestly report "no output" (the warning is not shown after errors).
        """
        with mock.patch.object(sa, "OUTPUT_EMPTY_GRACE_S", 0.05):
            with mock.patch.object(
                sa.requests,
                "get",
                side_effect=_routes(
                    _runs_list("Completed", None),
                    _run_detail("Completed", None),
                ),
            ):
                out = sa.wait_for_run_output(EVENT_ID, timeout_s=10, poll_interval_s=0.01)
        self.assertEqual(out, {})


class InngestRestConfigTest(unittest.TestCase):
	"""Env-driven Inngest REST config for production vs. local dev.

	Local dev server: base URL defaults to http://127.0.0.1:8288/v1 and needs
	no auth. Inngest Cloud: INNGEST_API_BASE points at https://api.inngest.com/v1
	and every request must carry the environment's signing key as a bearer token.
	"""

	def test_api_base_defaults_to_dev_server(self):
	    with mock.patch.dict(os.environ, {}, clear=False):
	        os.environ.pop("INNGEST_API_BASE", None)
	        self.assertEqual(sa._inngest_api_base(), "http://127.0.0.1:8288/v1")

	def test_api_base_env_override_for_cloud(self):
	    with mock.patch.dict(
	        os.environ, {"INNGEST_API_BASE": "https://api.inngest.com/v1"}
	    ):
	        self.assertEqual(sa._inngest_api_base(), "https://api.inngest.com/v1")

	def test_headers_empty_without_signing_key(self):
	    """Dev server requests are unauthenticated."""
	    with mock.patch.dict(os.environ, {}, clear=False):
	        os.environ.pop("INNGEST_SIGNING_KEY", None)
	        self.assertEqual(sa._inngest_api_headers(), {})

	def test_headers_blank_signing_key_is_treated_as_unset(self):
	    with mock.patch.dict(os.environ, {"INNGEST_SIGNING_KEY": "   "}):
	        self.assertEqual(sa._inngest_api_headers(), {})

	def test_headers_bearer_with_signing_key(self):
	    with mock.patch.dict(os.environ, {"INNGEST_SIGNING_KEY": "signkey-test"}):
	        self.assertEqual(
	            sa._inngest_api_headers(),
	            {"Authorization": "Bearer signkey-test"},
	        )


class InngestClientConfigTest(unittest.TestCase):
	"""The Inngest client must get the event key from Streamlit secrets.

	Production regression: the first key of a pasted secrets.toml can carry a
	UTF-8 BOM, so a literal "INNGEST_EVENT_KEY" env mirror never existed and
	client.send() raised EventKeyUnspecifiedError. The client now receives the
	key explicitly via the SDK's `event_key` constructor argument.
	"""

	def _client_kwargs(self):
		with mock.patch.object(sa.inngest, "Inngest") as ctor:
			sa.get_inngest_client()
		return ctor.call_args.kwargs

	def test_event_key_is_passed_explicitly_from_streamlit_secrets(self):
		with mock.patch.object(sa.st, "secrets", {"INNGEST_EVENT_KEY": "fake-event-key"}, create=True):
			with mock.patch.dict(os.environ, {}, clear=False):
				os.environ.pop("INNGEST_EVENT_KEY", None)
				kwargs = self._client_kwargs()
		self.assertEqual(kwargs.get("event_key"), "fake-event-key")
		self.assertEqual(kwargs.get("app_id"), "rag_app")
		# Mode stays environment-driven (INNGEST_DEV); don't hardcode it.
		self.assertNotIn("is_production", kwargs)

	def test_event_key_survives_bom_prefixed_secret_name(self):
		"""The exact production failure: the key name starts with a UTF-8 BOM."""
		secrets = {"\ufeffINNGEST_EVENT_KEY": "fake-event-key", "INNGEST_ENV": "production"}
		with mock.patch.object(sa.st, "secrets", secrets, create=True):
			with mock.patch.dict(os.environ, {}, clear=False):
				os.environ.pop("INNGEST_EVENT_KEY", None)
				kwargs = self._client_kwargs()
		self.assertEqual(kwargs.get("event_key"), "fake-event-key")

	def test_event_key_falls_back_to_env_for_local_dev(self):
		with mock.patch.object(sa.st, "secrets", {}, create=True):
			with mock.patch.dict(os.environ, {"INNGEST_EVENT_KEY": "env-key"}):
				kwargs = self._client_kwargs()
		self.assertEqual(kwargs.get("event_key"), "env-key")

	def test_no_event_key_anywhere_keeps_dev_server_default(self):
		"""Local dev: no key -> None -> SDK uses the dev server's built-in key."""
		with mock.patch.object(sa.st, "secrets", {}, create=True):
			with mock.patch.dict(os.environ, {}, clear=False):
				os.environ.pop("INNGEST_EVENT_KEY", None)
				kwargs = self._client_kwargs()
		self.assertIsNone(kwargs.get("event_key"))

	def test_secret_mirror_normalizes_bom_keys(self):
		"""Mirrored env names must be clean so os.getenv() lookups work."""
		secrets = {
			"\ufeffINNGEST_EVENT_KEY": "fake-event-key",
			"INNGEST_API_BASE": "https://api.inngest.com/v1",
		}
		with mock.patch.object(sa.st, "secrets", secrets, create=True):
			with mock.patch.dict(os.environ, {}, clear=False):
				for name in ("INNGEST_EVENT_KEY", "INNGEST_API_BASE"):
					os.environ.pop(name, None)
				sa._mirror_streamlit_secrets()
				self.assertEqual(os.environ.get("INNGEST_EVENT_KEY"), "fake-event-key")
				self.assertEqual(os.environ.get("INNGEST_API_BASE"), "https://api.inngest.com/v1")
				self.assertNotIn("\ufeffINNGEST_EVENT_KEY", os.environ)

	def test_installed_sdk_supports_event_key_argument(self):
		"""Guard against SDK API drift: `event_key` must remain a constructor arg."""
		import inspect

		params = inspect.signature(sa.inngest.Inngest.__init__).parameters
		self.assertIn("event_key", params)


if __name__ == "__main__":
	unittest.main()
