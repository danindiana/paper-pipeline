import json
import unittest
from threading import Event
from unittest.mock import patch

from paper_pipeline.errors import EmptyGenerationError
from paper_pipeline.ollama import OllamaClient, OllamaVRAM


class FakeStreamResponse:
    """Minimal stand-in for requests.Response used as a streaming context manager."""

    def __init__(self, lines, ok=True, status_code=200):
        self._lines = lines
        self.ok = ok
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        for line in self._lines:
            yield json.dumps(line).encode()


class FakeEvictResponse:
    """Stand-in for the non-streaming eviction POST (keep_alive=0)."""
    status_code = 200


def _done_line(**overrides):
    base = {
        "done": True,
        "done_reason": "stop",
        "eval_count": 10,
        "eval_duration": 1_000_000,
        "prompt_eval_count": 5,
        "prompt_eval_duration": 500_000,
        "total_duration": 2_000_000,
    }
    base.update(overrides)
    return base


def _is_evict_payload(payload):
    """True for the bare {"model", "keep_alive": 0} eviction payload.

    Distinct from a Mock call's kwargs dict — callers pass the request's
    `json=` payload directly, not a wrapper around it.
    """
    return bool(payload) and payload.get("keep_alive") == 0


class EmptyGenerationRetryTests(unittest.TestCase):
    def setUp(self):
        self.vram = OllamaVRAM(base_url="http://fake-ollama:11434")
        self.shutdown = Event()
        self.client = OllamaClient(self.vram, self.shutdown)

    def _mock_ps_empty(self, mock_get):
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = {"models": []}

    @patch("paper_pipeline.ollama.requests.get")
    @patch("paper_pipeline.ollama.requests.post")
    def test_empty_response_forces_reload_then_succeeds(self, mock_post, mock_get):
        self._mock_ps_empty(mock_get)

        stream_responses = iter([
            # First attempt: model exhausts context, no "response" chunk arrives.
            FakeStreamResponse([_done_line(done_reason="length", eval_count=4000)]),
            # Second attempt, after force_reload evicts the model: succeeds.
            FakeStreamResponse([{"response": "ok"}, _done_line()]),
        ])

        def fake_post(url, json=None, timeout=None, stream=None):
            if _is_evict_payload(json):
                return FakeEvictResponse()
            return next(stream_responses)

        mock_post.side_effect = fake_post

        text = self.client.generate("fake-model", "prompt")

        self.assertEqual(text, "ok")
        evict_calls = [c for c in mock_post.call_args_list if _is_evict_payload(c.kwargs.get("json"))]
        self.assertEqual(len(evict_calls), 1, "expected exactly one force_reload eviction between attempts")

    @patch("paper_pipeline.ollama.requests.get")
    @patch("paper_pipeline.ollama.requests.post")
    def test_empty_response_twice_raises_typed_error_with_diagnostics(self, mock_post, mock_get):
        self._mock_ps_empty(mock_get)

        def fake_post(url, json=None, timeout=None, stream=None):
            if _is_evict_payload(json):
                return FakeEvictResponse()
            return FakeStreamResponse([_done_line(done_reason="length", eval_count=4000)])

        mock_post.side_effect = fake_post

        with self.assertRaises(EmptyGenerationError) as ctx:
            self.client.generate("fake-model", "prompt")

        message = str(ctx.exception)
        self.assertIn("done_reason=length", message)
        self.assertIn("eval_count=4000", message)

    @patch("paper_pipeline.ollama.requests.get")
    @patch("paper_pipeline.ollama.requests.post")
    def test_thinking_only_response_is_not_treated_as_the_answer(self, mock_post, mock_get):
        self._mock_ps_empty(mock_get)

        def fake_post(url, json=None, timeout=None, stream=None):
            if _is_evict_payload(json):
                return FakeEvictResponse()
            return FakeStreamResponse([
                {"thinking": "reasoning " * 50},
                _done_line(done_reason="length"),
            ])

        mock_post.side_effect = fake_post

        with self.assertRaises(EmptyGenerationError) as ctx:
            self.client.generate("fake-model", "prompt")

        self.assertIn("thinking_chars=", str(ctx.exception))
        self.assertNotIn("reasoning", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
