from __future__ import annotations

import unittest

from elagent.config import Config
from elagent.llm import LLMClient, LLMResponse


class PayloadTest(unittest.TestCase):
    """llama.cpp owns generation parameters; ELAgent must not override them."""

    def client(self, **overrides) -> LLMClient:
        cfg = Config(**{"stream": False, **overrides})
        client = LLMClient(cfg)
        client._model = "test-model"  # skip GET /v1/models
        self.addCleanup(client.close)
        return client

    @staticmethod
    def capture(client: LLMClient) -> dict:
        seen: dict = {}

        def fake(payload: dict) -> LLMResponse:
            seen.update(payload)
            return LLMResponse(text="ok", prompt_tokens=3, completion_tokens=1, finish_reason="stop")

        client._chat_once = fake  # type: ignore[assignment]
        return seen

    def test_default_payload_has_no_generation_params(self):
        client = self.client()
        seen = self.capture(client)
        client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(seen["model"], "test-model")
        self.assertEqual(seen["stream"], False)
        for key in ("temperature", "top_p", "max_tokens"):
            self.assertNotIn(key, seen)

    def test_caller_overrides_are_ignored_by_default(self):
        # The judge used to force temperature=0.0 / judge_max_tokens; that must
        # no longer reach the server.
        client = self.client()
        seen = self.capture(client)
        client.chat([{"role": "user", "content": "judge"}], temperature=0.0, max_tokens=700)
        for key in ("temperature", "top_p", "max_tokens"):
            self.assertNotIn(key, seen)

    def test_opt_in_sends_configured_params(self):
        client = self.client(send_generation_params=True, temperature=0.7, top_p=0.8, max_tokens=123)
        seen = self.capture(client)
        client.chat([{"role": "user", "content": "hi"}], temperature=0.1, max_tokens=50)
        self.assertEqual(seen["temperature"], 0.1)
        self.assertEqual(seen["top_p"], 0.8)
        self.assertEqual(seen["max_tokens"], 50)

    def test_empty_response_does_not_mutate_budget(self):
        client = self.client(max_retries=0, send_generation_params=True, max_tokens=10)
        payloads: list[dict] = []

        def fake(payload: dict) -> LLMResponse:
            payloads.append(dict(payload))
            return LLMResponse(text="", reasoning="only reasoning", finish_reason="length")

        client._chat_once = fake  # type: ignore[assignment]
        with self.assertRaises(Exception):
            client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["max_tokens"], 10)

    def test_truncated_empty_stream_reports_local_cap(self):
        client = self.client(max_retries=0, stream=True)

        def fake_stream(payload, on_delta, on_reasoning):
            return LLMResponse(text="", reasoning="looping reasoning", truncated=True)

        client._chat_stream = fake_stream  # type: ignore[assignment]
        with self.assertRaises(Exception) as ctx:
            client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("流上限", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
