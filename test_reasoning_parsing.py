"""Prove the reasoning-aware SSE parsing against a synthetic stream.

Verifies the exact defect this change fixes: a stream whose tokens arrive on
`reasoning`/`reasoning_content` must not read as a near-empty, slow-TTFT run.
No network -- httpx.Client is stubbed.
"""
import importlib.util
import json
import os
import sys
import types
import unittest
from unittest import mock

# Load the hyphenated script as a module, relative to THIS file, so the test works
# from any clone location.
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "ctxbench", os.path.join(_HERE, "context-bench.py")
)
cb = importlib.util.module_from_spec(_spec)
sys.modules["ctxbench"] = cb
_spec.loader.exec_module(cb)


def sse(frames):
    for f in frames:
        yield "data: " + json.dumps(f)
    yield "data: [DONE]"


class FakeResp:
    status_code = 200

    def __init__(self, frames):
        self._frames = frames

    def iter_lines(self):
        return sse(self._frames)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeClient:
    frames = []
    captured = {}

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def stream(self, method, url, json=None):
        FakeClient.captured.clear()
        FakeClient.captured.update(json or {})
        return FakeResp(FakeClient.frames)


def d(**kw):
    return {"choices": [{"delta": kw}]}


class TestReasoningParsing(unittest.TestCase):
    def setUp(self):
        self._fake_httpx = types.SimpleNamespace(
            Client=FakeClient,
            Timeout=lambda *a, **k: None,
            TimeoutException=Exception,
        )
        self._patch = mock.patch.object(cb, "httpx", self._fake_httpx)
        self._patch.start()
        cb.THINK_KWARGS = {}

    def tearDown(self):
        cb.THINK_KWARGS = {}
        self._patch.stop()

    def test_reasoning_and_content_tokens(self):
        FakeClient.frames = [
            d(reasoning="Let me "),
            d(reasoning="think about "),
            d(reasoning="this. "),
            d(content="Buy"),
            d(content=" AAPL"),
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"completion_tokens": 5, "prompt_tokens": 4000}},
        ]
        r = cb.run_single_test([{"role": "user", "content": "x"}], "4K", "m")
        self.assertEqual(r["reasoning_chars"], 25)
        self.assertEqual(r["content_chars"], 8)
        self.assertIs(r["reasoned"], True)
        self.assertEqual(r["output_tokens"], 5)
        self.assertEqual(r["token_source"], "usage")
        self.assertEqual(r.get("reasoning_share"), round(25 / 33, 3))
        self.assertEqual(r["finish_reason"], "stop")
        self.assertIs(r["truncated"], False)
        self.assertGreaterEqual(r.get("ttft_content_s", 0), r["ttft_s"])
        self.assertNotEqual(r["output_tokens"], 2)

    def test_reasoning_content_alias(self):
        FakeClient.frames = [
            d(reasoning_content="hmm "),
            d(content="ok"),
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        r2 = cb.run_single_test([{"role": "user", "content": "x"}], "4K", "m")
        self.assertEqual(r2["reasoning_chars"], 4)
        self.assertTrue(r2["reasoned"])

    def test_truncation_surfaced(self):
        FakeClient.frames = [
            d(content="a"),
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
        ]
        r3 = cb.run_single_test([{"role": "user", "content": "x"}], "4K", "m")
        self.assertIs(r3["truncated"], True)

    def test_fallback_to_sse_chunks_when_usage_absent(self):
        FakeClient.frames = [d(content="a"), d(content="b"), d(content="c")]
        r4 = cb.run_single_test([{"role": "user", "content": "x"}], "4K", "m")
        self.assertEqual(r4["output_tokens"], 3)
        self.assertEqual(r4["token_source"], "sse_chunks")
        self.assertIs(r4["reasoned"], False)

    def test_think_kwargs_forwarded_and_omitted_when_unset(self):
        cb.THINK_KWARGS = {"enable_thinking": True}
        FakeClient.frames = [d(content="a")]
        cb.run_single_test([{"role": "user", "content": "x"}], "4K", "m")
        self.assertEqual(
            FakeClient.captured.get("chat_template_kwargs"),
            {"enable_thinking": True},
        )
        self.assertEqual(
            FakeClient.captured.get("stream_options"), {"include_usage": True}
        )

        cb.THINK_KWARGS = {}
        FakeClient.frames = [d(content="a")]
        cb.run_single_test([{"role": "user", "content": "x"}], "4K", "m")
        self.assertNotIn("chat_template_kwargs", FakeClient.captured)


if __name__ == "__main__":
    unittest.main()
