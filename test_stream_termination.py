"""Regression tests for SSE stream termination and summary rollup exclusion."""
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


def _load_context_bench():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "context_bench_under_test", os.path.join(here, "context-bench.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["context_bench_under_test"] = module
    spec.loader.exec_module(module)
    return module


cb = _load_context_bench()


def _delta(**kwargs):
    return {"choices": [{"delta": kwargs}]}


def _sse_lines(frames, include_done=True):
    lines = ["data: " + json.dumps(frame) for frame in frames]
    if include_done:
        lines.append("data: [DONE]")
    return lines


class _FakeResponse:
    def __init__(self, lines, status_code=200):
        self._lines = list(lines)
        self.status_code = status_code

    def iter_lines(self):
        return iter(self._lines)

    def read(self):
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _ScriptedClientFactory:
    def __init__(self, stream_sequence):
        self._responses = list(stream_sequence)
        self._index = 0

    def __call__(self, *args, **kwargs):
        factory = self

        class _Client:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def stream(self, method, url, json=None):
                resp = factory._responses[factory._index]
                factory._index += 1
                return resp

        return _Client()


class TestSSEStreamTermination(unittest.TestCase):
    def _run_with_lines(self, lines):
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )
        with mock.patch.object(cb, "httpx", fake_httpx):
            return cb.run_single_test(
                [{"role": "user", "content": "summarize"}], "4K", "test-model"
            )

    def test_partial_stream_without_done_or_finish_reason_fails(self):
        lines = _sse_lines(
            [_delta(content="partial "), _delta(content="output")],
            include_done=False,
        )
        result = self._run_with_lines(lines)

        self.assertTrue(result["truncated"])
        self.assertIsNone(result["finish_reason"])
        self.assertIn("error", result)
        self.assertTrue(result["error"])
        self.assertEqual(result["content_chars"], len("partial output"))
        self.assertEqual(result["output_tokens"], 2)

    def test_empty_http_200_stream_without_terminal_signal_fails(self):
        result = self._run_with_lines([])

        self.assertTrue(result["truncated"])
        self.assertIsNone(result["finish_reason"])
        self.assertIn("error", result)
        self.assertTrue(result["error"])

    def test_normal_stream_with_finish_reason_and_done_succeeds(self):
        lines = _sse_lines(
            [
                _delta(content="hello "),
                _delta(content="world"),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"completion_tokens": 2, "prompt_tokens": 64}},
            ],
            include_done=True,
        )
        result = self._run_with_lines(lines)

        self.assertNotIn("error", result)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(result["token_source"], "usage")

    def test_stream_with_finish_reason_without_done_succeeds(self):
        lines = _sse_lines(
            [
                _delta(content="complete"),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ],
            include_done=False,
        )
        result = self._run_with_lines(lines)

        self.assertNotIn("error", result)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["finish_reason"], "stop")

    def test_stream_with_done_without_finish_reason_succeeds(self):
        lines = _sse_lines(
            [_delta(content="a"), _delta(content="b")],
            include_done=True,
        )
        result = self._run_with_lines(lines)

        self.assertNotIn("error", result)
        self.assertFalse(result["truncated"])
        self.assertIsNone(result["finish_reason"])
        self.assertEqual(result["output_tokens"], 2)

    def test_summary_rollups_exclude_incomplete_streams(self):
        partial_resp = _FakeResponse(
            _sse_lines([_delta(content="cut off mid-sentence")], include_done=False)
        )
        complete_resp = _FakeResponse(
            _sse_lines(
                [
                    _delta(content="done"),
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"completion_tokens": 4, "prompt_tokens": 100}},
                ],
                include_done=True,
            )
        )
        all_partial_resp_1 = _FakeResponse(
            _sse_lines([_delta(content="dropped")], include_done=False)
        )
        all_partial_resp_2 = _FakeResponse(
            _sse_lines([_delta(content="dropped again")], include_done=False)
        )

        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory(
                [partial_resp, complete_resp, all_partial_resp_1, all_partial_resp_2]
            ),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.json"
            buf = io.StringIO()
            with (
                mock.patch.object(cb, "httpx", fake_httpx),
                mock.patch.object(cb, "get_pageout_count", return_value=0),
                mock.patch.object(cb, "get_memory_pressure", return_value=80),
                redirect_stdout(buf),
            ):
                cb.run_benchmark(
                    "http://127.0.0.1:8090/v1",
                    iterations=2,
                    context_sizes=[4096, 8192],
                    model_name="test-model",
                    results_file=results_path,
                )

            payload = json.loads(results_path.read_text())

        # 4K had 1 incomplete stream and 1 valid stream -> only 1 run counted, 1 failure
        summary_4k = payload["summary"]["4K"]
        self.assertEqual(summary_4k["runs"], 1)
        self.assertEqual(summary_4k["failures"], 1)
        self.assertIsNotNone(summary_4k["avg_ttft_s"])

        # 8K had 2 incomplete streams -> 0 runs counted, 2 failures, None averages
        summary_8k = payload["summary"]["8K"]
        self.assertEqual(summary_8k["runs"], 0)
        self.assertEqual(summary_8k["failures"], 2)
        self.assertIsNone(summary_8k["avg_ttft_s"])
        self.assertIsNone(summary_8k["avg_tps"])
        self.assertIsNone(summary_8k["avg_total_s"])
        self.assertIsNone(summary_8k["avg_pageouts"])

        # Raw records preserve the error and truncated flags on failed rows
        raw_4k_failed = payload["raw"][0]
        raw_4k_ok = payload["raw"][1]
        self.assertTrue(raw_4k_failed["truncated"])
        self.assertTrue(raw_4k_failed.get("error"))
        self.assertFalse(raw_4k_ok["truncated"])
        self.assertNotIn("error", raw_4k_ok)

        # Console output reports FAILED for incomplete rows and FAIL for 8K summary
        output_text = buf.getvalue()
        self.assertIn("FAILED: Incomplete SSE stream", output_text)
        self.assertIn("RECOMMENDATION: 4K context window", output_text)


if __name__ == "__main__":
    unittest.main()
