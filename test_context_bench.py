"""Comprehensive tests for CB01-CB04 fixes and the synthetic mock-SSE demo."""
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
        "context_bench_cb_tests", os.path.join(here, "context-bench.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["context_bench_cb_tests"] = module
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
    def __init__(self, lines, status_code=200, body=b""):
        self._lines = list(lines)
        self.status_code = status_code
        self._body = body

    def iter_lines(self):
        return iter(self._lines)

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _ScriptedClientFactory:
    def __init__(self, responses, on_init=None):
        self._responses = list(responses)
        self._index = 0
        self.requests = []
        self._on_init = on_init

    def __call__(self, *args, **kwargs):
        if self._on_init:
            self._on_init()
        factory = self

        class _Client:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def stream(self, method, url, json=None, headers=None):
                factory.requests.append(
                    {"method": method, "url": url, "json": json, "headers": headers}
                )
                resp = factory._responses[factory._index]
                factory._index += 1
                return resp

        return _Client()


class _SequenceClock:
    def __init__(self, timestamps):
        self._it = iter(timestamps)
        self._last = timestamps[-1]

    def __call__(self):
        try:
            self._last = next(self._it)
        except StopIteration:
            pass
        return self._last


class TestCB01BufferedAndDecodeTiming(unittest.TestCase):
    def test_single_frame_buffered_completion_never_reports_bogus_decode_tps(self):
        """CB01 reproduction: 1 answer frame + usage(512) + DONE with 10us teardown."""
        lines = _sse_lines(
            [
                _delta(content="Entire buffered answer in one SSE frame."),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"completion_tokens": 512, "prompt_tokens": 3584},
                },
            ],
            include_done=True,
        )
        # start=10.0, first_and_only_frame=12.0, end=12.000010 (10us teardown)
        clock = _SequenceClock([10.0, 12.0, 12.000010])
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )

        with (
            mock.patch.object(cb, "httpx", fake_httpx),
            mock.patch.object(cb.time, "monotonic", side_effect=clock),
        ):
            r = cb.run_single_test(
                [{"role": "user", "content": "summarize"}],
                "4K",
                "test-model",
                requested_context_tokens=4096,
                requested_prompt_tokens=3584,
            )

        self.assertEqual(r["run_status"], "completed")
        self.assertEqual(r["generation_frames"], 1)
        self.assertEqual(r["output_tokens"], 512)
        self.assertIsNone(r["tokens_per_sec"])
        self.assertIsNone(r["decode_time_s"])
        self.assertFalse(r["decode_tps_measurable"])
        self.assertEqual(r["decode_tps_exclusion_reason"], "buffered_single_frame")
        self.assertFalse(r["qualifies_for_recommendation"])
        self.assertIn("buffered_single_frame", r["disqualification_reasons"])
        # End-to-end throughput is reported separately: 512 / 2.00001s ~= 256.0
        self.assertAlmostEqual(r["end_to_end_tokens_per_sec"], 256.0, delta=0.1)

    def test_sub_threshold_coalesced_frames_mark_decode_tps_unmeasurable(self):
        """Two generation frames arriving 10us apart must not publish 51.2M tok/s."""
        lines = _sse_lines(
            [
                _delta(content="part 1 "),
                _delta(content="part 2"),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"completion_tokens": 512, "prompt_tokens": 3584},
                },
            ],
            include_done=True,
        )
        # start=10.0, frame1=11.0, frame2=11.000010 (10us delta), end=11.000020
        clock = _SequenceClock([10.0, 11.0, 11.000010, 11.000020])
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )

        with (
            mock.patch.object(cb, "httpx", fake_httpx),
            mock.patch.object(cb.time, "monotonic", side_effect=clock),
        ):
            r = cb.run_single_test(
                [{"role": "user", "content": "summarize"}],
                "4K",
                "test-model",
                requested_context_tokens=4096,
                requested_prompt_tokens=3584,
            )

        self.assertEqual(r["generation_frames"], 2)
        self.assertIsNone(r["tokens_per_sec"])
        self.assertIsNone(r["decode_time_s"])
        self.assertAlmostEqual(r["observed_decode_span_s"], 0.000010, places=6)
        self.assertFalse(r["decode_tps_measurable"])
        self.assertEqual(
            r["decode_tps_exclusion_reason"], "sub_threshold_decode_interval"
        )
        self.assertFalse(r["qualifies_for_recommendation"])
        self.assertIsNotNone(r["end_to_end_tokens_per_sec"])

    def test_decode_time_uses_first_to_last_generation_frame_not_teardown(self):
        """A long pause after the last text frame before [DONE] must not inflate decode_time."""
        lines = _sse_lines(
            [
                _delta(content="tok1 "),
                _delta(content="tok2 "),
                _delta(content="tok3"),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"completion_tokens": 100, "prompt_tokens": 3584},
                },
            ],
            include_done=True,
        )
        # start=0.0, f1=1.0, f2=2.0, f3=3.0 (decode_span=2.0s), end=8.0 (5s teardown stall)
        clock = _SequenceClock([0.0, 1.0, 2.0, 3.0, 8.0])
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )

        with (
            mock.patch.object(cb, "httpx", fake_httpx),
            mock.patch.object(cb.time, "monotonic", side_effect=clock),
        ):
            r = cb.run_single_test(
                [{"role": "user", "content": "summarize"}],
                "4K",
                "test-model",
                requested_context_tokens=4096,
                requested_prompt_tokens=3584,
            )

        self.assertTrue(r["decode_tps_measurable"])
        self.assertEqual(r["generation_frames"], 3)
        self.assertEqual(r["decode_time_s"], 2.0)
        self.assertEqual(r["tokens_per_sec"], 50.0)  # 100 / 2.0s
        self.assertEqual(r["total_s"], 8.0)
        self.assertEqual(r["end_to_end_tokens_per_sec"], 12.5)  # 100 / 8.0s
        self.assertTrue(r["qualifies_for_recommendation"])

    def test_combined_reasoning_and_content_in_one_frame_counts_as_single_frame(self):
        lines = _sse_lines(
            [
                _delta(reasoning="thought", content="answer"),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"completion_tokens": 20, "prompt_tokens": 3584},
                },
            ],
            include_done=True,
        )
        clock = _SequenceClock([0.0, 0.5, 0.501])
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )
        with (
            mock.patch.object(cb, "httpx", fake_httpx),
            mock.patch.object(cb.time, "monotonic", side_effect=clock),
        ):
            r = cb.run_single_test([{"role": "user", "content": "x"}], "4K", "m")

        self.assertEqual(r["generation_frames"], 1)
        self.assertIsNone(r["tokens_per_sec"])
        self.assertEqual(r["decode_tps_exclusion_reason"], "buffered_single_frame")


class TestCB02NoOutputAndZeroBaseline(unittest.TestCase):
    def test_done_only_stream_is_marked_no_output_and_failed(self):
        """CB02 reproduction: stream containing only `data: [DONE]`."""
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(["data: [DONE]"])]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )
        with mock.patch.object(cb, "httpx", fake_httpx):
            r = cb.run_single_test([{"role": "user", "content": "x"}], "4K", "m")

        self.assertEqual(r["run_status"], "no_output")
        self.assertEqual(r["output_tokens"], 0)
        self.assertEqual(r["generation_frames"], 0)
        self.assertIsNone(r["ttft_s"])
        self.assertIsNone(r["tokens_per_sec"])
        self.assertIsNone(r["end_to_end_tokens_per_sec"])
        self.assertEqual(r["decode_tps_exclusion_reason"], "no_generated_output")
        self.assertFalse(r["qualifies_for_recommendation"])
        self.assertIn("error", r)

    def test_stop_with_zero_tokens_is_marked_no_output(self):
        lines = _sse_lines(
            [
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"completion_tokens": 0, "prompt_tokens": 3584},
                },
            ],
            include_done=True,
        )
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )
        with mock.patch.object(cb, "httpx", fake_httpx):
            r = cb.run_single_test(
                [{"role": "user", "content": "x"}],
                "4K",
                "m",
                requested_context_tokens=4096,
                requested_prompt_tokens=3584,
            )

        self.assertEqual(r["run_status"], "no_output")
        self.assertEqual(r["output_tokens"], 0)
        self.assertFalse(r["qualifies_for_recommendation"])
        self.assertIn("error", r)

    def test_usage_tokens_without_any_text_frames_marked_no_output(self):
        lines = _sse_lines(
            [
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"completion_tokens": 5, "prompt_tokens": 3584},
                },
            ],
            include_done=True,
        )
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )
        with mock.patch.object(cb, "httpx", fake_httpx):
            r = cb.run_single_test(
                [{"role": "user", "content": "x"}],
                "4K",
                "m",
                requested_context_tokens=4096,
                requested_prompt_tokens=3584,
            )

        self.assertEqual(r["run_status"], "no_output")
        self.assertEqual(r["generation_frames"], 0)
        self.assertIsNone(r["tokens_per_sec"])
        self.assertEqual(r["decode_tps_exclusion_reason"], "no_generated_output")
        self.assertFalse(r["qualifies_for_recommendation"])
        self.assertIn("error", r)

    def test_http_500_and_timeout_rows_excluded_but_visible(self):
        fake_httpx_500 = types.SimpleNamespace(
            Client=_ScriptedClientFactory(
                [_FakeResponse([], status_code=500, body=b"internal error")]
            ),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )
        with mock.patch.object(cb, "httpx", fake_httpx_500):
            r500 = cb.run_single_test(
                [{"role": "user", "content": "x"}],
                "4K",
                "m",
                requested_context_tokens=4096,
                requested_prompt_tokens=3584,
            )
        self.assertEqual(r500["run_status"], "error")
        self.assertIn("HTTP 500", r500["error"])
        self.assertFalse(r500["qualifies_for_recommendation"])
        self.assertIsNone(r500["tokens_per_sec"])

        class _TimeoutClient:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def stream(self, *a, **k):
                raise TimeoutError("timed out")

        fake_httpx_to = types.SimpleNamespace(
            Client=_TimeoutClient,
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )
        with mock.patch.object(cb, "httpx", fake_httpx_to):
            rto = cb.run_single_test([{"role": "user", "content": "x"}], "4K", "m")
        self.assertEqual(rto["run_status"], "error")
        self.assertIn("Timeout", rto["error"])
        self.assertFalse(rto["qualifies_for_recommendation"])

    def test_no_output_first_tier_does_not_disable_throughput_drop_threshold(self):
        """A zero-output 4K row must not produce a zero baseline that disables drop checks."""
        summary = {
            "4K": cb.summarize_context_runs(
                4096,
                [
                    {
                        "run_status": "no_output",
                        "qualifies_for_recommendation": False,
                        "decode_tps_measurable": False,
                        "decode_tps_exclusion_reason": "no_generated_output",
                        "tokens_per_sec": None,
                        "end_to_end_tokens_per_sec": None,
                        "ttft_s": None,
                        "total_s": 0.1,
                        "error": "Empty completion",
                        "truncated": False,
                        "prompt_token_mismatch": False,
                        "host_pageouts_delta": 0,
                    }
                ],
            ),
            "8K": cb.summarize_context_runs(
                8192,
                [
                    {
                        "run_status": "completed",
                        "qualifies_for_recommendation": True,
                        "decode_tps_measurable": True,
                        "decode_tps_exclusion_reason": None,
                        "tokens_per_sec": 100.0,
                        "end_to_end_tokens_per_sec": 80.0,
                        "ttft_s": 0.5,
                        "total_s": 5.5,
                        "truncated": False,
                        "prompt_token_mismatch": False,
                        "host_pageouts_delta": 0,
                    }
                ],
            ),
            "16K": cb.summarize_context_runs(
                16384,
                [
                    {
                        "run_status": "completed",
                        "qualifies_for_recommendation": True,
                        "decode_tps_measurable": True,
                        "decode_tps_exclusion_reason": None,
                        "tokens_per_sec": 50.0,  # 50% drop vs 8K baseline!
                        "end_to_end_tokens_per_sec": 40.0,
                        "ttft_s": 1.0,
                        "total_s": 11.0,
                        "truncated": False,
                        "prompt_token_mismatch": False,
                        "host_pageouts_delta": 0,
                    }
                ],
            ),
        }
        ev = cb.evaluate_summary(summary)
        self.assertEqual(ev["baseline_tps"], 100.0)
        self.assertEqual(ev["recommended_context"], "8K")
        self.assertEqual(ev["rows"]["4K"]["status"], "NO_OUTPUT")
        self.assertEqual(ev["rows"]["8K"]["status"], "OK")
        self.assertEqual(ev["rows"]["16K"]["status"], "-50%")


class TestCB03TruncationTokenMismatchAndCacheMode(unittest.TestCase):
    def test_truncated_and_mismatched_4k_row_is_visible_but_never_recommended(self):
        """CB03 reproduction: 4K with finish_reason=length and usage_prompt_tokens=8192."""
        lines = _sse_lines(
            [
                _delta(content="truncated "),
                _delta(content="output"),
                {"choices": [{"delta": {}, "finish_reason": "length"}]},
                {
                    "choices": [],
                    "usage": {"completion_tokens": 512, "prompt_tokens": 8192},
                },
            ],
            include_done=True,
        )
        clock = _SequenceClock([0.0, 0.5, 5.5, 5.6])
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.json"
            buf = io.StringIO()
            with (
                mock.patch.object(cb, "httpx", fake_httpx),
                mock.patch.object(cb.time, "monotonic", side_effect=clock),
                mock.patch.object(cb, "get_pageout_count", return_value=0),
                mock.patch.object(cb, "get_memory_pressure", return_value=80),
                redirect_stdout(buf),
            ):
                payload = cb.run_benchmark(
                    "http://127.0.0.1:8090/v1",
                    iterations=1,
                    context_sizes=[4096],
                    model_name="test-model",
                    results_file=results_path,
                )

        raw = payload["raw"][0]
        self.assertEqual(raw["run_status"], "length_limited")
        self.assertTrue(raw["truncated"])
        self.assertEqual(raw["requested_prompt_tokens"], 3584)
        self.assertEqual(raw["observed_prompt_tokens"], 8192)
        self.assertFalse(raw["prompt_tokens_within_tolerance"])
        self.assertTrue(raw["prompt_token_mismatch"])
        self.assertFalse(raw["qualifies_for_recommendation"])
        self.assertIn("length_limited", raw["disqualification_reasons"])
        self.assertIn("prompt_token_mismatch", raw["disqualification_reasons"])

        summary_4k = payload["summary"]["4K"]
        self.assertEqual(summary_4k["total_runs"], 1)
        self.assertEqual(summary_4k["runs"], 0)
        self.assertEqual(summary_4k["qualifying_runs"], 0)
        self.assertEqual(summary_4k["cohorts"]["length_limited"], 1)
        self.assertEqual(summary_4k["cohorts"]["token_mismatch"], 1)
        self.assertIsNotNone(summary_4k["truncated_cohort"])
        self.assertEqual(summary_4k["truncated_cohort"]["runs"], 1)
        self.assertEqual(summary_4k["status"], "TRUNC+SIZE")
        self.assertFalse(summary_4k["qualifies"])

        self.assertIsNone(payload["recommendation"]["recommended_context"])
        out_text = buf.getvalue()
        self.assertIn("RECOMMENDATION: None", out_text)
        self.assertNotIn("Stay at 4K", out_text)
        self.assertIn("4K", out_text)
        self.assertIn("TRUNC+SIZE", out_text)

    def test_context_budget_overrun_flagged_even_within_percentage_tolerance(self):
        """usage.prompt_tokens > requested_context_tokens must be flagged as mismatch."""
        lines = _sse_lines(
            [
                _delta(content="a "),
                _delta(content="b"),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"completion_tokens": 10, "prompt_tokens": 4200},
                },
            ],
            include_done=True,
        )
        clock = _SequenceClock([0.0, 0.2, 1.2, 1.3])
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )
        with (
            mock.patch.object(cb, "httpx", fake_httpx),
            mock.patch.object(cb.time, "monotonic", side_effect=clock),
        ):
            r = cb.run_single_test(
                [{"role": "user", "content": "x"}],
                "4K",
                "m",
                requested_context_tokens=4096,
                requested_prompt_tokens=3584,
            )

        self.assertTrue(r["prompt_token_mismatch"])
        self.assertFalse(r["qualifies_for_recommendation"])
        self.assertIn("prompt_token_mismatch", r["disqualification_reasons"])

    def test_mixed_valid_and_truncated_tier_blocks_recommendation_of_that_tier(self):
        summary = {
            "4K": cb.summarize_context_runs(
                4096,
                [
                    {
                        "run_status": "completed",
                        "qualifies_for_recommendation": True,
                        "decode_tps_measurable": True,
                        "decode_tps_exclusion_reason": None,
                        "tokens_per_sec": 90.0,
                        "end_to_end_tokens_per_sec": 75.0,
                        "ttft_s": 0.4,
                        "total_s": 6.0,
                        "truncated": False,
                        "prompt_token_mismatch": False,
                        "host_pageouts_delta": 0,
                    }
                ],
            ),
            "8K": cb.summarize_context_runs(
                8192,
                [
                    {
                        "run_status": "completed",
                        "qualifies_for_recommendation": True,
                        "decode_tps_measurable": True,
                        "decode_tps_exclusion_reason": None,
                        "tokens_per_sec": 88.0,
                        "end_to_end_tokens_per_sec": 72.0,
                        "ttft_s": 0.6,
                        "total_s": 6.5,
                        "truncated": False,
                        "prompt_token_mismatch": False,
                        "host_pageouts_delta": 0,
                    },
                    {
                        "run_status": "length_limited",
                        "qualifies_for_recommendation": False,
                        "decode_tps_measurable": True,
                        "decode_tps_exclusion_reason": None,
                        "tokens_per_sec": 85.0,
                        "end_to_end_tokens_per_sec": 70.0,
                        "ttft_s": 0.6,
                        "total_s": 6.8,
                        "truncated": True,
                        "prompt_token_mismatch": False,
                        "host_pageouts_delta": 0,
                    },
                ],
            ),
        }
        ev = cb.evaluate_summary(summary)
        self.assertEqual(ev["recommended_context"], "4K")
        self.assertEqual(ev["rows"]["8K"]["status"], "TRUNCATED")
        self.assertFalse(ev["rows"]["8K"]["qualifies"])
        self.assertEqual(summary["8K"]["cohorts"]["completed"], 1)
        self.assertEqual(summary["8K"]["cohorts"]["length_limited"], 1)
        self.assertIsNotNone(summary["8K"]["truncated_cohort"])

    def test_cold_vs_warm_cache_modes_and_request_identities(self):
        p_cold_1 = cb.build_prompt(3584, request_id="run-1", cache_mode="cold")
        p_cold_2 = cb.build_prompt(3584, request_id="run-2", cache_mode="cold")
        self.assertNotEqual(
            cb.compute_prompt_sha256(p_cold_1), cb.compute_prompt_sha256(p_cold_2)
        )
        self.assertIn("run-1", p_cold_1[1]["content"])
        self.assertIn("run-2", p_cold_2[1]["content"])

        p_warm_1 = cb.build_prompt(3584, request_id="run-1", cache_mode="warm")
        p_warm_2 = cb.build_prompt(3584, request_id="run-2", cache_mode="warm")
        self.assertEqual(
            cb.compute_prompt_sha256(p_warm_1), cb.compute_prompt_sha256(p_warm_2)
        )

        lines = _sse_lines(
            [
                _delta(content="hello "),
                _delta(content="world"),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {
                        "completion_tokens": 10,
                        "prompt_tokens": 3584,
                        "prompt_tokens_details": {"cached_tokens": 3500},
                    },
                },
            ],
            include_done=True,
        )
        factory = _ScriptedClientFactory([_FakeResponse(lines)])
        fake_httpx = types.SimpleNamespace(
            Client=factory,
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )
        clock = _SequenceClock([0.0, 0.1, 0.6, 0.7])
        with (
            mock.patch.object(cb, "httpx", fake_httpx),
            mock.patch.object(cb.time, "monotonic", side_effect=clock),
        ):
            r = cb.run_single_test(
                p_warm_1,
                "4K",
                "m",
                requested_context_tokens=4096,
                requested_prompt_tokens=3584,
                request_id="req-warm-1",
                cache_mode="warm",
            )

        self.assertEqual(r["request_id"], "req-warm-1")
        self.assertEqual(r["cache_mode"], "warm")
        self.assertEqual(r["cached_prompt_tokens"], 3500)
        self.assertEqual(factory.requests[0]["headers"], {"X-Request-Id": "req-warm-1"})


class TestCB04MemoryTelemetryAndClientTiming(unittest.TestCase):
    def test_get_pageout_count_returns_none_when_vm_stat_unavailable(self):
        with mock.patch.object(cb.subprocess, "run", side_effect=FileNotFoundError):
            self.assertIsNone(cb.get_pageout_count())

    def test_is_local_endpoint_classification(self):
        self.assertTrue(cb.is_local_endpoint("http://127.0.0.1:8090/v1"))
        self.assertTrue(cb.is_local_endpoint("http://localhost:8000/v1"))
        self.assertTrue(cb.is_local_endpoint("http://[::1]:8000/v1"))
        self.assertFalse(cb.is_local_endpoint("https://api.example.com/v1"))
        self.assertFalse(cb.is_local_endpoint("http://192.0.2.10:8090/v1"))

    def test_remote_endpoint_outputs_null_host_memory_and_never_zero(self):
        lines = _sse_lines(
            [
                _delta(content="tok1 "),
                _delta(content="tok2"),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"completion_tokens": 10, "prompt_tokens": 3584},
                },
            ],
            include_done=True,
        )
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )
        clock = _SequenceClock([0.0, 0.2, 0.7, 0.8])
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "remote.json"
            buf = io.StringIO()
            with (
                mock.patch.object(cb, "httpx", fake_httpx),
                mock.patch.object(cb.time, "monotonic", side_effect=clock),
                mock.patch.object(cb, "get_pageout_count", return_value=999) as mock_pg,
                mock.patch.object(cb, "get_memory_pressure", return_value=50) as mock_mp,
                redirect_stdout(buf),
            ):
                payload = cb.run_benchmark(
                    "https://api.example.com/v1",
                    iterations=1,
                    context_sizes=[4096],
                    model_name="remote-model",
                    results_file=results_path,
                )

        mock_pg.assert_not_called()
        mock_mp.assert_not_called()
        self.assertFalse(payload["config"]["endpoint_is_local"])
        self.assertEqual(payload["config"]["endpoint_host"], "api.example.com")
        self.assertFalse(payload["config"]["memory_telemetry"]["applicable"])
        self.assertEqual(
            payload["config"]["memory_telemetry"]["reason"],
            "remote_endpoint_not_sampled",
        )
        self.assertIsNone(payload["raw"][0]["host_pageouts_delta"])
        self.assertIsNone(payload["raw"][0]["host_mem_free_pct_before"])
        self.assertIsNone(payload["summary"]["4K"]["avg_pageouts"])
        self.assertFalse(payload["recommendation"]["swap_telemetry_available"])
        self.assertIn("unavailable", payload["recommendation"]["note"])
        self.assertIn("N/A", buf.getvalue())

    def test_start_timer_begins_after_http_client_construction(self):
        lines = _sse_lines(
            [
                _delta(content="a "),
                _delta(content="b"),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"completion_tokens": 10, "prompt_tokens": 3584},
                },
            ],
            include_done=True,
        )
        now = {"t": 0.0}

        def advance_on_client_init():
            now["t"] += 5.0  # Simulate 5s client init overhead

        # Sequence after init (t=5.0): start=5.0, frame1=5.2, frame2=6.2, end=6.3
        post_init_offsets = iter([0.0, 0.2, 1.2, 1.3])

        def monotonic_fn():
            return now["t"] + next(post_init_offsets)

        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory(
                [_FakeResponse(lines)], on_init=advance_on_client_init
            ),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )
        with (
            mock.patch.object(cb, "httpx", fake_httpx),
            mock.patch.object(cb.time, "monotonic", side_effect=monotonic_fn),
        ):
            r = cb.run_single_test(
                [{"role": "user", "content": "x"}],
                "4K",
                "m",
                requested_context_tokens=4096,
                requested_prompt_tokens=3584,
            )

        self.assertEqual(r["client_ttft_s"], 0.2)
        self.assertEqual(r["total_s"], 1.3)

    def test_swap_pressure_blocks_recommendation(self):
        summary = {
            "4K": cb.summarize_context_runs(
                4096,
                [
                    {
                        "run_status": "completed",
                        "qualifies_for_recommendation": True,
                        "decode_tps_measurable": True,
                        "decode_tps_exclusion_reason": None,
                        "tokens_per_sec": 80.0,
                        "end_to_end_tokens_per_sec": 70.0,
                        "ttft_s": 0.4,
                        "total_s": 7.0,
                        "truncated": False,
                        "prompt_token_mismatch": False,
                        "host_pageouts_delta": 0,
                    }
                ],
            ),
            "8K": cb.summarize_context_runs(
                8192,
                [
                    {
                        "run_status": "completed",
                        "qualifies_for_recommendation": True,
                        "decode_tps_measurable": True,
                        "decode_tps_exclusion_reason": None,
                        "tokens_per_sec": 78.0,
                        "end_to_end_tokens_per_sec": 68.0,
                        "ttft_s": 0.6,
                        "total_s": 7.5,
                        "truncated": False,
                        "prompt_token_mismatch": False,
                        "host_pageouts_delta": 250,
                    }
                ],
            ),
        }
        ev = cb.evaluate_summary(summary)
        self.assertEqual(ev["recommended_context"], "4K")
        self.assertEqual(ev["rows"]["8K"]["status"], "SWAP")
        self.assertFalse(ev["rows"]["8K"]["qualifies"])


class TestSyntheticMockSSEDemo(unittest.TestCase):
    def test_end_to_end_synthetic_demo_over_real_loopback_http(self):
        """Full end-to-end HTTP+SSE demo exercising normal, buffered, truncated, and degraded rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "demo-results.json"
            buf = io.StringIO()
            with (
                mock.patch.object(cb, "get_pageout_count", return_value=None),
                mock.patch.object(cb, "get_memory_pressure", return_value=None),
                redirect_stdout(buf),
            ):
                payload = cb.run_demo(
                    results_path,
                    iterations=1,
                    context_sizes=[4096, 8192, 16384, 32768, 65536],
                    cache_mode="cold",
                )

            self.assertTrue(results_path.exists())
            saved = json.loads(results_path.read_text())

        # 4K and 8K stream cleanly and qualify; 8K is the highest qualifying tier
        self.assertEqual(saved["summary"]["4K"]["status"], "OK")
        self.assertEqual(saved["summary"]["8K"]["status"], "OK")
        self.assertIsNotNone(saved["summary"]["4K"]["avg_tps"])
        self.assertIsNotNone(saved["summary"]["8K"]["avg_tps"])

        # 16K is buffered single-frame: decode TPS must be None, E2E TPS present
        s16 = saved["summary"]["16K"]
        self.assertEqual(s16["status"], "NO_DECODE")
        self.assertIsNone(s16["avg_tps"])
        self.assertIsNotNone(s16["avg_e2e_tps"])
        self.assertEqual(s16["decode_tps_exclusions"], {"buffered_single_frame": 1})
        self.assertFalse(s16["qualifies"])

        # 32K is truncated + prompt token mismatch: visible in census, excluded
        s32 = saved["summary"]["32K"]
        self.assertEqual(s32["status"], "TRUNC+SIZE")
        self.assertEqual(s32["cohorts"]["length_limited"], 1)
        self.assertEqual(s32["cohorts"]["token_mismatch"], 1)
        self.assertIsNotNone(s32["truncated_cohort"])
        self.assertFalse(s32["qualifies"])

        # 64K has >15% decode TPS degradation
        s64 = saved["summary"]["64K"]
        self.assertTrue(s64["status"].startswith("-"))
        self.assertFalse(s64["qualifies"])

        # Recommendation lands cleanly on 8K
        self.assertEqual(saved["recommendation"]["recommended_context"], "8K")
        self.assertTrue(saved["recommendation"]["recommended_context_verified"])
        self.assertEqual(saved["recommendation"]["recommendation_status"], "verified")
        self.assertFalse(saved["recommendation"]["unverified"])
        self.assertEqual(saved["config"]["cache_mode"], "cold")
        self.assertTrue(saved["config"]["endpoint_is_local"])
        self.assertEqual(
            saved["config"]["memory_telemetry"]["reason"],
            "macos_counters_unavailable",
        )
        self.assertIn("RECOMMENDATION: 8K context window", buf.getvalue())


class TestCB05UnverifiedPromptUsageGating(unittest.TestCase):
    def test_stream_without_usage_block_never_qualifies_verified_recommendation(self):
        """CB03/CB05 counterexample: 2 nonempty frames + stop, no usage -> disqualified."""
        lines = _sse_lines(
            [
                _delta(content="Normal streamed "),
                _delta(content="completion without usage."),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ],
            include_done=True,
        )
        clock = _SequenceClock([0.0, 0.2, 0.7, 0.8])
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "no-usage.json"
            buf = io.StringIO()
            with (
                mock.patch.object(cb, "httpx", fake_httpx),
                mock.patch.object(cb.time, "monotonic", side_effect=clock),
                mock.patch.object(cb, "get_pageout_count", return_value=0),
                mock.patch.object(cb, "get_memory_pressure", return_value=80),
                redirect_stdout(buf),
            ):
                payload = cb.run_benchmark(
                    "http://127.0.0.1:8090/v1",
                    iterations=1,
                    context_sizes=[4096],
                    model_name="no-usage-model",
                    results_file=results_path,
                )

        raw = payload["raw"][0]
        self.assertEqual(raw["run_status"], "completed")
        self.assertIsNone(raw["observed_prompt_tokens"])
        self.assertFalse(raw["prompt_tokens_verified"])
        self.assertEqual(raw["prompt_size_basis"], "heuristic_estimate")
        self.assertTrue(raw["decode_tps_measurable"])
        self.assertGreater(raw["tokens_per_sec"], 0)
        self.assertFalse(raw["qualifies_for_recommendation"])
        self.assertFalse(raw["qualifies_for_verified_recommendation"])
        self.assertIn("missing_prompt_token_usage", raw["disqualification_reasons"])

        s4 = payload["summary"]["4K"]
        self.assertEqual(s4["total_runs"], 1)
        self.assertEqual(s4["runs"], 1)
        self.assertEqual(s4["qualifying_runs"], 0)
        self.assertEqual(s4["excluded_runs"], 1)
        self.assertIsNone(s4["observed_prompt_tokens_avg"])
        self.assertFalse(s4["prompt_tokens_verified"])
        self.assertEqual(s4["prompt_size_basis"], "heuristic_estimate")
        self.assertEqual(s4["cohorts"]["missing_prompt_token_usage"], 1)
        self.assertIsNotNone(s4["avg_tps"])
        self.assertEqual(s4["status"], "UNVERIFIED")
        self.assertFalse(s4["qualifies"])
        self.assertFalse(s4["verified"])

        rec = payload["recommendation"]
        self.assertIsNone(rec["recommended_context"])
        self.assertFalse(rec["recommended_context_verified"])
        self.assertEqual(rec["recommendation_status"], "none")
        self.assertFalse(rec["allow_unverified_usage"])
        self.assertEqual(rec["unverified_tiers"], ["4K"])
        self.assertIn("usage.prompt_tokens", rec["note"])

        out_text = buf.getvalue()
        self.assertIn("UNVERIFIED_SIZE(est~3584)", out_text)
        self.assertIn("~4K", out_text)
        self.assertIn("unverif", out_text)
        self.assertIn("UNVERIFIED", out_text)
        self.assertIn("missing_prompt_token_usage=1", out_text)
        self.assertIn("RECOMMENDATION: None", out_text)

    def test_usage_with_only_completion_tokens_still_disqualifies_missing_prompt_usage(self):
        """A usage object missing prompt_tokens (or with prompt_tokens=None) cannot qualify."""
        for usage_obj in (
            {"completion_tokens": 12},
            {"completion_tokens": 12, "prompt_tokens": None},
            {"completion_tokens": 12, "prompt_tokens": "3584"},
            {"completion_tokens": 12, "prompt_tokens": True},
        ):
            lines = _sse_lines(
                [
                    _delta(content="part 1 "),
                    _delta(content="part 2"),
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": usage_obj},
                ],
                include_done=True,
            )
            clock = _SequenceClock([0.0, 0.2, 0.7, 0.8])
            fake_httpx = types.SimpleNamespace(
                Client=_ScriptedClientFactory([_FakeResponse(lines)]),
                Timeout=lambda *a, **k: None,
                TimeoutException=TimeoutError,
            )
            with (
                mock.patch.object(cb, "httpx", fake_httpx),
                mock.patch.object(cb.time, "monotonic", side_effect=clock),
            ):
                r = cb.run_single_test(
                    [{"role": "user", "content": "x"}],
                    "4K",
                    "m",
                    requested_context_tokens=4096,
                    requested_prompt_tokens=3584,
                )
            self.assertIsNone(r["observed_prompt_tokens"], msg=f"usage={usage_obj}")
            self.assertFalse(r["prompt_tokens_verified"], msg=f"usage={usage_obj}")
            self.assertFalse(
                r["qualifies_for_recommendation"], msg=f"usage={usage_obj}"
            )
            self.assertIn(
                "missing_prompt_token_usage",
                r["disqualification_reasons"],
                msg=f"usage={usage_obj}",
            )

    def test_verified_usage_within_tolerance_still_qualifies(self):
        """Explicit server usage.prompt_tokens within tolerance qualifies cleanly."""
        lines = _sse_lines(
            [
                _delta(content="Verified "),
                _delta(content="streamed answer."),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"completion_tokens": 20, "prompt_tokens": 3600},
                },
            ],
            include_done=True,
        )
        clock = _SequenceClock([0.0, 0.2, 0.7, 0.8])
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "verified.json"
            buf = io.StringIO()
            with (
                mock.patch.object(cb, "httpx", fake_httpx),
                mock.patch.object(cb.time, "monotonic", side_effect=clock),
                mock.patch.object(cb, "get_pageout_count", return_value=0),
                mock.patch.object(cb, "get_memory_pressure", return_value=80),
                redirect_stdout(buf),
            ):
                payload = cb.run_benchmark(
                    "http://127.0.0.1:8090/v1",
                    iterations=1,
                    context_sizes=[4096],
                    model_name="verified-model",
                    results_file=results_path,
                )

        raw = payload["raw"][0]
        self.assertEqual(raw["observed_prompt_tokens"], 3600)
        self.assertTrue(raw["prompt_tokens_verified"])
        self.assertEqual(raw["prompt_size_basis"], "server_usage")
        self.assertTrue(raw["prompt_tokens_within_tolerance"])
        self.assertFalse(raw["prompt_token_mismatch"])
        self.assertTrue(raw["qualifies_for_recommendation"])
        self.assertTrue(raw["qualifies_for_verified_recommendation"])
        self.assertEqual(raw["disqualification_reasons"], [])

        s4 = payload["summary"]["4K"]
        self.assertEqual(s4["status"], "OK")
        self.assertTrue(s4["qualifies"])
        self.assertTrue(s4["verified"])
        self.assertTrue(s4["prompt_tokens_verified"])
        self.assertEqual(s4["cohorts"]["missing_prompt_token_usage"], 0)

        rec = payload["recommendation"]
        self.assertEqual(rec["recommended_context"], "4K")
        self.assertTrue(rec["recommended_context_verified"])
        self.assertEqual(rec["recommendation_status"], "verified")
        self.assertFalse(rec["unverified"])
        self.assertIn("RECOMMENDATION: 4K context window", buf.getvalue())
        self.assertNotIn("~4K", buf.getvalue())

    def test_opt_in_allow_unverified_usage_tags_recommendation_unverified(self):
        """Opt-in allow_unverified_usage permits recommendation ONLY with explicit unverified tags."""
        lines = _sse_lines(
            [
                _delta(content="Streamed "),
                _delta(content="without usage."),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ],
            include_done=True,
        )
        clock = _SequenceClock([0.0, 0.2, 0.7, 0.8])
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "optin-unverified.json"
            buf = io.StringIO()
            with (
                mock.patch.object(cb, "httpx", fake_httpx),
                mock.patch.object(cb.time, "monotonic", side_effect=clock),
                mock.patch.object(cb, "get_pageout_count", return_value=0),
                mock.patch.object(cb, "get_memory_pressure", return_value=80),
                redirect_stdout(buf),
            ):
                payload = cb.run_benchmark(
                    "http://127.0.0.1:8090/v1",
                    iterations=1,
                    context_sizes=[4096],
                    model_name="no-usage-model",
                    results_file=results_path,
                    allow_unverified_usage=True,
                )

        raw = payload["raw"][0]
        self.assertIsNone(raw["observed_prompt_tokens"])
        self.assertFalse(raw["prompt_tokens_verified"])
        self.assertFalse(raw["qualifies_for_verified_recommendation"])
        self.assertTrue(raw["qualifies_for_recommendation"])
        self.assertIn("missing_prompt_token_usage", raw["verification_warnings"])

        s4 = payload["summary"]["4K"]
        self.assertEqual(s4["status"], "OK (unverified)")
        self.assertTrue(s4["qualifies"])
        self.assertFalse(s4["verified"])
        self.assertFalse(s4["prompt_tokens_verified"])

        rec = payload["recommendation"]
        self.assertEqual(rec["recommended_context"], "4K")
        self.assertFalse(rec["recommended_context_verified"])
        self.assertEqual(rec["recommendation_status"], "unverified")
        self.assertTrue(rec["unverified"])
        self.assertTrue(rec["allow_unverified_usage"])
        self.assertIn("UNVERIFIED", rec["note"])

        out_text = buf.getvalue()
        self.assertIn("~4K", out_text)
        self.assertIn("OK (unverified)", out_text)
        self.assertIn("[UNVERIFIED", out_text)

    def test_mixed_verified_and_unverified_tiers_and_baseline_isolation(self):
        """Unverified 4K must not establish verified baseline or qualify; verified 8K qualifies."""
        unverified_4k = _FakeResponse(
            _sse_lines(
                [
                    _delta(content="fast "),
                    _delta(content="unverified"),
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                ],
                include_done=True,
            )
        )
        verified_8k = _FakeResponse(
            _sse_lines(
                [
                    _delta(content="verified "),
                    _delta(content="8k"),
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                    {
                        "choices": [],
                        "usage": {"completion_tokens": 100, "prompt_tokens": 7680},
                    },
                ],
                include_done=True,
            )
        )
        verified_16k_slow = _FakeResponse(
            _sse_lines(
                [
                    _delta(content="verified "),
                    _delta(content="16k"),
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                    {
                        "choices": [],
                        "usage": {"completion_tokens": 100, "prompt_tokens": 15872},
                    },
                ],
                include_done=True,
            )
        )
        # 4K: 2 tokens / 0.1s = 20 tps (unverified, excluded from baseline)
        # 8K: 100 tokens / 1.0s = 100 tps (first verified baseline!)
        # 16K: 100 tokens / 2.0s = 50 tps (-50% drop vs 8K baseline)
        clock = _SequenceClock(
            [
                0.0, 0.1, 0.2, 0.25,
                1.0, 1.2, 2.2, 2.3,
                3.0, 3.2, 5.2, 5.3,
            ]
        )
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([unverified_4k, verified_8k, verified_16k_slow]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "mixed.json"
            buf = io.StringIO()
            with (
                mock.patch.object(cb, "httpx", fake_httpx),
                mock.patch.object(cb.time, "monotonic", side_effect=clock),
                mock.patch.object(cb, "get_pageout_count", return_value=0),
                mock.patch.object(cb, "get_memory_pressure", return_value=80),
                redirect_stdout(buf),
            ):
                payload = cb.run_benchmark(
                    "http://127.0.0.1:8090/v1",
                    iterations=1,
                    context_sizes=[4096, 8192, 16384],
                    model_name="mixed-model",
                    results_file=results_path,
                )

        self.assertEqual(payload["summary"]["4K"]["status"], "UNVERIFIED")
        self.assertFalse(payload["summary"]["4K"]["qualifies"])
        self.assertEqual(payload["summary"]["8K"]["status"], "OK")
        self.assertTrue(payload["summary"]["8K"]["qualifies"])
        self.assertEqual(payload["summary"]["16K"]["status"], "-50%")
        self.assertFalse(payload["summary"]["16K"]["qualifies"])
        self.assertEqual(payload["recommendation"]["baseline_tps"], 100.0)
        self.assertEqual(payload["recommendation"]["recommended_context"], "8K")
        self.assertTrue(payload["recommendation"]["recommended_context_verified"])

    def test_opt_in_allow_unverified_usage_still_rejects_truncation_and_buffered_decode(self):
        """--allow-unverified-usage must never bypass truncation or single-frame decode gating."""
        truncated_no_usage = _sse_lines(
            [
                _delta(content="cut "),
                _delta(content="off"),
                {"choices": [{"delta": {}, "finish_reason": "length"}]},
            ],
            include_done=True,
        )
        buffered_no_usage = _sse_lines(
            [
                _delta(content="single buffered frame"),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ],
            include_done=True,
        )
        clock = _SequenceClock([0.0, 0.1, 0.6, 0.7, 1.0, 1.2, 1.2001])
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory(
                [_FakeResponse(truncated_no_usage), _FakeResponse(buffered_no_usage)]
            ),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )
        with (
            mock.patch.object(cb, "httpx", fake_httpx),
            mock.patch.object(cb.time, "monotonic", side_effect=clock),
        ):
            r_trunc = cb.run_single_test(
                [{"role": "user", "content": "x"}],
                "4K",
                "m",
                requested_context_tokens=4096,
                requested_prompt_tokens=3584,
                allow_unverified_usage=True,
            )
            r_buf = cb.run_single_test(
                [{"role": "user", "content": "x"}],
                "8K",
                "m",
                requested_context_tokens=8192,
                requested_prompt_tokens=7680,
                allow_unverified_usage=True,
            )

        self.assertFalse(r_trunc["qualifies_for_recommendation"])
        self.assertIn("length_limited", r_trunc["disqualification_reasons"])
        self.assertFalse(r_buf["qualifies_for_recommendation"])
        self.assertIn("buffered_single_frame", r_buf["disqualification_reasons"])

    def test_evaluate_summary_fail_closed_when_prompt_tokens_verified_false(self):
        """Even if a caller constructs a summary row with qualifying_runs=1, fail closed."""
        summary = {
            "4K": {
                "context_size": 4096,
                "runs": 1,
                "qualifying_runs": 1,
                "avg_tps": 80.0,
                "avg_host_pageouts": 0.0,
                "prompt_tokens_verified": False,
                "cohorts": {"missing_prompt_token_usage": 1},
            }
        }
        ev = cb.evaluate_summary(summary, allow_unverified_usage=False)
        self.assertIsNone(ev["recommended_context"])
        self.assertEqual(ev["rows"]["4K"]["status"], "UNVERIFIED")
        self.assertFalse(ev["rows"]["4K"]["qualifies"])

        ev_opt = cb.evaluate_summary(summary, allow_unverified_usage=True)
        self.assertEqual(ev_opt["recommended_context"], "4K")
        self.assertFalse(ev_opt["recommended_context_verified"])
        self.assertTrue(ev_opt["unverified"])
        self.assertEqual(ev_opt["rows"]["4K"]["status"], "OK (unverified)")

    def test_cli_allow_unverified_usage_flag_defaults_false_and_opts_in(self):
        parser = cb.build_arg_parser()
        default_args = parser.parse_args([])
        self.assertFalse(default_args.allow_unverified_usage)
        opt_in_args = parser.parse_args(["--allow-unverified-usage"])
        self.assertTrue(opt_in_args.allow_unverified_usage)


class TestCB06ObservedTotalContextAccounting(unittest.TestCase):
    def _make_4000_plus_512_stream(self):
        return _sse_lines(
            [
                _delta(content="First generation frame "),
                _delta(content="second generation frame."),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 4000, "completion_tokens": 512},
                },
            ],
            include_done=True,
        )

    def test_cb06_4000_prompt_plus_512_output_under_nominal_4096_default_mode(self):
        """CB06 reproduction: 4000 prompt + 512 output = 4512 total under nominal 4K (4096)."""
        lines = self._make_4000_plus_512_stream()
        clock = _SequenceClock([0.0, 0.2, 0.7, 0.8])
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "cb06-default.json"
            buf = io.StringIO()
            with (
                mock.patch.object(cb, "httpx", fake_httpx),
                mock.patch.object(cb.time, "monotonic", side_effect=clock),
                mock.patch.object(cb, "get_pageout_count", return_value=0),
                mock.patch.object(cb, "get_memory_pressure", return_value=80),
                redirect_stdout(buf),
            ):
                payload = cb.run_benchmark(
                    "http://127.0.0.1:8090/v1",
                    iterations=1,
                    context_sizes=[4096],
                    model_name="cb06-model",
                    results_file=results_path,
                )

        raw = payload["raw"][0]
        self.assertEqual(raw["requested_context_tokens"], 4096)
        self.assertEqual(raw["requested_prompt_tokens"], 3584)
        self.assertEqual(raw["observed_prompt_tokens"], 4000)
        self.assertEqual(raw["observed_output_tokens"], 512)
        self.assertEqual(raw["observed_total_tokens"], 4512)
        self.assertFalse(raw["effective_total_within_tier"])
        self.assertTrue(raw["total_token_overrun"])
        self.assertTrue(raw["prompt_tokens_verified"])
        self.assertTrue(raw["prompt_tokens_within_tolerance"])
        self.assertFalse(raw["prompt_token_mismatch"])
        self.assertTrue(raw["qualifies_for_recommendation"])
        self.assertTrue(raw["qualifies_for_verified_recommendation"])
        self.assertFalse(raw["qualifies_for_exact_total_recommendation"])
        self.assertEqual(raw["disqualification_reasons"], [])
        self.assertIn(
            "total_tokens_exceeded_nominal_tier", raw["verification_warnings"]
        )

        s4 = payload["summary"]["4K"]
        self.assertEqual(s4["observed_prompt_tokens_avg"], 4000.0)
        self.assertEqual(s4["observed_output_tokens_avg"], 512.0)
        self.assertEqual(s4["avg_observed_total_tokens"], 4512.0)
        self.assertEqual(s4["max_observed_total_tokens"], 4512)
        self.assertFalse(s4["effective_total_within_tier"])
        self.assertEqual(s4["total_within_tier_runs"], 0)
        self.assertEqual(s4["total_overrun_runs"], 1)
        self.assertEqual(s4["cohorts"]["total_context_overrun"], 1)
        self.assertEqual(s4["status"], "OK (tot>tier)")
        self.assertTrue(s4["qualifies"])
        self.assertTrue(s4["verified"])

        rec = payload["recommendation"]
        self.assertEqual(rec["recommended_context"], "4K")
        self.assertEqual(rec["recommended_nominal_tier"], "4K")
        self.assertEqual(rec["recommended_nominal_context_tokens"], 4096)
        self.assertEqual(rec["recommended_observed_prompt_tokens"], 4000.0)
        self.assertEqual(rec["recommended_observed_total_tokens"], 4512.0)
        self.assertFalse(rec["recommended_effective_total_within_tier"])
        self.assertTrue(rec["recommended_context_verified"])
        self.assertEqual(rec["recommendation_status"], "verified")
        self.assertEqual(rec["verification_scope"], "verified_prompt_tier_only")
        self.assertFalse(rec["strict_total_context"])
        self.assertEqual(rec["total_overrun_tiers"], ["4K"])
        self.assertIn("4512", rec["note"])
        self.assertIn("4096", rec["note"])
        self.assertIn("--strict-total-context", rec["note"])

        out_text = buf.getvalue()
        self.assertIn("ObsTot", out_text)
        self.assertIn("4512", out_text)
        self.assertIn("OK (tot>tier)", out_text)
        self.assertIn("TOTAL_OVER(obs=4512/4096)", out_text)
        self.assertIn("EXCEEDS nominal tier", out_text)

    def test_cb06_4000_prompt_plus_512_output_rejected_under_strict_total_context(self):
        """With --strict-total-context, 4000 + 512 = 4512 under nominal 4096 is rejected."""
        lines = self._make_4000_plus_512_stream()
        clock = _SequenceClock([0.0, 0.2, 0.7, 0.8])
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "cb06-strict.json"
            buf = io.StringIO()
            with (
                mock.patch.object(cb, "httpx", fake_httpx),
                mock.patch.object(cb.time, "monotonic", side_effect=clock),
                mock.patch.object(cb, "get_pageout_count", return_value=0),
                mock.patch.object(cb, "get_memory_pressure", return_value=80),
                redirect_stdout(buf),
            ):
                payload = cb.run_benchmark(
                    "http://127.0.0.1:8090/v1",
                    iterations=1,
                    context_sizes=[4096],
                    model_name="cb06-model",
                    results_file=results_path,
                    strict_total_context=True,
                )

        raw = payload["raw"][0]
        self.assertEqual(raw["observed_total_tokens"], 4512)
        self.assertFalse(raw["effective_total_within_tier"])
        self.assertTrue(raw["total_token_overrun"])
        self.assertFalse(raw["qualifies_for_recommendation"])
        self.assertFalse(raw["qualifies_for_verified_recommendation"])
        self.assertFalse(raw["qualifies_for_exact_total_recommendation"])
        self.assertIn("total_context_overrun", raw["disqualification_reasons"])

        s4 = payload["summary"]["4K"]
        self.assertEqual(s4["avg_observed_total_tokens"], 4512.0)
        self.assertFalse(s4["effective_total_within_tier"])
        self.assertEqual(s4["qualifying_runs"], 0)
        self.assertEqual(s4["excluded_runs"], 1)
        self.assertEqual(s4["cohorts"]["total_context_overrun"], 1)
        self.assertEqual(s4["status"], "TOTAL_OVER")
        self.assertFalse(s4["qualifies"])

        rec = payload["recommendation"]
        self.assertIsNone(rec["recommended_context"])
        self.assertIsNone(rec["recommended_nominal_tier"])
        self.assertIsNone(rec["recommended_nominal_context_tokens"])
        self.assertIsNone(rec["recommended_observed_prompt_tokens"])
        self.assertIsNone(rec["recommended_observed_total_tokens"])
        self.assertIsNone(rec["recommended_effective_total_within_tier"])
        self.assertFalse(rec["recommended_context_verified"])
        self.assertEqual(rec["recommendation_status"], "none")
        self.assertEqual(rec["verification_scope"], "none")
        self.assertTrue(rec["strict_total_context"])
        self.assertEqual(rec["total_overrun_tiers"], ["4K"])
        self.assertTrue(payload["config"]["strict_total_context"])

        out_text = buf.getvalue()
        self.assertIn("ObsTot", out_text)
        self.assertIn("4512", out_text)
        self.assertIn("TOTAL_OVER", out_text)
        self.assertIn("total_context_overrun=1", out_text)
        self.assertIn("RECOMMENDATION: None", out_text)

    def test_strict_total_context_accepts_exact_boundary_and_isolates_baseline(self):
        """Under --strict-total-context, overrun 4K must not set baseline; exact 8192/8192 8K qualifies."""
        overrun_4k = _FakeResponse(self._make_4000_plus_512_stream())
        exact_8k = _FakeResponse(
            _sse_lines(
                [
                    _delta(content="Exact "),
                    _delta(content="8K total."),
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 7680, "completion_tokens": 512},
                    },
                ],
                include_done=True,
            )
        )
        slow_16k = _FakeResponse(
            _sse_lines(
                [
                    _delta(content="Slow "),
                    _delta(content="16K total."),
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 15872, "completion_tokens": 512},
                    },
                ],
                include_done=True,
            )
        )
        # 4K: 512 / 0.5s = 1024 tps (rejected TOTAL_OVER -> must NOT become baseline)
        # 8K: 512 / 1.0s = 512 tps (first strict-qualifying baseline)
        # 16K: 512 / 2.0s = 256 tps (-50% vs 8K baseline)
        clock = _SequenceClock(
            [
                0.0, 0.1, 0.6, 0.7,
                1.0, 1.2, 2.2, 2.3,
                3.0, 3.2, 5.2, 5.3,
            ]
        )
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([overrun_4k, exact_8k, slow_16k]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "cb06-strict-multi.json"
            buf = io.StringIO()
            with (
                mock.patch.object(cb, "httpx", fake_httpx),
                mock.patch.object(cb.time, "monotonic", side_effect=clock),
                mock.patch.object(cb, "get_pageout_count", return_value=0),
                mock.patch.object(cb, "get_memory_pressure", return_value=80),
                redirect_stdout(buf),
            ):
                payload = cb.run_benchmark(
                    "http://127.0.0.1:8090/v1",
                    iterations=1,
                    context_sizes=[4096, 8192, 16384],
                    model_name="cb06-multi",
                    results_file=results_path,
                    strict_total_context=True,
                )

        self.assertEqual(payload["summary"]["4K"]["status"], "TOTAL_OVER")
        self.assertFalse(payload["summary"]["4K"]["qualifies"])
        self.assertEqual(payload["summary"]["8K"]["status"], "OK")
        self.assertTrue(payload["summary"]["8K"]["qualifies"])
        self.assertEqual(payload["summary"]["8K"]["avg_observed_total_tokens"], 8192.0)
        self.assertTrue(payload["summary"]["8K"]["effective_total_within_tier"])
        self.assertEqual(payload["summary"]["16K"]["status"], "-50%")
        self.assertFalse(payload["summary"]["16K"]["qualifies"])

        rec = payload["recommendation"]
        self.assertEqual(rec["baseline_tps"], 512.0)
        self.assertEqual(rec["recommended_context"], "8K")
        self.assertEqual(rec["recommended_nominal_tier"], "8K")
        self.assertEqual(rec["recommended_nominal_context_tokens"], 8192)
        self.assertEqual(rec["recommended_observed_prompt_tokens"], 7680.0)
        self.assertEqual(rec["recommended_observed_total_tokens"], 8192.0)
        self.assertTrue(rec["recommended_effective_total_within_tier"])
        self.assertEqual(
            rec["verification_scope"], "verified_prompt_and_total_within_tier"
        )
        self.assertIn("<= 8192 within tier", buf.getvalue())

    def test_strict_total_context_rejects_missing_completion_tokens(self):
        """If server reports prompt_tokens but omits completion_tokens, strict mode rejects."""
        lines = _sse_lines(
            [
                _delta(content="part 1 "),
                _delta(content="part 2"),
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 3584}},
            ],
            include_done=True,
        )
        clock = _SequenceClock([0.0, 0.2, 0.7, 0.8])
        fake_httpx = types.SimpleNamespace(
            Client=_ScriptedClientFactory([_FakeResponse(lines)]),
            Timeout=lambda *a, **k: None,
            TimeoutException=TimeoutError,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "cb06-missing-comp.json"
            buf = io.StringIO()
            with (
                mock.patch.object(cb, "httpx", fake_httpx),
                mock.patch.object(cb.time, "monotonic", side_effect=clock),
                mock.patch.object(cb, "get_pageout_count", return_value=0),
                mock.patch.object(cb, "get_memory_pressure", return_value=80),
                redirect_stdout(buf),
            ):
                payload = cb.run_benchmark(
                    "http://127.0.0.1:8090/v1",
                    iterations=1,
                    context_sizes=[4096],
                    model_name="cb06-no-comp",
                    results_file=results_path,
                    strict_total_context=True,
                )

        raw = payload["raw"][0]
        self.assertEqual(raw["observed_prompt_tokens"], 3584)
        self.assertIsNone(raw["observed_output_tokens"])
        self.assertIsNone(raw["observed_total_tokens"])
        self.assertIsNone(raw["effective_total_within_tier"])
        self.assertFalse(raw["qualifies_for_recommendation"])
        self.assertIn("missing_total_token_usage", raw["disqualification_reasons"])
        self.assertEqual(payload["summary"]["4K"]["status"], "UNVERIFIED")
        self.assertIsNone(payload["recommendation"]["recommended_context"])

    def test_cli_strict_total_context_flag_defaults_false_and_opts_in(self):
        parser = cb.build_arg_parser()
        default_args = parser.parse_args([])
        self.assertFalse(default_args.strict_total_context)
        opt_in_args = parser.parse_args(["--strict-total-context"])
        self.assertTrue(opt_in_args.strict_total_context)


if __name__ == "__main__":
    unittest.main()
