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

# Load the hyphenated script as a module, relative to THIS file, so the test works
# from any clone location.
_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "ctxbench", os.path.join(_HERE, "context-bench.py"))
cb = importlib.util.module_from_spec(spec)
sys.modules["ctxbench"] = cb
spec.loader.exec_module(cb)


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


cb.httpx = types.SimpleNamespace(
    Client=FakeClient,
    Timeout=lambda *a, **k: None,
    TimeoutException=Exception,
)


def d(**kw):
    return {"choices": [{"delta": kw}]}


fails = []


def check(name, cond, detail=""):
    print("  %-58s %s" % (name, "OK" if cond else "FAIL " + detail))
    if not cond:
        fails.append(name)


print("== reasoning model: 3 think tokens then 2 content tokens ==")
FakeClient.frames = [
    d(reasoning="Let me "), d(reasoning="think about "), d(reasoning="this. "),
    d(content="Buy"), d(content=" AAPL"),
    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    {"choices": [], "usage": {"completion_tokens": 5, "prompt_tokens": 4000}},
]
r = cb.run_single_test([{"role": "user", "content": "x"}], "4K", "m")
check("counts reasoning chars", r["reasoning_chars"] == 25, str(r["reasoning_chars"]))
check("counts content chars", r["content_chars"] == 8, str(r["content_chars"]))
check("flags that it reasoned", r["reasoned"] is True)
check("uses server usage for tokens", r["output_tokens"] == 5 and r["token_source"] == "usage")
check("reports reasoning share", r.get("reasoning_share") == round(25 / 33, 3))
check("captures finish_reason", r["finish_reason"] == "stop")
check("not truncated", r["truncated"] is False)
check("ttft_content >= ttft (think came first)",
      r.get("ttft_content_s", 0) >= r["ttft_s"])

print("\n== the OLD bug: content-only counting would have reported ~2 tokens ==")
check("output_tokens is NOT the content-only count", r["output_tokens"] != 2,
      "got %s" % r["output_tokens"])

print("\n== vendor field alias: reasoning_content ==")
FakeClient.frames = [d(reasoning_content="hmm "), d(content="ok"),
                     {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
r2 = cb.run_single_test([{"role": "user", "content": "x"}], "4K", "m")
check("accepts reasoning_content alias", r2["reasoning_chars"] == 4 and r2["reasoned"])

print("\n== truncation is surfaced ==")
FakeClient.frames = [d(content="a"), {"choices": [{"delta": {}, "finish_reason": "length"}]}]
r3 = cb.run_single_test([{"role": "user", "content": "x"}], "4K", "m")
check("finish_reason=length marks truncated", r3["truncated"] is True)

print("\n== no usage block -> falls back to chunk count, and says so ==")
FakeClient.frames = [d(content="a"), d(content="b"), d(content="c")]
r4 = cb.run_single_test([{"role": "user", "content": "x"}], "4K", "m")
check("falls back to sse chunk count", r4["output_tokens"] == 3
      and r4["token_source"] == "sse_chunks")
check("non-reasoning run flagged reasoned=False", r4["reasoned"] is False)

print("\n== think kwargs are actually sent ==")
cb.THINK_KWARGS = {"enable_thinking": True}
FakeClient.frames = [d(content="a")]
cb.run_single_test([{"role": "user", "content": "x"}], "4K", "m")
check("chat_template_kwargs forwarded",
      FakeClient.captured.get("chat_template_kwargs") == {"enable_thinking": True},
      str(FakeClient.captured.get("chat_template_kwargs")))
check("stream_options requests usage",
      FakeClient.captured.get("stream_options") == {"include_usage": True})
cb.THINK_KWARGS = {}
FakeClient.frames = [d(content="a")]
cb.run_single_test([{"role": "user", "content": "x"}], "4K", "m")
check("omitted entirely when unset", "chat_template_kwargs" not in FakeClient.captured)

print("\n%d check(s) failed" % len(fails))
sys.exit(1 if fails else 0)
