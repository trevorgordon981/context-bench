#!/usr/bin/env python3
"""
Context Window Benchmark for local LLM serving endpoints (MLX, vLLM, llama.cpp).
Tests client-observed TTFT, streaming decode throughput, end-to-end throughput,
and local host memory/swap telemetry across context sizes.

Usage:
  python3 context-bench.py [--api-url URL] [--iterations N] [--model NAME]
  python3 context-bench.py --demo
"""

import argparse
from collections import Counter
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import uuid

import httpx

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8090/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "default")
CONTEXT_SIZES = [4096, 8192, 16384, 32768, 65536, 98304, 131072]
ITERATIONS = 3
OUTPUT_TOKENS = 512
TIMEOUT_S = 600
RESULTS_FILE = Path("./context-bench-results.json")

# Minimum elapsed span between first and last nonempty generation frames for
# decode TPS to be considered measurable. Intervals below this threshold (e.g.
# buffered or TCP-coalesced bursts) are marked unmeasurable rather than
# dividing by a near-zero teardown delta.
MIN_DECODE_INTERVAL_S = 0.01

# Maximum allowed relative difference between requested prompt tokens and
# server-reported usage.prompt_tokens before a row is flagged as mismatched.
PROMPT_TOKEN_TOLERANCE = 0.25

# Recommendation thresholds.
SWAP_PAGEOUT_THRESHOLD = 100
TPS_DROP_THRESHOLD_PCT = 15.0

# Finish reasons considered normal completion (when combined with non-empty output).
NORMAL_FINISH_REASONS = {None, "stop", "eos", "end_turn"}

# Loopback hostnames considered local to the benchmark runner.
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

# Technical documentation filler — repeated to fill context windows.
# Each block is ~265 words (~1,889 chars).
FILLER_BLOCK = (
    "The transformer architecture processes input sequences through multi-head "
    "self-attention layers, where each head computes scaled dot-product attention "
    "over query, key, and value projections of the input embeddings. The attention "
    "mechanism allows each token position to attend to all other positions in the "
    "sequence, enabling the model to capture long-range dependencies. In the "
    "feed-forward sublayer, two linear transformations with a ReLU activation "
    "produce the output, which is combined with the attention output via residual "
    "connections and layer normalization. During inference, the key-value cache "
    "stores previously computed attention states, enabling autoregressive generation "
    "without recomputing attention over the full context at each step. The prefill "
    "phase processes all input tokens in parallel to populate the KV cache, while "
    "the decode phase generates output tokens one at a time, appending each new "
    "token's KV state to the cache. Memory consumption scales linearly with both "
    "sequence length and the number of attention heads, making context window size "
    "a critical factor in deployment. Quantization reduces memory pressure by "
    "representing weights and activations in lower precision formats such as 4-bit "
    "or 6-bit integers, trading a small accuracy penalty for significant memory "
    "savings. The mixture-of-experts architecture activates only a subset of "
    "parameters for each token, reducing computational cost while maintaining "
    "model capacity. Grouped query attention shares key and value projections "
    "across multiple query heads, reducing the KV cache size proportionally. "
    "Speculative decoding uses a smaller draft model to propose candidate tokens "
    "that the larger model verifies in parallel, improving throughput for latency- "
    "bound workloads. Continuous batching allows new requests to enter the batch "
    "as earlier ones complete, maximizing GPU utilization in serving scenarios.\n\n"
)

# Heuristic character-to-token ratio used for prompt sizing; server usage counts
# are checked against PROMPT_TOKEN_TOLERANCE when reported.
CHARS_PER_TOKEN = 3.5

# Extra chat-template arguments, forwarded verbatim as `chat_template_kwargs`.
# Reasoning models gate their think phase on a MODEL-SPECIFIC key with a
# MODEL-SPECIFIC default -- there is no portable switch:
#   DeepSeek / Qwen : {"enable_thinking": true|false}   default OFF
#   MiniMax         : {"thinking_mode": true|false}     default ON
# Benchmarking without pinning this compares a thinking run against a non-thinking
# one and calls the difference throughput. Empty dict = server default.
THINK_KWARGS = {}


def is_local_endpoint(api_url):
    """Return True if api_url targets a loopback address on the benchmark host."""
    try:
        host = (urlparse(api_url).hostname or "").lower()
    except Exception:
        return False
    return host in LOCAL_HOSTS


def get_benchmark_host_info():
    """Return non-identifying platform metadata for the benchmark runner host."""
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }


def get_memory_pressure():
    """Parse macOS `memory_pressure -Q` free-memory snapshot, or return None."""
    try:
        result = subprocess.run(
            ["memory_pressure", "-Q"], capture_output=True, text=True, timeout=10
        )
        output = result.stdout + result.stderr
        for line in output.splitlines():
            if "System-wide memory free percentage" in line:
                pct = line.strip().split(":")[-1].strip().rstrip("%")
                return int(pct)
    except Exception:
        pass
    return None


def get_pageout_count():
    """Get current macOS `vm_stat` page-out counter, or return None if unavailable."""
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if "Pageouts" in line:
                return int(line.split(":")[1].strip().rstrip("."))
    except Exception:
        pass
    return None


def sample_host_memory(endpoint_is_local):
    """Sample local host (pageouts, mem_free_pct) only when targeting a local endpoint."""
    if not endpoint_is_local:
        return None, None
    return get_pageout_count(), get_memory_pressure()


def compute_prompt_sha256(messages):
    """Return a deterministic SHA-256 hex digest of the serialized prompt messages."""
    encoded = json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_prompt(target_tokens, request_id=None, cache_mode="cold"):
    """Build a prompt padded to approximately target_tokens.

    In `cold` cache mode (default), a unique request_id prefix is placed at the
    beginning of the user message so server-side prefix caches cannot reuse KV
    states across iterations. In `warm` mode, the exact same prompt text is
    reused across iterations.
    """
    if cache_mode not in ("cold", "warm"):
        raise ValueError(f"Unsupported cache_mode: {cache_mode!r}")

    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    system_msg = (
        "You are a helpful AI assistant. Summarize the key technical concepts "
        "from the documentation provided."
    )
    nonce_prefix = (
        f"[Benchmark Request ID: {request_id}]\n\n"
        if (cache_mode == "cold" and request_id)
        else ""
    )
    user_prefix = (
        f"{nonce_prefix}Please read the following technical documentation carefully, "
        "then provide a concise summary of the most important concepts:\n\n"
    )
    user_suffix = "\n\nNow summarize the key concepts above in a few paragraphs."

    overhead_chars = len(system_msg) + len(user_prefix) + len(user_suffix)
    fill_chars = max(target_chars - overhead_chars, 100)

    # Repeat filler block to fill context
    repeats = (fill_chars // len(FILLER_BLOCK)) + 1
    filler = (FILLER_BLOCK * repeats)[:fill_chars]

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prefix + filler + user_suffix},
    ]


def _open_stream(client, url, body, request_id):
    """Open an SSE POST stream, passing X-Request-Id when supported by the client."""
    if request_id:
        try:
            return client.stream(
                "POST",
                url,
                json=body,
                headers={"X-Request-Id": request_id},
            )
        except TypeError:
            # Support minimal test stubs whose stream(method, url, json=None)
            # signature does not accept headers.
            pass
    return client.stream("POST", url, json=body)


def _parse_token_count(value):
    """Return a non-negative int token count from server usage, or None."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def _is_missing_prompt_usage(record):
    """Return True if a run record is missing verified server prompt token usage."""
    if "missing_prompt_token_usage" in (record.get("disqualification_reasons") or []):
        return True
    if "missing_prompt_token_usage" in (record.get("verification_warnings") or []):
        return True
    if "prompt_tokens_verified" in record:
        return (
            record["prompt_tokens_verified"] is False
            and record.get("observed_prompt_tokens") is None
        )
    if "observed_prompt_tokens" in record:
        return record["observed_prompt_tokens"] is None
    return False


def _is_contradictory_completion_usage(record):
    """Return True if a run record reports <=0 completion usage despite nonempty output."""
    if record.get("contradictory_completion_usage") is True:
        return True
    if "contradictory_completion_token_usage" in (
        record.get("disqualification_reasons") or []
    ):
        return True
    if "contradictory_completion_token_usage" in (
        record.get("verification_warnings") or []
    ):
        return True
    has_frames = (record.get("generation_frames") or 0) > 0 or (
        (record.get("content_chars") or 0) + (record.get("reasoning_chars") or 0)
    ) > 0
    if has_frames:
        obs_out = record.get("observed_output_tokens")
        if (
            not isinstance(obs_out, bool)
            and isinstance(obs_out, (int, float))
            and obs_out <= 0
        ):
            return True
        if record.get("completion_tokens_verified") is False:
            return True
    return False


def _extract_observed_total_tokens(record):
    """Return server-observed prompt+output total tokens from a run record, or None."""
    if _is_contradictory_completion_usage(record):
        return None
    if record.get("completion_tokens_verified") is False:
        return None
    total = _parse_token_count(record.get("observed_total_tokens"))
    if total is not None:
        return total
    prompt = _parse_token_count(record.get("observed_prompt_tokens"))
    output = _parse_token_count(record.get("observed_output_tokens"))
    if prompt is not None and output is not None:
        return prompt + output
    return None


def _error_result(
    error_message,
    *,
    requested_context_tokens,
    requested_prompt_tokens,
    requested_output_tokens,
    request_id,
    cache_mode,
    prompt_sha256,
):
    """Return a structured error row so failed runs remain visible in census reports."""
    return {
        "error": error_message,
        "run_status": "error",
        "qualifies_for_recommendation": False,
        "qualifies_for_verified_recommendation": False,
        "qualifies_for_exact_total_recommendation": False,
        "disqualification_reasons": ["error"],
        "verification_warnings": [],
        "ttft_s": None,
        "client_ttft_s": None,
        "total_s": None,
        "output_tokens": 0,
        "observed_output_tokens": None,
        "completion_tokens_verified": False,
        "contradictory_completion_usage": False,
        "requested_output_tokens": requested_output_tokens,
        "requested_context_tokens": requested_context_tokens,
        "requested_prompt_tokens": requested_prompt_tokens,
        "observed_prompt_tokens": None,
        "observed_total_tokens": None,
        "effective_total_within_tier": None,
        "total_token_overrun": False,
        "prompt_tokens_verified": False,
        "prompt_size_basis": "heuristic_estimate",
        "prompt_token_ratio": None,
        "prompt_tokens_within_tolerance": None,
        "prompt_token_mismatch": False,
        "tokens_per_sec": None,
        "end_to_end_tokens_per_sec": None,
        "decode_time_s": None,
        "observed_decode_span_s": None,
        "generation_frames": 0,
        "decode_tps_measurable": False,
        "decode_tps_exclusion_reason": "request_error",
        "reasoning_chars": 0,
        "content_chars": 0,
        "reasoned": False,
        "token_source": None,
        "finish_reason": None,
        "truncated": False,
        "think_kwargs": dict(THINK_KWARGS) or None,
        "request_id": request_id,
        "cache_mode": cache_mode,
        "prompt_sha256": prompt_sha256,
    }


def run_single_test(
    messages,
    ctx_label,
    model_name=None,
    *,
    requested_context_tokens=None,
    requested_prompt_tokens=None,
    requested_output_tokens=None,
    request_id=None,
    cache_mode="cold",
    prompt_token_tolerance=PROMPT_TOKEN_TOLERANCE,
    min_decode_interval_s=MIN_DECODE_INTERVAL_S,
    allow_unverified_usage=False,
    strict_total_context=False,
):
    """Send a streaming request and measure client-observed prefill/decode behaviour.

    Metric semantics:
      * `client_ttft_s` (`ttft_s`): wall time from HTTP request dispatch to the
        first nonempty generation frame (reasoning or content). Includes client
        network round-trip, server queueing, and prompt prefill.
      * `tokens_per_sec`: streaming decode throughput computed ONLY when at
        least two distinct nonempty generation frames arrive with an inter-frame
        span >= `min_decode_interval_s` between first and last generation frame.
        Single-frame/buffered responses, coalesced bursts, and zero-output
        responses report `tokens_per_sec = None` with an explicit
        `decode_tps_exclusion_reason`.
      * `end_to_end_tokens_per_sec`: `output_tokens / total_s`, reported
        separately from streaming decode throughput.
      * `run_status`: distinguishes `completed`, `length_limited`, `no_output`,
        `incomplete`, `other_finish`, and `error`.
      * Requested vs server-reported `usage.prompt_tokens` are compared against
        `prompt_token_tolerance` and `requested_context_tokens`. When the server
        omits `usage.prompt_tokens`, `prompt_tokens_verified` is False and the
        run is disqualified (`missing_prompt_token_usage`) unless
        `allow_unverified_usage=True` is explicitly opted in (still tagged
        unverified).
      * Combined `observed_total_tokens` (`observed_prompt_tokens +
        observed_output_tokens`) and `effective_total_within_tier`
        (`observed_total_tokens <= requested_context_tokens`) are tracked
        separately from prompt-tier verification. In default mode, a combined
        total exceeding `requested_context_tokens` is flagged as a warning
        (`total_tokens_exceeded_nominal_tier`); with `strict_total_context=True`,
        combined total overrun or missing total usage disqualifies the run.
    """
    req_output_tokens = (
        OUTPUT_TOKENS if requested_output_tokens is None else requested_output_tokens
    )
    prompt_sha256 = compute_prompt_sha256(messages)
    body = {
        "model": model_name or MODEL_NAME,
        "messages": messages,
        "max_tokens": req_output_tokens,
        "stream": True,
        "temperature": 0.7,
        # Ask for exact token accounting. Servers that don't support it ignore it.
        "stream_options": {"include_usage": True},
    }
    if THINK_KWARGS:
        body["chat_template_kwargs"] = dict(THINK_KWARGS)

    content_chunks = 0
    reasoning_chunks = 0
    content_chars = 0
    reasoning_chars = 0
    generation_frames = 0
    first_any_time = None
    last_gen_time = None
    first_content_time = None
    finish_reason = None
    usage = None
    saw_done = False

    err_kwargs = dict(
        requested_context_tokens=requested_context_tokens,
        requested_prompt_tokens=requested_prompt_tokens,
        requested_output_tokens=req_output_tokens,
        request_id=request_id,
        cache_mode=cache_mode,
        prompt_sha256=prompt_sha256,
    )

    try:
        with httpx.Client(timeout=httpx.Timeout(TIMEOUT_S, connect=30)) as client:
            start = time.monotonic()
            with _open_stream(
                client, f"{API_URL}/chat/completions", body, request_id
            ) as resp:
                if resp.status_code != 200:
                    error_body = resp.read().decode(errors="replace")
                    return _error_result(
                        f"HTTP {resp.status_code}: {error_body[:200]}",
                        **err_kwargs,
                    )

                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue  # usage-only frame carries no choices
                    choice = choices[0]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}

                    # Vendors disagree on the field name; accept both.
                    think = (
                        delta.get("reasoning") or delta.get("reasoning_content") or ""
                    )
                    content = delta.get("content") or ""

                    if think or content:
                        frame_time = time.monotonic()
                        generation_frames += 1
                        if first_any_time is None:
                            first_any_time = frame_time
                        last_gen_time = frame_time
                        if think:
                            reasoning_chunks += 1
                            reasoning_chars += len(think)
                        if content:
                            if first_content_time is None:
                                first_content_time = frame_time
                            content_chunks += 1
                            content_chars += len(content)

            end = time.monotonic()
    except httpx.TimeoutException:
        return _error_result(f"Timeout after {TIMEOUT_S}s", **err_kwargs)
    except Exception as e:
        return _error_result(str(e), **err_kwargs)

    total_time = max(end - start, 0.0)
    ttft = (first_any_time - start) if first_any_time is not None else None

    # Server-reported counts are authoritative when present and non-contradictory;
    # chunk counts are fallback. Reporting completion_tokens <= 0 despite nonempty
    # generated frames/text is contradictory and never treated as verified usage.
    chunk_total = content_chunks + reasoning_chunks
    raw_completion_tokens = (
        usage.get("completion_tokens") if isinstance(usage, dict) else None
    )
    usage_completion_tokens = _parse_token_count(raw_completion_tokens)
    usage_prompt_tokens = _parse_token_count(
        usage.get("prompt_tokens") if isinstance(usage, dict) else None
    )
    prompt_tokens_details = (
        usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    )
    cached_prompt_tokens = _parse_token_count(
        prompt_tokens_details.get("cached_tokens")
        if isinstance(prompt_tokens_details, dict)
        else None
    )

    has_generated_frames = (
        generation_frames > 0 or (content_chars + reasoning_chars) > 0
    )
    contradictory_completion_usage = (
        has_generated_frames
        and not isinstance(raw_completion_tokens, bool)
        and isinstance(raw_completion_tokens, (int, float))
        and raw_completion_tokens <= 0
    )
    verified_usage_completion_tokens = (
        usage_completion_tokens
        if (
            usage_completion_tokens is not None
            and not contradictory_completion_usage
        )
        else None
    )
    completion_tokens_verified = verified_usage_completion_tokens is not None

    if verified_usage_completion_tokens is not None and (
        verified_usage_completion_tokens > 0 or chunk_total == 0
    ):
        output_tokens = int(verified_usage_completion_tokens)
        token_source = "usage"
    else:
        output_tokens = chunk_total
        token_source = "sse_chunks"

    stream_completed = saw_done or (finish_reason is not None)
    has_generated_output = (generation_frames > 0) and (output_tokens > 0)

    # Decode TPS: strictly measured across distinct nonempty generation frames.
    observed_decode_span = None
    decode_time_s = None
    tokens_per_sec = None
    decode_tps_measurable = False
    decode_tps_exclusion_reason = None

    if not has_generated_output:
        decode_tps_exclusion_reason = "no_generated_output"
    elif generation_frames < 2:
        decode_tps_exclusion_reason = "buffered_single_frame"
    else:
        observed_decode_span = max(last_gen_time - first_any_time, 0.0)
        if observed_decode_span < min_decode_interval_s:
            decode_tps_exclusion_reason = "sub_threshold_decode_interval"
        else:
            decode_tps_measurable = True
            decode_time_s = round(observed_decode_span, 3)
            tokens_per_sec = round(output_tokens / observed_decode_span, 2)

    end_to_end_tps = (
        round(output_tokens / total_time, 2)
        if (output_tokens > 0 and total_time > 0)
        else None
    )

    # Stream outcome classification.
    error_msg = None
    if not stream_completed:
        run_status = "incomplete"
        truncated = True
        error_msg = "Incomplete SSE stream: closed before [DONE] or finish_reason"
    elif not has_generated_output:
        run_status = "no_output"
        truncated = False
        error_msg = "Empty completion: stream terminated with zero generated output"
    elif finish_reason == "length":
        run_status = "length_limited"
        truncated = True
    elif finish_reason not in NORMAL_FINISH_REASONS:
        run_status = "other_finish"
        truncated = False
    else:
        run_status = "completed"
        truncated = False

    # Requested vs observed prompt token budget verification.
    prompt_token_ratio = None
    prompt_tokens_within_tolerance = None
    prompt_token_mismatch = False
    if (
        usage_prompt_tokens is not None
        and requested_prompt_tokens is not None
        and requested_prompt_tokens > 0
    ):
        prompt_token_ratio = round(usage_prompt_tokens / requested_prompt_tokens, 3)
        rel_diff = (
            abs(usage_prompt_tokens - requested_prompt_tokens)
            / requested_prompt_tokens
        )
        over_context_budget = (
            requested_context_tokens is not None
            and usage_prompt_tokens > requested_context_tokens
        )
        prompt_tokens_within_tolerance = (
            rel_diff <= prompt_token_tolerance and not over_context_budget
        )
        prompt_token_mismatch = not prompt_tokens_within_tolerance

    # Prompt token count is only verified when the server returns a positive
    # usage.prompt_tokens count that passes budget/tolerance checks (when
    # requested_prompt_tokens is known).
    missing_prompt_usage = (
        usage_prompt_tokens is None
        or (usage_prompt_tokens == 0 and not prompt_token_mismatch)
    )
    prompt_tokens_verified = (
        not missing_prompt_usage and not prompt_token_mismatch
    )
    prompt_size_basis = (
        "server_usage" if usage_prompt_tokens is not None else "heuristic_estimate"
    )

    # Combined prompt + completion total token accounting (separate from prompt-tier
    # tolerance verification). Contradictory completion_tokens (<= 0 despite nonempty
    # generated frames/text) is never trusted as verified completion usage.
    observed_prompt_int = (
        int(usage_prompt_tokens) if usage_prompt_tokens is not None else None
    )
    observed_output_int = (
        int(verified_usage_completion_tokens)
        if verified_usage_completion_tokens is not None
        else None
    )
    observed_total_tokens = (
        observed_prompt_int + observed_output_int
        if (observed_prompt_int is not None and observed_output_int is not None)
        else None
    )
    effective_total_within_tier = (
        observed_total_tokens <= requested_context_tokens
        if (
            observed_total_tokens is not None
            and requested_context_tokens is not None
        )
        else None
    )
    total_token_overrun = effective_total_within_tier is False

    disqualification_reasons = []
    verification_warnings = []
    if run_status != "completed":
        disqualification_reasons.append(run_status)
    if (
        truncated
        and "length_limited" not in disqualification_reasons
        and "incomplete" not in disqualification_reasons
    ):
        disqualification_reasons.append("truncated")
    if prompt_token_mismatch:
        disqualification_reasons.append("prompt_token_mismatch")
    if missing_prompt_usage and error_msg is None:
        if allow_unverified_usage:
            verification_warnings.append("missing_prompt_token_usage")
        else:
            disqualification_reasons.append("missing_prompt_token_usage")
    if contradictory_completion_usage and error_msg is None:
        if strict_total_context:
            disqualification_reasons.append("contradictory_completion_token_usage")
        else:
            verification_warnings.append("contradictory_completion_token_usage")
    if total_token_overrun:
        if strict_total_context:
            disqualification_reasons.append("total_context_overrun")
        else:
            verification_warnings.append("total_tokens_exceeded_nominal_tier")
    elif (
        strict_total_context
        and observed_total_tokens is None
        and requested_context_tokens is not None
        and error_msg is None
    ):
        disqualification_reasons.append("missing_total_token_usage")
    if not decode_tps_measurable:
        disqualification_reasons.append(
            decode_tps_exclusion_reason or "unmeasurable_decode_tps"
        )
    qualifies_for_recommendation = len(disqualification_reasons) == 0
    qualifies_for_verified_recommendation = (
        qualifies_for_recommendation and prompt_tokens_verified
    )
    qualifies_for_exact_total_recommendation = (
        qualifies_for_verified_recommendation
        and completion_tokens_verified
        and (effective_total_within_tier is True)
    )

    rounded_ttft = round(ttft, 3) if ttft is not None else None
    out = {
        "run_status": run_status,
        "qualifies_for_recommendation": qualifies_for_recommendation,
        "qualifies_for_verified_recommendation": qualifies_for_verified_recommendation,
        "qualifies_for_exact_total_recommendation": qualifies_for_exact_total_recommendation,
        "disqualification_reasons": disqualification_reasons,
        "verification_warnings": verification_warnings,
        "ttft_s": rounded_ttft,
        "client_ttft_s": rounded_ttft,
        "total_s": round(total_time, 3),
        "output_tokens": output_tokens,
        "observed_output_tokens": observed_output_int,
        "completion_tokens_verified": completion_tokens_verified,
        "contradictory_completion_usage": contradictory_completion_usage,
        "requested_output_tokens": req_output_tokens,
        "requested_context_tokens": requested_context_tokens,
        "requested_prompt_tokens": requested_prompt_tokens,
        "observed_prompt_tokens": observed_prompt_int,
        "observed_total_tokens": observed_total_tokens,
        "effective_total_within_tier": effective_total_within_tier,
        "total_token_overrun": total_token_overrun,
        "prompt_tokens_verified": prompt_tokens_verified,
        "prompt_size_basis": prompt_size_basis,
        "prompt_token_ratio": prompt_token_ratio,
        "prompt_tokens_within_tolerance": prompt_tokens_within_tolerance,
        "prompt_token_mismatch": prompt_token_mismatch,
        "tokens_per_sec": tokens_per_sec,
        "end_to_end_tokens_per_sec": end_to_end_tps,
        "decode_time_s": decode_time_s,
        "observed_decode_span_s": (
            round(observed_decode_span, 6)
            if observed_decode_span is not None
            else None
        ),
        "generation_frames": generation_frames,
        "decode_tps_measurable": decode_tps_measurable,
        "decode_tps_exclusion_reason": decode_tps_exclusion_reason,
        # reasoning-model visibility
        "reasoning_chars": reasoning_chars,
        "content_chars": content_chars,
        "reasoned": reasoning_chars > 0,
        "token_source": token_source,
        "finish_reason": finish_reason,
        "truncated": truncated,
        "think_kwargs": dict(THINK_KWARGS) or None,
        "request_id": request_id,
        "cache_mode": cache_mode,
        "prompt_sha256": prompt_sha256,
    }
    if error_msg is not None:
        out["error"] = error_msg
    if first_content_time is not None:
        rounded_content_ttft = round(first_content_time - start, 3)
        out["ttft_content_s"] = rounded_content_ttft
        out["client_ttft_content_s"] = rounded_content_ttft
    if reasoning_chars and (reasoning_chars + content_chars):
        out["reasoning_share"] = round(
            reasoning_chars / (reasoning_chars + content_chars), 3
        )
    if usage is not None:
        out["usage_prompt_tokens"] = usage.get("prompt_tokens")
        if "completion_tokens" in usage:
            out["usage_completion_tokens"] = usage.get("completion_tokens")
    if cached_prompt_tokens is not None:
        out["cached_prompt_tokens"] = cached_prompt_tokens
    return out


def _avg(values, ndigits=2):
    """Return rounded arithmetic mean of non-None values, or None if empty."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), ndigits)


def summarize_context_runs(ctx_size, iter_results, requested_prompt_tokens=None):
    """Aggregate per-iteration records into a cohort-aware summary row."""
    completed_normal = [
        r
        for r in iter_results
        if r.get("run_status") == "completed"
        and not r.get("truncated")
        and not r.get("prompt_token_mismatch")
        and not r.get("error")
    ]
    decode_measured = [
        r
        for r in completed_normal
        if r.get("decode_tps_measurable")
        and r.get("tokens_per_sec") is not None
        and r.get("tokens_per_sec") > 0
    ]
    qualifying = [
        r for r in decode_measured if r.get("qualifies_for_recommendation")
    ]

    length_limited = [r for r in iter_results if r.get("run_status") == "length_limited"]
    no_output = [r for r in iter_results if r.get("run_status") == "no_output"]
    incomplete = [r for r in iter_results if r.get("run_status") == "incomplete"]
    error_runs = [r for r in iter_results if r.get("run_status") == "error"]
    other_finish = [r for r in iter_results if r.get("run_status") == "other_finish"]
    token_mismatch = [r for r in iter_results if r.get("prompt_token_mismatch")]
    missing_prompt_usage = [
        r
        for r in iter_results
        if not r.get("error") and _is_missing_prompt_usage(r)
    ]
    verified_prompt_runs = [
        r
        for r in completed_normal
        if not _is_missing_prompt_usage(r)
        and r.get("prompt_tokens_verified", True) is not False
    ]
    failures = [r for r in iter_results if r.get("error")]

    observed_prompt_avg = _avg(
        [r.get("observed_prompt_tokens") for r in iter_results], ndigits=1
    )
    observed_output_avg = _avg(
        [
            None if _is_contradictory_completion_usage(r) else r.get("observed_output_tokens")
            for r in iter_results
        ],
        ndigits=1,
    )
    contradictory_completion_usage = [
        r
        for r in iter_results
        if not r.get("error") and _is_contradictory_completion_usage(r)
    ]
    observed_totals = [_extract_observed_total_tokens(r) for r in iter_results]
    valid_observed_totals = [t for t in observed_totals if t is not None]
    avg_observed_total_tokens = _avg(valid_observed_totals, ndigits=1)
    max_observed_total_tokens = (
        max(valid_observed_totals) if valid_observed_totals else None
    )

    total_overrun = [
        r
        for r in iter_results
        if not r.get("error")
        and (
            r.get("total_token_overrun") is True
            or r.get("effective_total_within_tier") is False
            or (
                _extract_observed_total_tokens(r) is not None
                and _extract_observed_total_tokens(r) > ctx_size
            )
        )
    ]
    missing_total_usage = [
        r
        for r in iter_results
        if not r.get("error") and _extract_observed_total_tokens(r) is None
    ]
    completed_with_total = [
        r for r in completed_normal if _extract_observed_total_tokens(r) is not None
    ]
    if total_overrun:
        effective_total_within_tier = False
    elif completed_with_total and len(missing_total_usage) == 0:
        effective_total_within_tier = True
    else:
        effective_total_within_tier = None

    prompt_tokens_verified = (
        len(missing_prompt_usage) == 0
        and len(token_mismatch) == 0
        and len(verified_prompt_runs) > 0
    )
    prompt_size_basis = (
        "server_usage"
        if (observed_prompt_avg is not None and len(missing_prompt_usage) == 0)
        else "heuristic_estimate"
    )

    decode_exclusions = Counter(
        r["decode_tps_exclusion_reason"]
        for r in iter_results
        if not r.get("decode_tps_measurable") and r.get("decode_tps_exclusion_reason")
    )

    avg_host_pageouts = _avg(
        [
            r.get("host_pageouts_delta", r.get("pageouts_delta"))
            for r in completed_normal
        ],
        ndigits=1,
    )

    truncated_cohort = None
    if length_limited:
        truncated_cohort = {
            "runs": len(length_limited),
            "avg_ttft_s": _avg([r.get("ttft_s") for r in length_limited], 3),
            "avg_tps": _avg(
                [
                    r.get("tokens_per_sec")
                    for r in length_limited
                    if r.get("decode_tps_measurable")
                ],
                2,
            ),
            "avg_e2e_tps": _avg(
                [r.get("end_to_end_tokens_per_sec") for r in length_limited], 2
            ),
            "avg_total_s": _avg([r.get("total_s") for r in length_limited], 3),
        }

    return {
        "context_size": ctx_size,
        "requested_prompt_tokens": (
            requested_prompt_tokens
            if requested_prompt_tokens is not None
            else ctx_size - OUTPUT_TOKENS
        ),
        "observed_prompt_tokens_avg": observed_prompt_avg,
        "observed_output_tokens_avg": observed_output_avg,
        "avg_observed_total_tokens": avg_observed_total_tokens,
        "max_observed_total_tokens": max_observed_total_tokens,
        "effective_total_within_tier": effective_total_within_tier,
        "prompt_tokens_verified": prompt_tokens_verified,
        "prompt_size_basis": prompt_size_basis,
        "verified_prompt_runs": len(verified_prompt_runs),
        "unverified_prompt_runs": len(missing_prompt_usage),
        "total_within_tier_runs": sum(
            1
            for r in completed_with_total
            if _extract_observed_total_tokens(r) <= ctx_size
        ),
        "total_overrun_runs": len(total_overrun),
        "avg_ttft_s": _avg([r.get("ttft_s") for r in completed_normal], 3),
        "avg_tps": _avg([r.get("tokens_per_sec") for r in decode_measured], 2),
        "avg_e2e_tps": _avg(
            [r.get("end_to_end_tokens_per_sec") for r in completed_normal], 2
        ),
        "avg_total_s": _avg([r.get("total_s") for r in completed_normal], 3),
        "avg_pageouts": avg_host_pageouts,
        "avg_host_pageouts": avg_host_pageouts,
        "total_runs": len(iter_results),
        "runs": len(completed_normal),
        "qualifying_runs": len(qualifying),
        "excluded_runs": len(iter_results) - len(qualifying),
        "failures": len(failures),
        "decode_tps_measured_runs": len(decode_measured),
        "decode_tps_excluded_runs": len(iter_results) - len(decode_measured),
        "decode_tps_exclusions": dict(decode_exclusions),
        "cohorts": {
            "completed": len(completed_normal),
            "qualifying": len(qualifying),
            "length_limited": len(length_limited),
            "no_output": len(no_output),
            "incomplete": len(incomplete),
            "error": len(error_runs),
            "other_finish": len(other_finish),
            "token_mismatch": len(token_mismatch),
            "missing_prompt_token_usage": len(missing_prompt_usage),
            "total_context_overrun": len(total_overrun),
            "missing_total_token_usage": len(missing_total_usage),
            "contradictory_completion_token_usage": len(contradictory_completion_usage),
        },
        "truncated_cohort": truncated_cohort,
    }


def evaluate_summary(
    results, *, allow_unverified_usage=False, strict_total_context=False
):
    """Evaluate summary rows and determine the safe context-window recommendation.

    Only rows with strictly positive measurable decode TPS, verified server
    `usage.prompt_tokens` within tolerance (unless `allow_unverified_usage=True`
    is explicitly requested, in which case recommendations are clearly tagged
    `unverified`), no length truncation, no swap pressure, and <= 15% decode TPS
    drop from the first qualifying baseline can qualify as a recommended safe
    context.

    Prompt-tier fidelity (`prompt_tokens_verified`) is tracked separately from
    combined prompt+output containment (`effective_total_within_tier`). When
    `strict_total_context=True`, tiers whose observed prompt+completion total
    exceeds `context_size` (or lacks server-reported total usage) are rejected
    from qualification (`TOTAL_OVER` / `UNVERIFIED`).
    """
    baseline_tps = None
    recommended = None
    recommended_verified = False
    recommended_nominal_context_tokens = None
    recommended_observed_prompt_tokens = None
    recommended_observed_total_tokens = None
    recommended_effective_total_within_tier = None
    swap_telemetry_available = False
    unverified_tiers = []
    unverified_total_tiers = []
    contradictory_completion_tiers = []
    total_overrun_tiers = []
    row_evaluations = {}

    for label, data in results.items():
        cohorts = data.get("cohorts") or {}
        qualifying_runs = data.get("qualifying_runs", 0)
        avg_tps = data.get("avg_tps")
        avg_pageouts = data.get("avg_host_pageouts", data.get("avg_pageouts"))
        if avg_pageouts is not None:
            swap_telemetry_available = True

        context_size = data.get("context_size")
        avg_obs_prompt = data.get("observed_prompt_tokens_avg")
        avg_obs_total = data.get("avg_observed_total_tokens")
        eff_total_within = data.get("effective_total_within_tier")
        if (
            eff_total_within is None
            and avg_obs_total is not None
            and context_size is not None
        ):
            eff_total_within = avg_obs_total <= context_size

        has_missing_usage = cohorts.get("missing_prompt_token_usage", 0) > 0
        has_unverified_completed = (
            data.get("runs", 0) > 0 and data.get("prompt_tokens_verified") is False
        )
        row_unverified = has_missing_usage or has_unverified_completed
        if row_unverified:
            unverified_tiers.append(label)

        has_total_overrun = (
            cohorts.get("total_context_overrun", 0) > 0
            or eff_total_within is False
        )
        if has_total_overrun:
            total_overrun_tiers.append(label)

        has_contradictory_completion = (
            cohorts.get("contradictory_completion_token_usage", 0) > 0
        )
        if has_contradictory_completion:
            contradictory_completion_tiers.append(label)

        has_missing_total_strict = strict_total_context and (
            cohorts.get("missing_total_token_usage", 0) > 0
            or has_contradictory_completion
            or (data.get("runs", 0) > 0 and eff_total_within is None)
        )
        if has_missing_total_strict:
            unverified_total_tiers.append(label)
        blocked_by_strict_total = strict_total_context and (
            has_total_overrun or has_missing_total_strict
        )

        if (
            qualifying_runs == 0
            or avg_tps is None
            or avg_tps <= 0
            or (row_unverified and not allow_unverified_usage)
            or blocked_by_strict_total
        ):
            if cohorts.get("token_mismatch", 0) > 0 and cohorts.get("length_limited", 0) > 0:
                status = "TRUNC+SIZE"
            elif cohorts.get("token_mismatch", 0) > 0:
                status = "SIZE_MISMATCH"
            elif cohorts.get("length_limited", 0) > 0:
                status = "TRUNCATED"
            elif data.get("runs", 0) > 0 and avg_tps is None:
                status = "NO_DECODE"
            elif strict_total_context and has_total_overrun:
                status = "TOTAL_OVER"
            elif row_unverified or has_missing_total_strict:
                status = "UNVERIFIED"
            elif (
                cohorts.get("no_output", 0) > 0
                and cohorts.get("incomplete", 0) == 0
                and cohorts.get("error", 0) == 0
            ):
                status = "NO_OUTPUT"
            else:
                status = "FAIL"

            row_evaluations[label] = {
                "status": status,
                "qualifies": False,
                "verified": not (row_unverified or has_missing_total_strict),
                "effective_total_within_tier": eff_total_within,
                "tps_drop_pct": None,
            }
            continue

        if baseline_tps is None:
            baseline_tps = avg_tps

        tps_drop = round((1 - avg_tps / baseline_tps) * 100, 1)
        swap_detected = (
            avg_pageouts is not None and avg_pageouts > SWAP_PAGEOUT_THRESHOLD
        )

        if swap_detected:
            status = "SWAP"
            qualifies = False
        elif cohorts.get("token_mismatch", 0) > 0 and cohorts.get("length_limited", 0) > 0:
            status = "TRUNC+SIZE"
            qualifies = False
        elif cohorts.get("token_mismatch", 0) > 0:
            status = "SIZE_MISMATCH"
            qualifies = False
        elif cohorts.get("length_limited", 0) > 0:
            status = "TRUNCATED"
            qualifies = False
        elif tps_drop > TPS_DROP_THRESHOLD_PCT:
            status = f"-{tps_drop:.0f}%"
            qualifies = False
        else:
            if row_unverified:
                status = "OK (unverified)"
            elif has_total_overrun:
                status = "OK (tot>tier)"
            else:
                status = "OK"
            qualifies = True
            recommended = label
            recommended_verified = not row_unverified
            recommended_nominal_context_tokens = context_size
            recommended_observed_prompt_tokens = avg_obs_prompt
            recommended_observed_total_tokens = avg_obs_total
            recommended_effective_total_within_tier = eff_total_within

        row_evaluations[label] = {
            "status": status,
            "qualifies": qualifies,
            "verified": not row_unverified,
            "effective_total_within_tier": eff_total_within,
            "tps_drop_pct": tps_drop,
        }

    swap_note = (
        "no measured host swap pressure"
        if swap_telemetry_available
        else "host swap telemetry unavailable (not verified)"
    )
    if recommended and recommended_verified:
        recommendation_status = "verified"
        if recommended_effective_total_within_tier is True:
            verification_scope = "verified_prompt_and_total_within_tier"
            obs_detail = ""
            if (
                recommended_observed_prompt_tokens is not None
                and recommended_observed_total_tokens is not None
                and recommended_nominal_context_tokens is not None
            ):
                obs_detail = (
                    f" (observed avg prompt={recommended_observed_prompt_tokens:.0f}, "
                    f"total={recommended_observed_total_tokens:.0f} <= "
                    f"{recommended_nominal_context_tokens})"
                )
            note = (
                f"Highest verified qualifying nominal tier ({recommended}) with "
                f"server-reported prompt_tokens within tolerance, combined "
                f"prompt+completion total within nominal tier{obs_detail}, "
                f"<={TPS_DROP_THRESHOLD_PCT:.0f}% streaming decode TPS drop, and "
                f"{swap_note}."
            )
        elif recommended_effective_total_within_tier is False:
            verification_scope = "verified_prompt_tier_only"
            obs_detail = (
                f"observed avg prompt={recommended_observed_prompt_tokens:.0f}, "
                f"total={recommended_observed_total_tokens:.0f} exceeded nominal "
                f"{recommended_nominal_context_tokens}"
                if (
                    recommended_observed_prompt_tokens is not None
                    and recommended_observed_total_tokens is not None
                    and recommended_nominal_context_tokens is not None
                )
                else "observed prompt+completion total exceeded the nominal context size"
            )
            note = (
                f"Highest qualifying nominal tier ({recommended}) with "
                f"server-reported prompt_tokens within ±{PROMPT_TOKEN_TOLERANCE:.0%} "
                f"tolerance, <={TPS_DROP_THRESHOLD_PCT:.0f}% streaming decode TPS "
                f"drop, and {swap_note}. Combined total note: {obs_detail} "
                "(prompt-tier verified only, not an exact <=nominal-tier total run; "
                "pass --strict-total-context to require observed total <= nominal tier)."
            )
        else:
            verification_scope = "verified_prompt_only_total_unreported"
            detail = (
                "server reported contradictory completion_tokens <= 0 despite "
                "non-empty output; combined total unverified"
                if recommended in contradictory_completion_tiers
                else "combined prompt+completion total not reported by server"
            )
            note = (
                f"Highest verified qualifying nominal tier ({recommended}) with "
                f"server-reported prompt_tokens within tolerance, "
                f"<={TPS_DROP_THRESHOLD_PCT:.0f}% streaming decode TPS drop, and "
                f"{swap_note} ({detail})."
            )
    elif recommended:
        recommendation_status = "unverified"
        verification_scope = "unverified_heuristic"
        note = (
            f"UNVERIFIED: server omitted usage.prompt_tokens for {recommended}; "
            f"context size is an approximate heuristic estimate (~{CHARS_PER_TOKEN} "
            f"chars/token), not a verified token count. Selected only because "
            f"--allow-unverified-usage was enabled (<={TPS_DROP_THRESHOLD_PCT:.0f}% "
            f"streaming decode TPS drop and {swap_note})."
        )
    else:
        recommendation_status = "none"
        verification_scope = "none"
        if strict_total_context and total_overrun_tiers:
            over_str = ", ".join(total_overrun_tiers)
            note = (
                f"No tested context size met strict-total-context qualification "
                f"criteria. Observed prompt+completion totals exceeded nominal "
                f"context size on tier(s) [{over_str}] (--strict-total-context "
                f"requires observed_prompt_tokens + observed_output_tokens <= "
                f"requested_context_tokens)."
            )
        elif strict_total_context and unverified_total_tiers:
            unv_tot_str = ", ".join(unverified_total_tiers)
            if contradictory_completion_tiers:
                contra_str = ", ".join(contradictory_completion_tiers)
                note = (
                    f"No tested context size met strict-total-context qualification "
                    f"criteria. Server reported contradictory completion_tokens <= 0 "
                    f"despite non-empty generated output on tier(s) [{contra_str}] "
                    f"(unverified total usage on [{unv_tot_str}]; --strict-total-context "
                    f"requires verified server prompt_tokens and completion_tokens)."
                )
            else:
                note = (
                    f"No tested context size met strict-total-context qualification "
                    f"criteria. Server omitted verified combined prompt+completion "
                    f"token usage on tier(s) [{unv_tot_str}] (--strict-total-context "
                    f"requires verified observed_prompt_tokens + observed_output_tokens "
                    f"<= requested_context_tokens)."
                )
        elif unverified_tiers:
            tiers_str = ", ".join(unverified_tiers)
            note = (
                f"No tested context size met verified qualification criteria. "
                f"Server omitted usage.prompt_tokens on tier(s) [{tiers_str}], so "
                f"actual prompt token counts could not be verified (pass "
                f"--allow-unverified-usage to opt in to unverified heuristic-size "
                f"recommendations)."
            )
        else:
            note = (
                "No tested context size met all qualification criteria "
                "(normal completion, verified prompt-token tolerance, measurable "
                "streaming decode TPS, and no swap/throughput cliff)."
            )

    return {
        "recommended_context": recommended,
        "recommended_nominal_tier": recommended,
        "recommended_nominal_context_tokens": recommended_nominal_context_tokens,
        "recommended_observed_prompt_tokens": recommended_observed_prompt_tokens,
        "recommended_observed_total_tokens": recommended_observed_total_tokens,
        "recommended_effective_total_within_tier": recommended_effective_total_within_tier,
        "recommended_context_verified": recommended_verified,
        "recommendation_status": recommendation_status,
        "verification_scope": verification_scope,
        "unverified": bool(recommended and not recommended_verified),
        "allow_unverified_usage": allow_unverified_usage,
        "strict_total_context": strict_total_context,
        "unverified_tiers": unverified_tiers,
        "total_overrun_tiers": total_overrun_tiers,
        "baseline_tps": baseline_tps,
        "tps_drop_threshold_pct": TPS_DROP_THRESHOLD_PCT,
        "swap_pageout_threshold": SWAP_PAGEOUT_THRESHOLD,
        "swap_telemetry_available": swap_telemetry_available,
        "note": note,
        "rows": row_evaluations,
    }


def _fmt_opt(val, fmt, empty="—"):
    return empty if val is None else fmt.format(val)


def run_benchmark(
    api_url,
    iterations,
    context_sizes,
    model_name,
    results_file,
    *,
    cache_mode="cold",
    prompt_token_tolerance=PROMPT_TOKEN_TOLERANCE,
    min_decode_interval_s=MIN_DECODE_INTERVAL_S,
    allow_unverified_usage=False,
    strict_total_context=False,
):
    global API_URL, MODEL_NAME
    API_URL = api_url
    MODEL_NAME = model_name

    endpoint_host = (urlparse(api_url).hostname or "").lower() or None
    endpoint_is_local = is_local_endpoint(api_url)
    host_info = get_benchmark_host_info()
    sweep_id = uuid.uuid4().hex[:8]

    results = {}
    all_raw = []
    any_host_memory_sampled = False

    usage_policy = (
        "unverified allowed (opt-in --allow-unverified-usage)"
        if allow_unverified_usage
        else "verified server usage.prompt_tokens required"
    )
    total_policy = (
        "strict (observed prompt+output <= nominal tier)"
        if strict_total_context
        else "advisory (prompt-tier tolerance; --strict-total-context for exact total <= tier)"
    )
    print(f"\n{'='*106}")
    print("  Context Window Benchmark")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  API: {api_url} ({'local' if endpoint_is_local else 'remote'} endpoint)")
    print(
        f"  Runner host: {host_info['platform']} {host_info['machine']} "
        f"(Python {host_info['python_version']})"
    )
    print(f"  Model: {model_name}")
    print(f"  Context sizes: {[f'{s//1024}K' for s in context_sizes]}")
    print(
        f"  Iterations: {iterations}, Output tokens: {OUTPUT_TOKENS}, "
        f"Cache mode: {cache_mode}, Prompt tol: ±{prompt_token_tolerance:.0%}"
    )
    print(f"  Prompt size verification: {usage_policy}")
    print(f"  Total context gating: {total_policy}")
    print(f"{'='*106}\n")

    for ctx_size in context_sizes:
        ctx_label = f"{ctx_size // 1024}K"
        input_tokens = ctx_size - OUTPUT_TOKENS

        print(f"--- {ctx_label} context ({input_tokens} target input tokens) ---")
        iter_results = []

        for i in range(iterations):
            request_id = f"{sweep_id}-{ctx_label}-iter{i+1}"
            messages = build_prompt(
                input_tokens, request_id=request_id, cache_mode=cache_mode
            )

            pageouts_before, mem_free_before = sample_host_memory(endpoint_is_local)

            print(f"  Run {i+1}/{iterations} [{request_id}]...", end=" ", flush=True)
            result = run_single_test(
                messages,
                ctx_label,
                model_name,
                requested_context_tokens=ctx_size,
                requested_prompt_tokens=input_tokens,
                requested_output_tokens=OUTPUT_TOKENS,
                request_id=request_id,
                cache_mode=cache_mode,
                prompt_token_tolerance=prompt_token_tolerance,
                min_decode_interval_s=min_decode_interval_s,
                allow_unverified_usage=allow_unverified_usage,
                strict_total_context=strict_total_context,
            )

            pageouts_after, mem_free_after = sample_host_memory(endpoint_is_local)
            if pageouts_before is not None and pageouts_after is not None:
                pageouts_delta = pageouts_after - pageouts_before
                any_host_memory_sampled = True
            else:
                pageouts_delta = None

            if mem_free_before is not None or mem_free_after is not None:
                any_host_memory_sampled = True

            result["host_pageouts_delta"] = pageouts_delta
            result["pageouts_delta"] = pageouts_delta
            result["host_mem_free_pct_before"] = mem_free_before
            result["host_mem_free_pct_after"] = mem_free_after
            result["mem_free_pct_before"] = mem_free_before
            result["mem_free_pct_after"] = mem_free_after
            result["context_size"] = ctx_size
            result["iteration"] = i + 1

            if result.get("error"):
                print(f"FAILED ({result['run_status']}): {result['error']}")
            else:
                _think = ""
                if result.get("reasoned"):
                    _think = f"  think={result.get('reasoning_share', 0):.0%}"
                    if result.get("ttft_content_s") is not None:
                        _think += f" (ttft_content={result['ttft_content_s']:.2f}s)"
                _flags = []
                if result.get("truncated"):
                    _flags.append("TRUNCATED")
                if result.get("prompt_token_mismatch"):
                    _flags.append(
                        f"SIZE_MISMATCH(obs={result.get('observed_prompt_tokens')})"
                    )
                if result.get("observed_prompt_tokens") is None:
                    _flags.append(
                        f"UNVERIFIED_SIZE(est~{result.get('requested_prompt_tokens')})"
                    )
                if result.get("contradictory_completion_usage"):
                    _flags.append("UNVERIFIED_OUTPUT(completion_tokens<=0)")
                if result.get("total_token_overrun"):
                    _flags.append(
                        f"TOTAL_OVER(obs={result.get('observed_total_tokens')}/{ctx_size})"
                    )
                if not result.get("decode_tps_measurable"):
                    _flags.append(
                        f"DECODE_NA({result.get('decode_tps_exclusion_reason')})"
                    )
                _flag_str = ("  " + " ".join(_flags)) if _flags else ""
                tps_str = (
                    f"{result['tokens_per_sec']:.1f}"
                    if result["tokens_per_sec"] is not None
                    else "N/A"
                )
                e2e_str = (
                    f"{result['end_to_end_tokens_per_sec']:.1f}"
                    if result["end_to_end_tokens_per_sec"] is not None
                    else "N/A"
                )
                pg_str = "N/A" if pageouts_delta is None else str(pageouts_delta)
                mem_str = "N/A" if mem_free_after is None else f"{mem_free_after}%"
                print(
                    f"TTFT={_fmt_opt(result['ttft_s'], '{:.2f}s', 'N/A')}  "
                    f"decode_TPS={tps_str}  "
                    f"e2e_TPS={e2e_str}  "
                    f"total={_fmt_opt(result['total_s'], '{:.2f}s', 'N/A')}  "
                    f"host_pageouts={pg_str}  "
                    f"host_mem_free={mem_str}"
                    f"{_think}{_flag_str}"
                )

            iter_results.append(result)
            all_raw.append(result)

        results[ctx_label] = summarize_context_runs(
            ctx_size, iter_results, requested_prompt_tokens=input_tokens
        )
        print()

    evaluation = evaluate_summary(
        results,
        allow_unverified_usage=allow_unverified_usage,
        strict_total_context=strict_total_context,
    )
    for label, row_eval in evaluation["rows"].items():
        results[label]["status"] = row_eval["status"]
        results[label]["qualifies"] = row_eval["qualifies"]
        results[label]["verified"] = row_eval["verified"]
        results[label]["tps_drop_pct"] = row_eval["tps_drop_pct"]

    if not endpoint_is_local:
        mem_telemetry = {
            "applicable": False,
            "scope": "unavailable",
            "reason": "remote_endpoint_not_sampled",
        }
    elif any_host_memory_sampled:
        mem_telemetry = {
            "applicable": True,
            "scope": "local_macos_host_pre_post_snapshots",
            "reason": None,
        }
    else:
        mem_telemetry = {
            "applicable": False,
            "scope": "unavailable",
            "reason": "macos_counters_unavailable",
        }

    # Save raw results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "sweep_id": sweep_id,
            "api_url": api_url,
            "endpoint_host": endpoint_host,
            "endpoint_is_local": endpoint_is_local,
            "benchmark_host": host_info,
            "model": model_name,
            "iterations": iterations,
            "output_tokens": OUTPUT_TOKENS,
            "context_sizes": context_sizes,
            "cache_mode": cache_mode,
            "prompt_token_tolerance": prompt_token_tolerance,
            "min_decode_interval_s": min_decode_interval_s,
            "allow_unverified_usage": allow_unverified_usage,
            "require_prompt_token_usage": not allow_unverified_usage,
            "strict_total_context": strict_total_context,
            "think_kwargs": dict(THINK_KWARGS) or None,
            "memory_telemetry": mem_telemetry,
        },
        "recommendation": {
            "recommended_context": evaluation["recommended_context"],
            "recommended_nominal_tier": evaluation["recommended_nominal_tier"],
            "recommended_nominal_context_tokens": evaluation[
                "recommended_nominal_context_tokens"
            ],
            "recommended_observed_prompt_tokens": evaluation[
                "recommended_observed_prompt_tokens"
            ],
            "recommended_observed_total_tokens": evaluation[
                "recommended_observed_total_tokens"
            ],
            "recommended_effective_total_within_tier": evaluation[
                "recommended_effective_total_within_tier"
            ],
            "recommended_context_verified": evaluation["recommended_context_verified"],
            "recommendation_status": evaluation["recommendation_status"],
            "verification_scope": evaluation["verification_scope"],
            "unverified": evaluation["unverified"],
            "allow_unverified_usage": evaluation["allow_unverified_usage"],
            "strict_total_context": evaluation["strict_total_context"],
            "unverified_tiers": evaluation["unverified_tiers"],
            "total_overrun_tiers": evaluation["total_overrun_tiers"],
            "baseline_tps": evaluation["baseline_tps"],
            "tps_drop_threshold_pct": evaluation["tps_drop_threshold_pct"],
            "swap_pageout_threshold": evaluation["swap_pageout_threshold"],
            "swap_telemetry_available": evaluation["swap_telemetry_available"],
            "note": evaluation["note"],
        },
        "summary": results,
        "raw": all_raw,
    }
    results_file.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Raw results saved to {results_file}\n")

    # Summary table
    print(f"{'='*106}")
    print("  SUMMARY")
    print(f"{'='*106}")
    print(
        f"  {'Context':>7}  {'ObsIn':>7}  {'ObsTot':>7}  {'TTFT':>7}  {'DecTPS':>7}  "
        f"{'E2ETPS':>7}  {'Total':>7}  {'Pageouts':>8}  {'Qual':>5}  {'Status':>15}"
    )
    print(
        f"  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  "
        f"{'─'*7}  {'─'*7}  {'─'*8}  {'─'*5}  {'─'*15}"
    )

    total_exclusions = Counter()
    total_disqualified_cohorts = Counter()
    any_unverified_row = False
    any_total_overrun_ok = False

    cohort_keys = [
        "length_limited",
        "no_output",
        "incomplete",
        "error",
        "other_finish",
        "token_mismatch",
        "missing_prompt_token_usage",
    ]
    if strict_total_context:
        cohort_keys.extend(
            [
                "total_context_overrun",
                "missing_total_token_usage",
                "contradictory_completion_token_usage",
            ]
        )

    for label, data in results.items():
        status = data["status"]
        cohorts = data.get("cohorts") or {}
        is_unverified_row = cohorts.get("missing_prompt_token_usage", 0) > 0
        is_unverified_total = (
            cohorts.get("contradictory_completion_token_usage", 0) > 0
            or (strict_total_context and cohorts.get("missing_total_token_usage", 0) > 0)
        )
        if is_unverified_row:
            any_unverified_row = True
        if status == "OK (tot>tier)":
            any_total_overrun_ok = True
        display_label = f"~{label}" if is_unverified_row else label
        obs_in_empty = "unverif" if is_unverified_row else "—"
        obs_in_str = _fmt_opt(
            data.get("observed_prompt_tokens_avg"), "{:.0f}", empty=obs_in_empty
        )
        obs_tot_empty = "unverif" if (is_unverified_row or is_unverified_total) else "—"
        obs_tot_str = _fmt_opt(
            data.get("avg_observed_total_tokens"), "{:.0f}", empty=obs_tot_empty
        )
        qual_str = f"{data['qualifying_runs']}/{data['total_runs']}"
        ttft_str = _fmt_opt(data["avg_ttft_s"], "{:.2f}s")
        dec_empty = "N/A" if data["runs"] > 0 else "—"
        dec_str = _fmt_opt(data["avg_tps"], "{:.1f}", empty=dec_empty)
        e2e_str = _fmt_opt(data["avg_e2e_tps"], "{:.1f}")
        tot_str = _fmt_opt(data["avg_total_s"], "{:.2f}s")
        pg_str = _fmt_opt(data["avg_pageouts"], "{:.0f}", empty="N/A")

        print(
            f"  {display_label:>7}  {obs_in_str:>7}  {obs_tot_str:>7}  {ttft_str:>7}  "
            f"{dec_str:>7}  {e2e_str:>7}  {tot_str:>7}  {pg_str:>8}  "
            f"{qual_str:>5}  {status:>15}"
        )

        total_exclusions.update(data.get("decode_tps_exclusions") or {})
        for key in cohort_keys:
            count = cohorts.get(key, 0)
            if count:
                total_disqualified_cohorts[key] += count

    print(
        f"  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  "
        f"{'─'*7}  {'─'*7}  {'─'*8}  {'─'*5}  {'─'*15}"
    )

    if any_unverified_row:
        print(
            "  Note: '~' context sizes and ObsIn/ObsTot=unverif indicate approximate "
            "heuristic prompt sizes (server omitted usage.prompt_tokens)."
        )
    if any_total_overrun_ok:
        print(
            "  Note: 'OK (tot>tier)' means prompt_tokens passed tolerance, but "
            "observed prompt+completion total (ObsTot) exceeded the nominal context "
            "tier (pass --strict-total-context to require ObsTot <= nominal tier)."
        )
    if total_exclusions:
        excl_fmt = ", ".join(f"{k}={v}" for k, v in sorted(total_exclusions.items()))
        print(f"  Decode TPS exclusions: {excl_fmt}")
    if total_disqualified_cohorts:
        coh_fmt = ", ".join(
            f"{k}={v}" for k, v in sorted(total_disqualified_cohorts.items())
        )
        print(f"  Non-qualifying cohorts: {coh_fmt}")
    if not mem_telemetry["applicable"]:
        print(f"  Host memory/swap telemetry: unavailable ({mem_telemetry['reason']})")

    # Recommendation
    print()
    if evaluation["recommended_context"]:
        if evaluation["recommended_context_verified"]:
            obs_p = evaluation["recommended_observed_prompt_tokens"]
            obs_t = evaluation["recommended_observed_total_tokens"]
            nom_t = evaluation["recommended_nominal_context_tokens"]
            within = evaluation["recommended_effective_total_within_tier"]
            if obs_p is not None and obs_t is not None and nom_t is not None:
                budget_tag = (
                    f"<= {nom_t} within tier"
                    if within
                    else f"> {nom_t} EXCEEDS nominal tier"
                )
                detail_str = (
                    f" (nominal tier {evaluation['recommended_nominal_tier']}; "
                    f"obs prompt={obs_p:.0f}, obs total={obs_t:.0f} {budget_tag})"
                )
            else:
                detail_str = ""
            print(
                f"  RECOMMENDATION: {evaluation['recommended_context']} context "
                f"window{detail_str}"
            )
        else:
            print(
                f"  RECOMMENDATION: ~{evaluation['recommended_context']} context window "
                "[UNVERIFIED — missing server usage.prompt_tokens]"
            )
        print(f"  {evaluation['note']}")
    else:
        print(
            "  RECOMMENDATION: None — no tested context size met verified "
            "qualification criteria."
        )
        print(f"  {evaluation['note']}")
    print()
    return output


# ---------------------------------------------------------------------------
# Synthetic mock-SSE server for reproducible offline demos and integration tests
# ---------------------------------------------------------------------------

def _estimate_demo_context_k(body):
    """Map incoming prompt character length to the synthetic context tier in K."""
    messages = body.get("messages") or []
    total_chars = sum(len(m.get("content") or "") for m in messages)
    approx_tokens = total_chars / CHARS_PER_TOKEN + OUTPUT_TOKENS
    for tier in (4, 8, 16, 32, 64, 96, 128):
        if approx_tokens <= tier * 1024 * 1.25:
            return tier
    return 128


def _make_mock_sse_handler(scenarios=None):
    """Build an HTTP handler serving OpenAI-compatible SSE streams for demo/testing."""

    class MockSSEHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Keep demo/test output clean.
            return

        def do_POST(self):
            if not self.path.endswith("/chat/completions"):
                self.send_response(404)
                self.end_headers()
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw_body.decode("utf-8"))
            tier_k = _estimate_demo_context_k(body)
            target_prompt_tokens = tier_k * 1024 - int(body.get("max_tokens", OUTPUT_TOKENS))

            if scenarios and tier_k in scenarios:
                spec = scenarios[tier_k]
            else:
                spec = _default_demo_scenario(tier_k, target_prompt_tokens)

            status_code = spec.get("status_code", 200)
            if status_code != 200:
                payload = spec.get("error_body", "synthetic server error").encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            prefill_delay_s = spec.get("prefill_delay_s", 0.02)
            if prefill_delay_s > 0:
                time.sleep(prefill_delay_s)

            frame_delay_s = spec.get("frame_delay_s", 0.02)
            frames = spec.get("frames", [])
            for idx, frame in enumerate(frames):
                if idx > 0 and frame_delay_s > 0:
                    time.sleep(frame_delay_s)
                line = f"data: {json.dumps(frame)}\n\n".encode("utf-8")
                self.wfile.write(line)
                self.wfile.flush()

            if spec.get("include_done", True):
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

    return MockSSEHandler


def _default_demo_scenario(tier_k, target_prompt_tokens):
    """Built-in synthetic scenarios illustrating normal, buffered, truncated, and degraded rows."""
    if tier_k == 4:
        return {
            "prefill_delay_s": 0.03,
            "frame_delay_s": 0.025,
            "frames": [
                {"choices": [{"delta": {"content": "Transformers "}}]},
                {"choices": [{"delta": {"content": "use "}}]},
                {"choices": [{"delta": {"content": "attention "}}]},
                {"choices": [{"delta": {"content": "and "}}]},
                {"choices": [{"delta": {"content": "KV caches."}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": target_prompt_tokens,
                        "completion_tokens": 16,
                    },
                },
            ],
        }
    if tier_k == 8:
        return {
            "prefill_delay_s": 0.05,
            "frame_delay_s": 0.026,
            "frames": [
                {"choices": [{"delta": {"content": "Attention "}}]},
                {"choices": [{"delta": {"content": "scales "}}]},
                {"choices": [{"delta": {"content": "linearly "}}]},
                {"choices": [{"delta": {"content": "with "}}]},
                {"choices": [{"delta": {"content": "context."}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": target_prompt_tokens,
                        "completion_tokens": 16,
                    },
                },
            ],
        }
    if tier_k == 16:
        # Buffered single-frame completion: decode TPS must be N/A, not inflated!
        return {
            "prefill_delay_s": 0.08,
            "frame_delay_s": 0.0,
            "frames": [
                {
                    "choices": [
                        {
                            "delta": {
                                "content": "Buffered single-frame summary delivered all at once."
                            },
                            "finish_reason": "stop",
                        }
                    ]
                },
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": target_prompt_tokens,
                        "completion_tokens": 512,
                    },
                },
            ],
        }
    if tier_k == 32:
        # Truncated at max_tokens + severe prompt-token overrun: excluded from recommendation!
        return {
            "prefill_delay_s": 0.06,
            "frame_delay_s": 0.025,
            "frames": [
                {"choices": [{"delta": {"content": "Truncated "}}]},
                {"choices": [{"delta": {"content": "and "}}]},
                {"choices": [{"delta": {"content": "over-budget "}}]},
                {"choices": [{"delta": {"content": "output..."}}]},
                {"choices": [{"delta": {}, "finish_reason": "length"}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": target_prompt_tokens * 2,
                        "completion_tokens": 512,
                    },
                },
            ],
        }
    # 64K+: degraded streaming decode (>15% drop from baseline)
    return {
        "prefill_delay_s": 0.12,
        "frame_delay_s": 0.060,
        "frames": [
            {"choices": [{"delta": {"content": "Slower "}}]},
            {"choices": [{"delta": {"content": "decode "}}]},
            {"choices": [{"delta": {"content": "at "}}]},
            {"choices": [{"delta": {"content": "large "}}]},
            {"choices": [{"delta": {"content": "context."}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": target_prompt_tokens,
                    "completion_tokens": 16,
                },
            },
        ],
    }


def start_mock_sse_server(scenarios=None):
    """Start an ephemeral local mock-SSE server on 127.0.0.1 and return (server, api_url)."""
    handler = _make_mock_sse_handler(scenarios=scenarios)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}/v1"


def run_demo(
    results_file,
    iterations=1,
    context_sizes=None,
    cache_mode="cold",
    allow_unverified_usage=False,
    strict_total_context=False,
):
    """Run an offline end-to-end synthetic mock-SSE benchmark demo."""
    sizes = context_sizes or [4096, 8192, 16384, 32768, 65536]
    server, mock_url = start_mock_sse_server()
    try:
        return run_benchmark(
            mock_url,
            iterations=iterations,
            context_sizes=sizes,
            model_name="synthetic-mock-sse",
            results_file=results_file,
            cache_mode=cache_mode,
            allow_unverified_usage=allow_unverified_usage,
            strict_total_context=strict_total_context,
        )
    finally:
        server.shutdown()
        server.server_close()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Local LLM context window benchmark")
    parser.add_argument(
        "--api-url", default=API_URL, help=f"API base URL (default: {API_URL})"
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Model name to send in request body (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=f"Runs per size (default: {ITERATIONS}, or 1 in --demo mode)",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Comma-separated context sizes in K, e.g. '4,8,16,32'",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=str(RESULTS_FILE),
        help=f"Path to write JSON results (default: {RESULTS_FILE})",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["cold", "warm"],
        default="cold",
        help="Prompt prefix caching mode: 'cold' prepends a unique request ID per "
        "iteration to defeat prefix caching; 'warm' reuses identical prompts "
        "(default: cold)",
    )
    parser.add_argument(
        "--prompt-token-tolerance",
        type=float,
        default=PROMPT_TOKEN_TOLERANCE,
        help="Allowed relative difference between requested and server-reported "
        f"prompt tokens (default: {PROMPT_TOKEN_TOLERANCE})",
    )
    parser.add_argument(
        "--min-decode-interval",
        type=float,
        default=MIN_DECODE_INTERVAL_S,
        help="Minimum first-to-last generation frame span in seconds required "
        f"to publish streaming decode TPS (default: {MIN_DECODE_INTERVAL_S})",
    )
    parser.add_argument(
        "--allow-unverified-usage",
        action="store_true",
        default=False,
        help="Opt in to allowing tiers where the server omitted usage.prompt_tokens "
        "to qualify for an explicitly tagged UNVERIFIED recommendation based on "
        "heuristic prompt sizing (~3.5 chars/token). Off by default.",
    )
    parser.add_argument(
        "--strict-total-context",
        action="store_true",
        default=False,
        help="Exact-total comparison mode: require server-reported combined "
        "observed_prompt_tokens + observed_output_tokens <= requested_context_tokens "
        "(nominal tier) to qualify. Tiers that exceed the nominal tier budget are "
        "rejected as TOTAL_OVER. Off by default (advisory tracking).",
    )
    parser.add_argument(
        "--think",
        choices=["on", "off"],
        default=None,
        help="Force the model's reasoning phase on or off. Sends the right "
        "chat_template_kwargs key for --think-style. Omit to use the server default "
        "(which DIFFERS per model: DeepSeek/Qwen default off, MiniMax defaults on).",
    )
    parser.add_argument(
        "--think-style",
        choices=["deepseek", "qwen", "minimax"],
        default="deepseek",
        help="Which vendor key --think should set: deepseek/qwen -> enable_thinking, "
        "minimax -> thinking_mode (default: deepseek)",
    )
    parser.add_argument(
        "--think-kwargs",
        type=str,
        default=None,
        help="Raw JSON forwarded as chat_template_kwargs, e.g. "
        "'{\"enable_thinking\": true}'. Overrides --think/--think-style.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a reproducible offline benchmark against a built-in synthetic "
        "mock-SSE server covering streamed, buffered, truncated/mismatched, "
        "and degraded context tiers.",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.think_kwargs:
        THINK_KWARGS = json.loads(args.think_kwargs)
    elif args.think is not None:
        _key = "thinking_mode" if args.think_style == "minimax" else "enable_thinking"
        THINK_KWARGS = {_key: args.think == "on"}
    if THINK_KWARGS:
        print(f"chat_template_kwargs: {THINK_KWARGS}")

    sizes = None
    if args.sizes:
        sizes = [int(s) * 1024 for s in args.sizes.split(",")]

    if args.demo:
        run_demo(
            Path(args.results_file),
            iterations=args.iterations if args.iterations is not None else 1,
            context_sizes=sizes,
            cache_mode=args.cache_mode,
            allow_unverified_usage=args.allow_unverified_usage,
            strict_total_context=args.strict_total_context,
        )
    else:
        run_benchmark(
            args.api_url,
            args.iterations if args.iterations is not None else ITERATIONS,
            sizes if sizes is not None else CONTEXT_SIZES,
            args.model,
            Path(args.results_file),
            cache_mode=args.cache_mode,
            prompt_token_tolerance=args.prompt_token_tolerance,
            min_decode_interval_s=args.min_decode_interval,
            allow_unverified_usage=args.allow_unverified_usage,
            strict_total_context=args.strict_total_context,
        )
