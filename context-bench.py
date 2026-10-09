#!/usr/bin/env python3
"""
Context Window Benchmark for local LLM serving endpoints (MLX, vLLM, llama.cpp).
Tests throughput and memory pressure across context sizes.

Usage: python3 context-bench.py [--api-url URL] [--iterations N] [--model NAME]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8090/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "default")
CONTEXT_SIZES = [4096, 8192, 16384, 32768, 65536, 98304, 131072]
ITERATIONS = 3
OUTPUT_TOKENS = 512
TIMEOUT_S = 600
RESULTS_FILE = Path("./context-bench-results.json")

# Technical documentation filler — repeated to fill context windows.
# Each block is ~500 tokens when tokenized.
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

# ~130 words per block, roughly 170 tokens. We'll estimate 3.5 chars/token.
CHARS_PER_TOKEN = 3.5

# Extra chat-template arguments, forwarded verbatim as `chat_template_kwargs`.
# Reasoning models gate their think phase on a MODEL-SPECIFIC key with a
# MODEL-SPECIFIC default -- there is no portable switch:
#   DeepSeek / Qwen : {"enable_thinking": true|false}   default OFF
#   MiniMax         : {"thinking_mode": true|false}     default ON
# Benchmarking without pinning this compares a thinking run against a non-thinking
# one and calls the difference throughput. Empty dict = server default.
THINK_KWARGS = {}


def get_memory_pressure():
    """Parse macOS memory_pressure output."""
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
    """Get current page-out count from vm_stat."""
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if "Pageouts" in line:
                return int(line.split(":")[1].strip().rstrip("."))
    except Exception:
        pass
    return 0


def build_prompt(target_tokens):
    """Build a prompt padded to approximately target_tokens."""
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    system_msg = "You are a helpful AI assistant. Summarize the key technical concepts from the documentation provided."
    user_prefix = "Please read the following technical documentation carefully, then provide a concise summary of the most important concepts:\n\n"
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


def run_single_test(messages, ctx_label, model_name=None):
    """Send a streaming request and measure prefill/decode behaviour.

    REASONING MODELS (2026-08-19). Servers emit chain-of-thought on a SEPARATE
    delta field -- `reasoning` or `reasoning_content` -- not on `content`. An
    earlier version of this function counted only `content`, which silently broke
    three metrics at once on any reasoning model:

      * output_tokens undercounted by however much the model thought;
      * tokens_per_sec computed over that undercount, so decode throughput read
        far lower than reality;
      * TTFT -- worst of the three -- fired on the first CONTENT token, so the
        entire think phase was billed as prefill latency.

    Measured on deepseek-v4-flash: reasoning_len 9359 vs content_len 537, i.e.
    ~95% of generated tokens were invisible, and TTFT read ~90s instead of ~1s.

    So: TTFT is now the first token of EITHER kind (true prefill), `ttft_content_s`
    is reported alongside it for the answer-latency question, and reasoning and
    content tokens are counted separately.

    Token counts prefer the server's own `usage` block (via stream_options) and
    fall back to counting SSE chunks only when the server does not send one.
    `finish_reason` is captured because a run truncated at max_tokens is not
    comparable to one that stopped on its own.
    """
    body = {
        "model": model_name or MODEL_NAME,
        "messages": messages,
        "max_tokens": OUTPUT_TOKENS,
        "stream": True,
        "temperature": 0.7,
        # Ask for the exact token accounting. Servers that don't support it ignore it.
        "stream_options": {"include_usage": True},
    }
    if THINK_KWARGS:
        body["chat_template_kwargs"] = dict(THINK_KWARGS)

    content_chunks = 0
    reasoning_chunks = 0
    content_chars = 0
    reasoning_chars = 0
    first_any_time = None
    first_content_time = None
    finish_reason = None
    usage = None
    start = time.monotonic()

    try:
        with httpx.Client(timeout=httpx.Timeout(TIMEOUT_S, connect=30)) as client:
            with client.stream(
                "POST", f"{API_URL}/chat/completions", json=body
            ) as resp:
                if resp.status_code != 200:
                    error_body = resp.read().decode()
                    return {"error": f"HTTP {resp.status_code}: {error_body[:200]}"}

                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("usage"):
                        usage = chunk["usage"]

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue           # usage-only frame carries no choices
                    choice = choices[0]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}

                    # Vendors disagree on the field name; accept both.
                    think = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    content = delta.get("content") or ""

                    if (think or content) and first_any_time is None:
                        first_any_time = time.monotonic()
                    if think:
                        reasoning_chunks += 1
                        reasoning_chars += len(think)
                    if content:
                        if first_content_time is None:
                            first_content_time = time.monotonic()
                        content_chunks += 1
                        content_chars += len(content)

    except httpx.TimeoutException:
        return {"error": f"Timeout after {TIMEOUT_S}s"}
    except Exception as e:
        return {"error": str(e)}

    end = time.monotonic()
    total_time = end - start
    ttft = (first_any_time - start) if first_any_time else total_time
    decode_time = (end - first_any_time) if first_any_time else 0

    # Server-reported counts are authoritative; chunk counts are the fallback.
    chunk_total = content_chunks + reasoning_chunks
    output_tokens = (usage or {}).get("completion_tokens") or chunk_total

    # A server that buffers the whole completion into a single SSE frame
    # collapses first_any_time onto `end`. decode_time then goes to ~0 and
    # output_tokens/decode_time explodes into a number that looks like a
    # spectacular throughput result and is really a division artifact.
    # One frame is not a decode curve, so refuse to report a rate at all.
    streamed = chunk_total > 1
    tps = output_tokens / decode_time if (streamed and decode_time > 0) else None

    out = {
        "ttft_s": round(ttft, 2),
        "total_s": round(total_time, 2),
        "output_tokens": output_tokens,
        "tokens_per_sec": round(tps, 2) if tps is not None else None,
        "streamed": streamed,
        "decode_time_s": round(decode_time, 2),
        # reasoning-model visibility
        "reasoning_chars": reasoning_chars,
        "content_chars": content_chars,
        "reasoned": reasoning_chars > 0,
        "token_source": "usage" if (usage or {}).get("completion_tokens") else "sse_chunks",
        "finish_reason": finish_reason,
        # a run cut off at max_tokens is not comparable to one that stopped itself
        "truncated": finish_reason == "length",
        "think_kwargs": dict(THINK_KWARGS) or None,
    }
    if first_content_time:
        out["ttft_content_s"] = round(first_content_time - start, 2)
    if reasoning_chars and (reasoning_chars + content_chars):
        out["reasoning_share"] = round(
            reasoning_chars / (reasoning_chars + content_chars), 3)
    if usage:
        out["usage_prompt_tokens"] = usage.get("prompt_tokens")
    return out


def run_benchmark(api_url, iterations, context_sizes, model_name, results_file):
    global API_URL, MODEL_NAME
    API_URL = api_url
    MODEL_NAME = model_name

    results = {}
    all_raw = []

    print(f"\n{'='*70}")
    print(f"  Context Window Benchmark")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  API: {api_url}")
    print(f"  Model: {model_name}")
    print(f"  Context sizes: {[f'{s//1024}K' for s in context_sizes]}")
    print(f"  Iterations: {iterations}, Output tokens: {OUTPUT_TOKENS}")
    print(f"{'='*70}\n")

    for ctx_size in context_sizes:
        ctx_label = f"{ctx_size // 1024}K"
        input_tokens = ctx_size - OUTPUT_TOKENS
        messages = build_prompt(input_tokens)

        print(f"--- {ctx_label} context ({input_tokens} input tokens) ---")
        iter_results = []

        for i in range(iterations):
            pageouts_before = get_pageout_count()
            mem_free_before = get_memory_pressure()

            print(f"  Run {i+1}/{iterations}...", end=" ", flush=True)
            result = run_single_test(messages, ctx_label, model_name)

            pageouts_after = get_pageout_count()
            mem_free_after = get_memory_pressure()
            pageouts_delta = pageouts_after - pageouts_before

            result["pageouts_delta"] = pageouts_delta
            result["mem_free_pct_before"] = mem_free_before
            result["mem_free_pct_after"] = mem_free_after
            result["context_size"] = ctx_size
            result["iteration"] = i + 1

            if "error" in result:
                print(f"FAILED: {result['error']}")
            else:
                _think = ""
                if result.get("reasoned"):
                    _think = f"  think={result.get('reasoning_share', 0):.0%}"
                    if result.get("ttft_content_s"):
                        _think += f" (ttft_content={result['ttft_content_s']:.1f}s)"
                _trunc = "  TRUNCATED" if result.get("truncated") else ""
                _tps = result["tokens_per_sec"]
                _tps_s = f"{_tps:.1f}" if _tps is not None else "n/a (not streamed)"
                print(
                    f"TTFT={result['ttft_s']:.1f}s  "
                    f"TPS={_tps_s}  "
                    f"total={result['total_s']:.1f}s  "
                    f"pageouts={pageouts_delta}  "
                    f"mem_free={mem_free_after}%"
                    f"{_think}{_trunc}"
                )

            iter_results.append(result)
            all_raw.append(result)

        # Compute averages (excluding errors)
        good = [r for r in iter_results if "error" not in r]
        if good:
            results[ctx_label] = {
                "context_size": ctx_size,
                "avg_ttft_s": round(sum(r["ttft_s"] for r in good) / len(good), 2),
                "avg_tps": (
                    round(sum(_m) / len(_m), 2)
                    if (_m := [r["tokens_per_sec"] for r in good
                               if r.get("tokens_per_sec") is not None])
                    else None
                ),
                "tps_measurable": any(r.get("streamed") for r in good),
                "avg_total_s": round(
                    sum(r["total_s"] for r in good) / len(good), 2
                ),
                "avg_pageouts": round(
                    sum(r["pageouts_delta"] for r in good) / len(good), 1
                ),
                "failures": len(iter_results) - len(good),
                "runs": len(good),
            }
        else:
            results[ctx_label] = {
                "context_size": ctx_size,
                "avg_ttft_s": None,
                "avg_tps": None,
                "tps_measurable": False,
                "avg_total_s": None,
                "avg_pageouts": None,
                "failures": len(iter_results),
                "runs": 0,
            }
        print()

    # Save raw results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "api_url": api_url,
            "model": model_name,
            "iterations": iterations,
            "output_tokens": OUTPUT_TOKENS,
            "context_sizes": context_sizes,
        },
        "summary": results,
        "raw": all_raw,
    }
    results_file.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Raw results saved to {results_file}\n")

    # Summary table
    print(f"{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(
        f"  {'Context':>8}  {'TTFT':>8}  {'TPS':>8}  {'Total':>8}  "
        f"{'Pageouts':>9}  {'Status':>8}"
    )
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}")

    baseline_tps = None
    recommended = None
    unmeasurable = False

    for label, data in results.items():
        if data["runs"] == 0:
            print(f"  {label:>8}  {'—':>8}  {'—':>8}  {'—':>8}  {'—':>9}  {'FAIL':>8}")
            continue

        if data["avg_tps"] is None:
            # Server never streamed, so there is no decode rate to compare.
            # Fall back to TTFT and swap pressure and say so.
            unmeasurable = True
            status = "SWAP" if data["avg_pageouts"] > 100 else "no TPS"
            print(
                f"  {label:>8}  {data['avg_ttft_s']:>7.1f}s  "
                f"{'n/a':>8}  {data['avg_total_s']:>7.1f}s  "
                f"{data['avg_pageouts']:>8.0f}  {status:>8}"
            )
            continue

        if baseline_tps is None:
            baseline_tps = data["avg_tps"]

        tps_drop = (
            (1 - data["avg_tps"] / baseline_tps) * 100 if baseline_tps else 0
        )

        status = "OK"
        if data["avg_pageouts"] > 100:
            status = "SWAP"
        elif tps_drop > 15:
            status = f"-{tps_drop:.0f}%"
        else:
            recommended = label

        print(
            f"  {label:>8}  {data['avg_ttft_s']:>7.1f}s  "
            f"{data['avg_tps']:>7.1f}  {data['avg_total_s']:>7.1f}s  "
            f"{data['avg_pageouts']:>8.0f}  {status:>8}"
        )

    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}")

    # Recommendation
    print()
    if unmeasurable:
        print("  THROUGHPUT NOT MEASURED")
        print(
            "  At least one size returned the whole completion in a single frame,"
        )
        print(
            "  so there is no decode window to divide by and no tokens/sec to"
        )
        print(
            "  report. Check that the endpoint honours stream=true before citing"
        )
        print(
            "  any number from this run. TTFT and pageouts above are still valid."
        )
    elif recommended:
        print(f"  RECOMMENDATION: {recommended} context window")
        print(
            f"  Highest size with <15% throughput drop and no swap pressure."
        )
    else:
        print("  RECOMMENDATION: Stay at 4K, all larger sizes showed degradation.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local LLM context window benchmark")
    parser.add_argument(
        "--api-url", default=API_URL, help=f"API base URL (default: {API_URL})"
    )
    parser.add_argument(
        "--model", default=MODEL_NAME, help=f"Model name to send in request body (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--iterations", type=int, default=ITERATIONS, help=f"Runs per size (default: {ITERATIONS})"
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
    args = parser.parse_args()

    if args.think_kwargs:
        THINK_KWARGS = json.loads(args.think_kwargs)
    elif args.think is not None:
        _key = "thinking_mode" if args.think_style == "minimax" else "enable_thinking"
        THINK_KWARGS = {_key: args.think == "on"}
    if THINK_KWARGS:
        print(f"chat_template_kwargs: {THINK_KWARGS}")

    sizes = CONTEXT_SIZES
    if args.sizes:
        sizes = [int(s) * 1024 for s in args.sizes.split(",")]

    run_benchmark(args.api_url, args.iterations, sizes, args.model, Path(args.results_file))
