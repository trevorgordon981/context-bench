# context-bench

A simple benchmark for characterizing the safe operating range of a local LLM
serving endpoint (MLX, vLLM, llama.cpp, or anything speaking an OpenAI-compatible
`/v1/chat/completions` API) across context-window sizes.

## Overview

`context-bench.py` sweeps a configurable set of context sizes (default:
4k, 8k, 16k, 32k, 64k, 96k, 128k tokens), issues streaming chat-completion
requests at each size, and records:

- **Time-to-first-token (TTFT)** — prefill latency
- **Tokens/sec during decode** — steady-state throughput
- **Total wall time per request**
- **Peak memory pressure** — macOS `memory_pressure` free-percentage deltas
- **Page-out counts** — swap activity via `vm_stat`, a proxy for memory exhaustion

The tool is most useful before putting a new model deployment into production:
run it once to learn where throughput drops off a cliff and where the OS starts
swapping, then set your serving layer's `max_context` accordingly.

## Reasoning models

Servers stream chain-of-thought on a **separate delta field** — `reasoning` or
`reasoning_content` — not on `content`. A benchmark that counts only `content`
silently breaks three metrics at once on any reasoning model:

- **`output_tokens`** undercounts by however much the model thought
- **`tokens_per_sec`** is computed over that undercount, so decode throughput
  reads far lower than it actually is
- **TTFT** — the worst of the three — fires on the first *content* token, so the
  entire thinking phase gets billed as prefill latency

Measured on `deepseek-v4-flash` at 4k context: `reasoning_len 9359` vs
`content_len 537`. About 95% of generated tokens were invisible to a content-only
counter, and TTFT read ~90s instead of ~1s.

This tool therefore reports **TTFT as the first token of either kind** (true
prefill), with `ttft_content_s` alongside it for the answer-latency question, and
counts reasoning and content separately (`reasoning_chars`, `content_chars`,
`reasoning_share`).

### Pinning the thinking mode

There is **no portable switch** for reasoning. The key *and its default* are
vendor-specific:

| family          | `chat_template_kwargs` key | default |
|-----------------|----------------------------|---------|
| DeepSeek, Qwen  | `enable_thinking`          | off     |
| MiniMax         | `thinking_mode`            | on      |

Benchmarking without pinning it compares a thinking run against a non-thinking one
and calls the difference throughput. Use:

```bash
# force reasoning ON for a DeepSeek/Qwen endpoint
./context-bench.py --think on --think-style deepseek

# force it OFF for MiniMax
./context-bench.py --think off --think-style minimax

# or pass the kwargs verbatim
./context-bench.py --think-kwargs '{"enable_thinking": true}'
```

The kwargs actually sent are recorded on every result row as `think_kwargs`, so a
results file always says which mode produced it.

### Token counts and truncation

Counts come from the server's own `usage` block where available (requested via
`stream_options.include_usage`); `token_source` reports `usage` or `sse_chunks` so
you know whether a number is exact or estimated. `finish_reason` is recorded, and
`truncated` flags any run cut off at `max_tokens` — a truncated run is not
comparable to one that stopped on its own, and at large context sizes truncation
becomes common enough to distort an average silently.

## Design

The benchmark constructs prompts by repeating a fixed **filler block** of generic
transformer-architecture prose until the target token count is reached. This
approach was chosen deliberately over random tokens or real documents because:

1. **Consistent tokenization** — the same text across runs means the same token
   count across runs, eliminating tokenizer variance between model families.
2. **Bounded memory growth** — filler content is semantically coherent English,
   which avoids pathological tokenizer behavior (e.g. byte-fallback explosions
   on random bytes) that would skew memory-pressure measurements.
3. **Reproducibility** — no external corpus required; the benchmark is
   self-contained and deterministic across machines.

The prompt uses a rough `3.5 chars/token` heuristic to size the filler; actual
token counts will vary by tokenizer but are consistent within a single run.

Memory metrics rely on macOS-specific commands (`memory_pressure`, `vm_stat`).
On Linux the benchmark still runs — throughput numbers are accurate — but those
fields will be `None` / `0`.

## Usage

```bash
pip install -r requirements.txt

# Default: hits http://127.0.0.1:8090/v1, model name "default", 3 iterations
python context-bench.py

# Point at your endpoint and model
python context-bench.py \
    --api-url http://127.0.0.1:8080/v1 \
    --model my-model-name \
    --iterations 5

# Custom context sizes (in K tokens)
python context-bench.py --sizes 4,8,16,32,64

# Write results somewhere specific
python context-bench.py --results-file /tmp/bench.json
```

The default API URL assumes your serving endpoint exposes an OpenAI-compatible
chat-completions API on port `8090`. Override via `--api-url` or the `API_URL`
environment variable. Override the model name via `--model` or the `MODEL_NAME`
environment variable — this is the string sent as the `model` field in the
request body (many local servers accept anything, but some require an exact
match).

## Expected Output

A console summary table plus a JSON report with the fields below.

**The numbers in this block are illustrative placeholders, not measurements.**
They exist to show the shape of the report. Do not cite them as results for any
model or machine. Real figures depend entirely on the model, the quantisation,
the serving engine, and the hardware.

```json
{
  "timestamp": "2026-04-05T12:34:56",
  "config": {
    "api_url": "...",
    "model": "...",
    "iterations": 3,
    "output_tokens": 512,
    "context_sizes": [4096, 8192, ...]
  },
  "summary": {
    "4K": {
      "context_size": 4096,
      "avg_ttft_s": 0.42,          // placeholder value
      "avg_tps": 58.3,            // placeholder value
      "avg_total_s": 9.1,         // placeholder value
      "avg_pageouts": 0,
      "failures": 0,
      "runs": 3
    },
    ...
  },
  "raw": [ /* per-iteration records */ ]
}
```

<!-- TODO(trevor): no real measured sweep is published here yet.
     Probed 2026-10-08: 127.0.0.1:8090 was not listening on the laptop or on
     bat-studio (8082/8090 both refused; 8091/8095 answered 401, those are the
     gateway). Once the MLX endpoint is back up, run:
         python context-bench.py --sizes 4,16,32,64,120 --iterations 3 \
             --model <served-model-id> --results-file results-macstudio.json
     then add a "Measured results" section naming the hardware (Mac Studio
     M3 Ultra 512GB) and the exact model and quantisation. Note the caveat
     below about avg_tps before publishing any number from it. -->

<!-- CAVEAT(trevor): avg_tps is computed as output_tokens / (total_s - ttft_s)
     at context-bench.py:226. If the server buffers the whole completion into
     one chunk instead of streaming, ttft_s collapses onto total_s, the
     denominator goes to ~0, and avg_tps explodes. That is exactly what
     alfred-infra/context-bench-results.json shows: avg_total_s minus
     avg_ttft_s is 0.02s on every row, and avg_tps reads ~11,000 on a Mac
     Studio, which is not a real decode rate. Confirm the run actually
     streamed before trusting avg_tps. -->

## Interpretation

What you're looking for in the summary table:

- **Monotonic throughput decay** — TPS gently trending down as context grows is
  normal and expected; KV-cache attention is O(N) per token at decode time.
- **Sudden TPS cliff** — a step-change drop (e.g. 60 TPS → 8 TPS) between two
  adjacent sizes usually means one of:
  - **OOM / swap thrashing** — check `avg_pageouts`; non-zero values indicate
    the working set no longer fits in RAM and the OS is paging.
  - **Prefill saturation** — the model's attention kernels are falling off a
    hardware-friendly tile size, or the KV cache is crossing a buffer boundary.
- **Failures at large sizes** — the serving layer is rejecting prompts above
  some configured `max_context`. Check your server logs.
- **Recommendation line** — the tool prints the largest tested size with
  <15% throughput drop from baseline and no swap pressure, as a starting point
  for production configuration.

## Related projects

Part of a self-hosted LLM operations toolkit:

- [blockops-proxy](https://github.com/trevorgordon981/blockops-proxy) — the proxy layer whose MAX_CONCURRENT and context thresholds this benchmark helps tune
- [llm-otel-proxy](https://github.com/trevorgordon981/llm-otel-proxy) — metrics proxy that tracks the exact same tokens/latency dimensions in production
- [alfred-infra](https://github.com/trevorgordon981/alfred-infra) — dashboards/monitoring for the infrastructure this benchmark characterizes
- [alfred-rag](https://github.com/trevorgordon981/alfred-rag) — RAG stack that benefits from knowing the safe operating context range

## License

MIT. See [LICENSE](LICENSE).
