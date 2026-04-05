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

A console summary table plus a JSON report with these fields:

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
      "avg_ttft_s": 0.42,
      "avg_tps": 58.3,
      "avg_total_s": 9.1,
      "avg_pageouts": 0,
      "failures": 0,
      "runs": 3
    },
    ...
  },
  "raw": [ /* per-iteration records */ ]
}
```

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
