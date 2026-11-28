# context-bench

A lightweight benchmark for characterizing the safe operating range of an
OpenAI-compatible LLM serving endpoint (MLX, vLLM, llama.cpp, or any server
exposing `/v1/chat/completions`) across context-window sizes.

## Overview

`context-bench.py` sweeps a configurable set of context sizes (default:
4K, 8K, 16K, 32K, 64K, 96K, 128K tokens), issues streaming chat-completion
requests at each size, and separates valid streaming measurements from buffered,
empty, truncated, token-mismatched, or failed responses.

## Metric Definitions

Every iteration record in `raw` and aggregated tier in `summary` uses explicit
client-side definitions:

- **Client-observed TTFT (`client_ttft_s` / `ttft_s`)** — wall time from HTTP
  request dispatch (after `httpx.Client` creation) to the first nonempty
  generation frame (`reasoning`, `reasoning_content`, or `content`). Because
  this is measured at the client, it includes TCP/TLS/HTTP overhead, server
  queueing, and prompt prefill. Returns `null` if zero output tokens arrive.
- **Content TTFT (`client_ttft_content_s` / `ttft_content_s`)** — wall time from
  dispatch to the first `content` frame, reported alongside `client_ttft_s` so
  reasoning-model think time is not confused with prefill latency.
- **Streaming decode TPS (`tokens_per_sec` / `avg_tps`)** — computed strictly as
  `output_tokens / (last_gen_time - first_gen_time)` across distinct nonempty
  generation frames, **only when** `generation_frames >= 2` and the observed
  inter-frame span is at least `--min-decode-interval` (default `0.01` s / 10 ms).
  Single-frame buffered completions, zero-output completions, and sub-threshold
  coalesced bursts report `null` (`N/A` in console output) with an explicit
  `decode_tps_exclusion_reason` instead of dividing by stream teardown time.
- **End-to-end throughput (`end_to_end_tokens_per_sec` / `avg_e2e_tps`)** —
  `output_tokens / total_s`, reported separately from streaming decode TPS so
  buffered responses still retain honest overall throughput without polluting
  decode-rate curves.
- **Total request wall time (`total_s`)** — elapsed time from request dispatch
  to stream close.
- **Local-host memory snapshots (`host_mem_free_pct_before`, `host_mem_free_pct_after`)** —
  macOS `memory_pressure -Q` free-memory percentages sampled before and after
  each request **only** when targeting a local loopback endpoint (`127.0.0.1`,
  `localhost`, `::1`, `0.0.0.0`). On Linux or remote endpoints, these fields are
  `null` (`N/A`). They do **not** measure peak intra-request memory pressure or
  remote GPU VRAM.
- **Local-host page-out delta (`host_pageouts_delta` / `avg_host_pageouts`)** —
  macOS `vm_stat` page-out delta sampled before and after each request on local
  loopback endpoints only. Returns `null` (`N/A`) when unavailable or targeting
  a remote endpoint — never `0` as a fallback.

## Stream Outcomes, Cohorts, and Safe-Context Gating

Every run is classified into an explicit `run_status` before any aggregation:

- `completed`: terminated cleanly (`[DONE]` or normal `finish_reason` in
  `stop`/`eos`/`end_turn`) with nonempty generated output.
- `length_limited`: terminated with `finish_reason="length"` (`truncated=true`).
- `no_output`: terminated (`[DONE]` or `finish_reason`) with zero generated
  output (treated as a failure, never a valid 0-TPS sample).
- `incomplete`: HTTP 200 stream closed before `[DONE]` or `finish_reason`.
- `other_finish`: terminated with a non-standard `finish_reason` (e.g.
  `content_filter`).
- `error`: HTTP non-200 status, timeout, or transport failure.

### Prompt Token Tolerance, Combined Total Tokens & Cache Modes

- **Nominal tier vs. observed prompt and combined total tokens**: each tier is
  configured with a nominal context size (`requested_context_tokens`, e.g.
  `4096` for `4K`), a target input prompt size (`requested_prompt_tokens =
  requested_context_tokens - requested_output_tokens`, e.g. `3584`), and a
  reserved output budget (`requested_output_tokens`, default `512`). Every raw
  row records `observed_prompt_tokens`, `observed_output_tokens`,
  `observed_total_tokens` (`observed_prompt_tokens + observed_output_tokens`),
  `effective_total_within_tier` (`observed_total_tokens <=
  requested_context_tokens`), `prompt_tokens_verified`, and `prompt_size_basis`.
  Every tier summary reports `observed_prompt_tokens_avg`,
  `observed_output_tokens_avg`, `avg_observed_total_tokens`, and
  `effective_total_within_tier`, and the console summary prints `ObsIn` and
  `ObsTot` right beside the nominal `Context` tier.
- **Prompt-tier tolerance vs. exact combined-total gating (`--strict-total-context`)**:
  - When the server reports `usage.prompt_tokens`, the benchmark verifies both
    relative tolerance (`--prompt-token-tolerance`, default `±25%`) against
    `requested_prompt_tokens` and that `observed_prompt_tokens <=
    requested_context_tokens`. Out-of-tolerance or over-budget prompts set
    `prompt_token_mismatch=true` and `prompt_tokens_verified=false`.
  - Because prompt verification allows `±25%` tokenizer variance, a prompt can
    pass prompt-tier tolerance while `observed_prompt_tokens +
    observed_output_tokens` exceeds the nominal tier (for example, `4000` prompt
    + `512` completion = `4512` total under nominal `4K` / `4096`). In default
    mode, such a run is **not** falsely called an exact `<=4096` combined-token
    measurement: the row records `effective_total_within_tier=false` and a
    `total_tokens_exceeded_nominal_tier` warning, the summary table flags
    `OK (tot>tier)` alongside `ObsTot=4512`, and `recommendation` reports
    `recommended_nominal_tier`, `recommended_observed_prompt_tokens`,
    `recommended_observed_total_tokens`, `recommended_effective_total_within_tier`,
    and `verification_scope="verified_prompt_tier_only"` (distinct from
    `"verified_prompt_and_total_within_tier"`).
  - Pass `--strict-total-context` for exact-total tier comparisons: any tier
    whose combined `observed_total_tokens` exceeds `requested_context_tokens`,
    omits total usage, or reports contradictory `completion_tokens <= 0` despite
    non-empty generated frames is disqualified as `TOTAL_OVER` (or `UNVERIFIED`)
    rather than qualifying for `recommended_context`.
- **Missing server token usage (fail-closed by default)**: if an endpoint omits
  `usage.prompt_tokens` (so `observed_prompt_tokens` is `null`), streaming
  latency and throughput are still measured and displayed in the report/census,
  but the run is marked `prompt_tokens_verified=false` with
  `prompt_size_basis="heuristic_estimate"`, disqualified from verified safe
  context recommendations (`missing_prompt_token_usage`), and labeled with a
  `~` approximate size marker and `UNVERIFIED` status. Pass
  `--allow-unverified-usage` only if you explicitly want an opt-in recommendation
  based on heuristic prompt sizes — such recommendations are still explicitly
  tagged `unverified` (`OK (unverified)` / `recommendation_status="unverified"`).
- **Cold vs. warm sweeps (`--cache-mode`)**:
  - `cold` (default) prepends a unique `[Benchmark Request ID: ...]` nonce to
    each iteration's prompt so prefix KV caches cannot mask true prefill latency.
  - `warm` sends identical prompt text across iterations at a given size so you
    can intentionally measure prefix-cache-warmed sweeps.
  - Every run records `request_id`, `cache_mode`, `prompt_sha256`, and (when
    reported by the server) `cached_prompt_tokens`, and sends `X-Request-Id`.

### Safe-Context Recommendation Rules

All rows remain visible in `raw`, `summary[label].cohorts`, and the console
table. However, a context tier can only qualify for `recommended_context` when:

1. It has at least one qualifying `completed` run with measurable positive
   streaming decode TPS (`avg_tps > 0`).
2. It has verified server `usage.prompt_tokens` within tolerance (`prompt_tokens_verified=true`,
   zero `missing_prompt_token_usage` runs) — unless `--allow-unverified-usage` is
   explicitly passed, in which case the tier and recommendation are tagged `unverified`.
3. It has zero `length_limited` (truncated) and zero `prompt_token_mismatch`
   runs.
4. When `--strict-total-context` is enabled, it has server-reported combined
   `observed_total_tokens <= requested_context_tokens`
   (`effective_total_within_tier=true`, zero `total_context_overrun` runs).
5. Its `avg_tps` drops by no more than `15%` from the first qualifying baseline
   tier (unmeasurable, unverified, strict-overrun, or zero-output rows never
   establish or disable the verified baseline).
6. Measured local host page-outs do not exceed `100` (when host swap telemetry
   is unavailable, the recommendation explicitly notes that swap pressure was
   not verified).

The `recommendation` object always separates the nominal tier
(`recommended_nominal_tier`, `recommended_nominal_context_tokens`) from the
server-observed token counts (`recommended_observed_prompt_tokens`,
`recommended_observed_total_tokens`), `recommended_effective_total_within_tier`,
and `verification_scope`. If no tested tier meets all criteria, the benchmark
reports `RECOMMENDATION: None` rather than defaulting to `4K`.

## Reasoning Models

Servers stream chain-of-thought on a separate delta field (`reasoning` or
`reasoning_content`) rather than `content`. Counting only `content` undercounts
output tokens, underreports decode TPS, and bills the entire thinking phase to
TTFT. `context-bench.py` counts both fields, reports `client_ttft_s` at the
first token of either kind, reports `client_ttft_content_s` separately, and
records `reasoning_chars`, `content_chars`, and `reasoning_share`.

### Pinning the Thinking Mode

There is no portable switch for reasoning across model families:

| Family         | `chat_template_kwargs` key | Default |
|----------------|----------------------------|---------|
| DeepSeek, Qwen | `enable_thinking`          | off     |
| MiniMax        | `thinking_mode`            | on      |

```bash
# Force reasoning ON for a DeepSeek/Qwen endpoint
python3 context-bench.py --think on --think-style deepseek

# Force reasoning OFF for MiniMax
python3 context-bench.py --think off --think-style minimax

# Pass raw JSON kwargs verbatim
python3 context-bench.py --think-kwargs '{"enable_thinking": true}'
```

The kwargs sent are recorded on every result row (`think_kwargs`) and in
`config.think_kwargs`.

## Usage

**Prerequisite:** **Python >= 3.11** (use `python3.11` explicitly on macOS so stock system Python 3.9 is not selected).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Offline synthetic mock-SSE demo (no LLM server required)
python context-bench.py --demo

# Default: hits http://127.0.0.1:8090/v1, model "default", 3 cold iterations
python context-bench.py

# Point at your endpoint and model with custom sizes and cache mode
python context-bench.py \
    --api-url http://127.0.0.1:8080/v1 \
    --model my-model-name \
    --sizes 4,8,16,32,64 \
    --iterations 5 \
    --cache-mode cold \
    --results-file /tmp/bench.json
```

### Synthetic Mock-SSE Demo

Run the built-in offline demo to inspect the full report format and gating
behavior without starting a live inference server:

```bash
python context-bench.py --demo --results-file /tmp/context-bench-demo.json
```

The demo starts a local ephemeral mock OpenAI SSE server on `127.0.0.1` and
sweeps five synthetic scenarios (`4K`, `8K`, `16K`, `32K`, `64K`):

- `4K` / `8K`: normal multi-frame streaming (`OK`, recommends `8K`)
- `16K`: buffered single-frame completion (`DecTPS=N/A`, `E2ETPS` still shown, `NO_DECODE`)
- `32K`: `finish_reason=length` + 2x prompt-token overrun (`TRUNC+SIZE`, excluded from recommendation, preserved in census)
- `64K`: >15% streaming decode throughput drop (not recommended)

## Report Schema

The JSON report includes full run configuration (`benchmark_host`,
`endpoint_host`, `endpoint_is_local`, `cache_mode`, `prompt_token_tolerance`,
`min_decode_interval_s`, `strict_total_context`, `memory_telemetry`), a
top-level `recommendation` object (`recommended_context`,
`recommended_nominal_tier`, `recommended_nominal_context_tokens`,
`recommended_observed_prompt_tokens`, `recommended_observed_total_tokens`,
`recommended_effective_total_within_tier`, `verification_scope`), per-size
`summary` cohorts and exclusions (including `avg_observed_total_tokens` and
`effective_total_within_tier`), and all per-iteration `raw` records.

## Methodology Limits

- **Heuristic prompt sizing and combined-total variance**: filler prompts use a
  `3.5 chars/token` heuristic for repeated English technical prose. Actual token
  counts vary by tokenizer; by default, a tier cannot qualify as a verified safe
  context recommendation unless the server returns `usage.prompt_tokens` within
  tolerance (`±25%` of `requested_prompt_tokens` and `<= requested_context_tokens`).
  Even when prompt verification succeeds, `observed_prompt_tokens +
  observed_output_tokens` can exceed the nominal tier size; the benchmark tracks
  `observed_total_tokens` (`ObsTot`) and `effective_total_within_tier` separately
  from prompt verification, labels prompt-only matches distinctly from exact
  within-tier totals, and provides `--strict-total-context` for exact-total
  comparisons. When a server omits `usage.prompt_tokens`, streaming performance
  is still reported under an approximate (`~`) size label with `UNVERIFIED`
  status unless `--allow-unverified-usage` is explicitly passed (in which case
  any resulting recommendation is still tagged `unverified`).
- **Client-side timing**: `client_ttft_s` includes network and queueing latency
  in addition to server prefill. For pure server-side prefill timing, pair with
  engine telemetry.
- **Host-local memory snapshots only**: macOS `memory_pressure` and `vm_stat`
  capture pre/post snapshots of the benchmark runner host only when the endpoint
  is local loopback. They cannot observe peak intra-request allocations or
  remote GPU/host memory.
- **Repeated-prose compressibility**: synthetic filler prose is deterministic
  and avoids byte-fallback tokenizer explosions, but its KV/attention behavior
  may differ from high-entropy code or multilingual workloads.

## Running Tests

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_reasoning_parsing.py
python -m unittest discover -s . -p "test_*.py" -v
```

## Measured Results

Verified offline in an isolated Python 3.11.14 virtual environment (Apple Silicon macOS / `Darwin arm64`, 2026-11-27):

- **Unit test suite** (`python -m unittest discover -s . -p "test_*.py" -v`): **48/48 passed** in **1.82s** (1.93s wall time), including `test_reasoning_parsing.py` (**5/5 passed**, 0.08s wall time standalone).
- **Offline synthetic mock-SSE demo** (`python context-bench.py --demo`, 1 cold iteration across 5 synthetic tiers, **1.93s** wall time):
  - `4K`: ObsIn `3584`, ObsTot `3600`, TTFT `0.04s`, DecTPS `136.5`, E2ETPS `75.4`, Total `0.21s`, Qual `1/1`, Status `OK`
  - `8K`: ObsIn `7680`, ObsTot `7696`, TTFT `0.06s`, DecTPS `133.6`, E2ETPS `68.7`, Total `0.23s`, Qual `1/1`, Status `OK`
  - `16K`: ObsIn `15872`, ObsTot `16384`, TTFT `0.09s`, DecTPS `N/A` (`buffered_single_frame`), E2ETPS `5878.0`, Total `0.09s`, Qual `0/1`, Status `NO_DECODE`
  - `32K`: ObsIn `64512`, ObsTot `65024`, Qual `0/1`, Status `TRUNC+SIZE` (`length_limited`, `token_mismatch`, `TOTAL_OVER`)
  - `64K`: ObsIn `65024`, ObsTot `65040`, TTFT `0.13s`, DecTPS `62.1`, E2ETPS `31.4`, Total `0.51s`, Qual `1/1`, Status `-54%`
  - **Recommendation:** `8K` (`verification_scope=verified_prompt_and_total_within_tier`, observed prompt `7680`, observed total `7696 <= 8192`, baseline decode TPS `136.48`, host pageouts `0`).

## Related Projects

Part of a self-hosted LLM operations toolkit:

- [blockops-proxy](https://github.com/trevorgordon981/blockops-proxy) — proxy layer whose concurrency and context thresholds this benchmark helps tune
- [llm-otel-proxy](https://github.com/trevorgordon981/llm-otel-proxy) — metrics proxy tracking tokens and latency dimensions in production
- [alfred-infra](https://github.com/trevorgordon981/alfred-infra) — dashboards and monitoring for local LLM infrastructure
- [alfred-rag](https://github.com/trevorgordon981/alfred-rag) — RAG stack that benefits from knowing the safe operating context range

## License

MIT. See [LICENSE](LICENSE).
