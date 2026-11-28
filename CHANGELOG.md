# Changelog

All notable changes to `context-bench` are documented in this file.

## [0.1.0] - 2026-11-27

### Added
- Streaming context-window benchmark (`context-bench.py`) for OpenAI-compatible `/v1/chat/completions` endpoints across configurable context tiers (4K–128K default).
- Reasoning-aware SSE parsing counting both `reasoning`/`reasoning_content` and `content` tokens, separating client-observed TTFT (`client_ttft_s`) from first-content TTFT (`client_ttft_content_s`), plus thinking-mode controls (`--think`, `--think-style`, `--think-kwargs`).
- Strict streaming decode TPS measurement (`first_gen_time` to `last_gen_time` with `--min-decode-interval`), excluding buffered single-frame completions and sub-threshold coalesced bursts while still reporting `end_to_end_tokens_per_sec`.
- Explicit stream outcome classification (`completed`, `length_limited`, `no_output`, `incomplete`, `other_finish`, `error`) failing incomplete SSE streams lacking `[DONE]` or `finish_reason` and preventing zero-output rows from skewing baselines.
- Fail-closed server `usage.prompt_tokens` verification (with opt-in `--allow-unverified-usage` tagging), combined prompt+completion total-token accounting (`observed_total_tokens`, `effective_total_within_tier`), `--strict-total-context` exact-total gating, and rejection of contradictory `completion_tokens <= 0` usage alongside generated frames.
- Cold vs. warm cache-mode sweeps (`--cache-mode cold|warm`) with per-request `X-Request-Id` and prompt SHA-256 logging, plus loopback-only macOS `memory_pressure`/`vm_stat` swap telemetry.
- Built-in offline synthetic mock-SSE demo (`--demo`), 48-test unit suite, and GitHub Actions CI.
