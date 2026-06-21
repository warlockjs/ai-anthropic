# Changelog — @warlock.js/ai-anthropic

All notable changes to `@warlock.js/ai-anthropic` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). `@warlock.js/*` packages are released in lockstep — every package shares the same version number, so a version below may list only the changes that affected this package.

## 4.3.0 - 2026-06-21

### Added

- Cost-truth wiring against the core `@warlock.js/ai` contract:
  - **Usage accounting** — `usage.cacheWriteTokens` is now populated from Anthropic's `cache_creation_input_tokens` (the cache-write surcharge), alongside the existing `cachedTokens` (from `cache_read_input_tokens`). In streaming, both are seeded from `message_start` and then refreshed from the cumulative `message_delta` counts. `reasoningTokens` is intentionally left unset because Anthropic bills extended-thinking tokens inside `output_tokens`.
  - **Extended thinking** — `ModelCallOptions.reasoning` now maps to Anthropic's `thinking: { type: "enabled", budget_tokens }`. `reasoning.maxTokens` sets the budget directly; `reasoning.effort` (`low`/`medium`/`high`) maps to a tiered budget (`1024`/`4096`/`12000`); any budget is floored at the `1024` minimum. `temperature` is dropped when thinking is enabled (Anthropic rejects the combination). New `reasoning?: boolean` model-config override.
  - **System-prompt prompt caching** — a per-call `ModelCallOptions.cacheControl.breakpoints >= 1` now emits the hoisted system prompt as a `TextBlockParam` carrying `cache_control: { type: "ephemeral" }`, independent of the existing tools-caching `promptCaching` flag.
  - **Capabilities** — `reasoning` (default `true`, overridable), `promptCaching` (`true`), and `pdf` (`true`) are now advertised on `ModelCapabilities`; `audio` stays absent (Anthropic has no audio-input block).

## 4.2.0

### Added

- Opt-in `promptCaching` flag on the model config. When enabled, the tool definitions are marked with `cache_control: { type: "ephemeral" }`, so multi-trip agents reuse the static tool schemas at Anthropic's cache-read rate instead of re-sending them at full price every trip. Off by default — a cache write costs more, so it only pays off across repeated trips; the system prompt is left uncached because it carries per-turn content.

## 4.1.15

- Baseline — per-package changelog tracking starts at this version.
