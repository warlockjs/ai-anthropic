# Changelog — @warlock.js/ai-anthropic

All notable changes to `@warlock.js/ai-anthropic` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). `@warlock.js/*` packages are released in lockstep — every package shares the same version number, so a version below may list only the changes that affected this package.

## 4.5.0 - 2026-07-01

### Fixed

- **All upstream `ClientOptions` now reach the Anthropic client.** The SDK constructor peels off the framework-only `provider` / `pricing` keys and forwards the rest (`timeout`, `maxRetries`, `defaultHeaders`, custom `fetch`, `baseURL`, …) verbatim, instead of dropping everything but `apiKey` / `baseURL`.

## 4.3.0 - 2026-06-21

### Added

- **Usage accounting** — `usage.cacheWriteTokens` is populated from Anthropic's `cache_creation_input_tokens` (alongside `cachedTokens`); `reasoningTokens` is left unset because Anthropic bills thinking inside `output_tokens`.
- **Extended thinking** — `ModelCallOptions.reasoning` maps to Anthropic's `thinking` budget (`reasoning.effort` → a tiered budget, floored at 1024); `temperature` is dropped when thinking is enabled.
- **System-prompt prompt caching** — `cacheControl.breakpoints >= 1` emits the system prompt with `cache_control: { type: "ephemeral" }`.
- **Capabilities** — `reasoning`, `promptCaching`, and `pdf` are now advertised; `audio` stays absent.

## 4.2.0

### Added

- Opt-in `promptCaching` flag on the model config — marks tool definitions with `cache_control: { type: "ephemeral" }` so multi-trip agents reuse the static tool schemas at the cache-read rate. Off by default.

## 4.1.15

- Baseline — per-package changelog tracking starts at this version.
