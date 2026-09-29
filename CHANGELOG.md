# Changelog — @warlock.js/ai-anthropic

All notable changes to `@warlock.js/ai-anthropic` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). `@warlock.js/*` packages are released in lockstep — every package shares the same version number, so a version below may list only the changes that affected this package.

## Unreleased

### Fixed

- **Behaviour change:** Reasoning now works on current Claude models. Claude Opus 4.7, 4.8, 5 and 5.5, Sonnet 5 and 5.5, Fable 5 and 5.1, and any model id the adapter does not recognise now get `thinking: { type: "adaptive" }` with `output_config.effort` (`low`, `medium` or `high`, taken from `reasoning.effort`; `medium` when only `reasoning.maxTokens` is given). Before, every model got `budget_tokens`, which these models reject with a 400. Claude Haiku 4.5, Opus 4.6, Sonnet 4.6 and older models keep the `budget_tokens` shape. The default `max_tokens` is unchanged in both modes.
- **Behaviour change:** `temperature` is no longer sent to models released after Claude Opus 4.6 (the same adaptive-thinking models listed above), because they reject any value other than the default with a 400. Haiku 4.5, the 4.6 models and older models still receive it.
- **Behaviour change:** Claude 5-series models (`claude-opus-5*`, `claude-sonnet-5*`, `claude-fable-5*`, `claude-mythos-5*`) are now recognised as vision-capable, so image attachments are no longer rejected for them.

### Added

- `thinkingMode: "adaptive" | "budget"` on `anthropic.model({ ... })` overrides the thinking shape inferred from the model name. Use it for gateway model ids that don't follow Anthropic's naming, or to move Opus/Sonnet 4.6 onto adaptive thinking. It also decides whether `temperature` is sent. The `AnthropicThinkingMode` type is exported.

## 5.25.0 - 2026-09-28

### Changed

- Lockstep release; package APIs are unchanged.

## 5.24.0 - 2026-09-27

### Fixed

- Preserve signed and redacted extended-thinking blocks across Anthropic tool-use follow-ups, and ensure `max_tokens` exceeds the thinking budget.

### Changed

- Lockstep release; package APIs are unchanged.

## 5.23.2 - 2026-09-26

### Changed

- Lockstep patch release; package APIs are unchanged.

## 5.23.1 - 2026-09-26

### Changed

- Lockstep patch release; package APIs are unchanged.

## 5.23.0 - 2026-09-25

### Changed

- Lockstep patch release; package APIs are unchanged.

## 5.22.1 - 2026-09-25

### Changed

- Lockstep patch release; package APIs are unchanged.

## 5.22.0 - 2026-09-25

### Changed

- Lockstep release maintenance and dependency refresh.

## 5.21.0 - 2026-09-25

### Changed

- Lockstep release maintenance and dependency refresh.
## 5.20.0 - 2026-09-24

### Changed

- Added a package-level skill index and clearer discovery descriptions for grouped agent guidance.

## 5.19.0 - 2026-09-23

### Changed

- Refined package skill-discovery descriptions and regenerated the llms projections.

## 5.11.0 - 2026-09-14

_Released in lockstep with the `@warlock.js/*` family; no package-specific changes in 5.11.0._

## 5.10.0 - 2026-09-14

_Released in lockstep with the `@warlock.js/*` family; no package-specific changes in 5.10.0._

## 5.9.0 - 2026-09-13

_Released in lockstep with the `@warlock.js/*` family; no package-specific changes in 5.9.0._

## 5.2.3 - 2026-09-02

### Fixed

- Released in exact lockstep with Core's Web generator repairs so every family dependency remains installable at 5.2.3.

## 5.2.2

### Maintenance

- Restored the Warlock family to one exact, installable lockstep version.

## 5.1.0

No changes to `@warlock.js/ai-anthropic`. Released in lockstep with the `@warlock.js/web`
React-execution fix and the `@warlock.js/core` CLI additions — see those packages'
changelogs.

## 5.0.2 - 2026-08-25

No changes to `@warlock.js/ai-anthropic`. Released in lockstep with the `@warlock.js/web` SSR
fix (`ssr.noExternal`) — see that package's changelog.

## 5.0.1 - 2026-08-25

No changes to `@warlock.js/ai-anthropic`. Released in lockstep with the `create-warlock` vite
resolution pin and the `@warlock.js/web` peer narrowing — see those packages'
changelogs.

## 5.0.0 - 2026-08-25

### Changed

- This package is unchanged in 5.0.0; its version moved only because the Warlock family releases in lockstep.

## 4.12.0

### Changed

- Declares its own test runner and pins it to an exact version (`vitest@4.1.10`). The package is its own repository, so a runner resolved from a workspace root it may not be cloned with is a runner it cannot rely on. The pin is exact rather than a range because the version moved underneath the suite mid-development on an unrelated install — a suite whose runner can change without anyone choosing it proves less than it appears to

## 4.8.0 - 2026-07-19

### Changed

- **`reasoning: { effort: "none" }`** disables extended thinking (emits no `thinking` block) — the neutral "run without reasoning" level, consistent across adapters.

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
