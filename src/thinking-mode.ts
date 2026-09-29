/**
 * How a Claude model accepts extended thinking on the Messages API.
 *
 * - `"adaptive"` — `thinking: { type: "adaptive" }`, with depth steered by
 *   `output_config.effort`. Sending `budget_tokens` to these models is a
 *   400. Installed SDK: `ThinkingConfigAdaptive`
 *   (`@anthropic-ai/sdk/resources/messages/messages.d.ts:1065`) and
 *   `OutputConfig.effort` (`messages.d.ts:796-800`).
 * - `"budget"` — the legacy `thinking: { type: "enabled", budget_tokens }`
 *   (`ThinkingConfigEnabled`, `messages.d.ts:1078-1090`). Required on
 *   Haiku 4.5 and older models.
 */
export type AnthropicThinkingMode = "adaptive" | "budget";

/*
 * Model-family table (source: claude-api ref, "Thinking & Effort" per-model
 * table, cached 2026-09-25; summarised in `releases/5.26-ai-sdk-review.md` §A):
 *
 * | Model ids                                    | Mode       | `budget_tokens`         | `temperature`      |
 * |----------------------------------------------|------------|-------------------------|--------------------|
 * | claude-fable-5, claude-fable-5-1 (+ mythos)  | adaptive   | removed (400)           | removed (400)      |
 * | claude-opus-5-5, claude-opus-5               | adaptive   | removed (400)           | removed (400)      |
 * | claude-opus-4-8, claude-opus-4-7             | adaptive   | removed (400)           | removed (400)      |
 * | claude-sonnet-5-5, claude-sonnet-5           | adaptive   | removed (400)           | 400 (5.5: non-1.0) |
 * | claude-opus-4-6, claude-sonnet-4-6           | budget (*) | deprecated, still works | allowed            |
 * | claude-haiku-4-5 and every older model       | budget     | required for thinking   | allowed            |
 * | anything not listed (future / unknown ids)   | adaptive   | —                       | —                  |
 *
 * (*) The 4.6 pair accepts both shapes; the claude-api ref recommends
 * adaptive there but keeps `budget_tokens` functional as a transitional
 * path. It stays on `budget` so 4.6 callers see no behaviour change (their
 * `reasoning.maxTokens` is still honoured as a hard budget) and so that
 * the `budget` set is exactly the set that still accepts `temperature`
 * (see `acceptsTemperature`). Opt 4.6 into adaptive with
 * `anthropic.model({ name, thinkingMode: "adaptive" })`.
 *
 * Unknown ids default to adaptive because that is the current API
 * direction: every model released after Opus 4.6 rejects `budget_tokens`.
 */

/**
 * Family prefixes that are budget-mode in their entirety — plain
 * `startsWith` match (dated and `-latest` variants included).
 */
const BUDGET_FAMILY_PREFIXES = [
  // Claude 3 / 3.5 / 3.7 legacy naming (claude-3-haiku-…, claude-3-5-sonnet-…).
  "claude-3",
  // Pre-3 text models: no thinking support at all, never adaptive.
  "claude-2",
  "claude-instant",
  // Haiku 4.x: Haiku 4.5 still requires budget_tokens (claude-api ref).
  "claude-haiku-4",
];

/**
 * Individual pre-4.7 releases in the `claude-<tier>-4-*` naming. Matched
 * as a release root: the id must equal the root or continue with a
 * separator that is not a new minor version number, so `claude-opus-4`
 * matches `claude-opus-4-20250514` but NOT `claude-opus-4-7`. Ids from the
 * installed SDK `Model` union (`messages.d.ts:795`) plus the 4.6 pair.
 */
const BUDGET_RELEASE_ROOTS = [
  "claude-opus-4",
  "claude-opus-4-0",
  "claude-opus-4-1",
  "claude-opus-4-5",
  "claude-opus-4-6",
  "claude-sonnet-4",
  "claude-sonnet-4-0",
  "claude-sonnet-4-5",
  "claude-sonnet-4-6",
];

/**
 * A remainder that starts a new minor version (`-7`, `-10`) — used to
 * stop a release root from swallowing a later release of the same line.
 * Dates (`-20250514`) and tags (`-latest`, `-v1`) are not minor versions.
 */
const NEXT_MINOR_VERSION = /^-\d{1,2}(?!\d)/;

function matchesReleaseRoot(name: string, root: string): boolean {
  if (!name.startsWith(root)) {
    return false;
  }

  const rest = name.slice(root.length);

  if (rest === "") {
    return true;
  }

  return (rest.startsWith("-") || rest.startsWith("@")) && !NEXT_MINOR_VERSION.test(rest);
}

/**
 * Infer the thinking mode for a Claude model id. Budget for Haiku 4.5, the
 * 4.6 pair, and every older family; adaptive for everything else,
 * including unknown and future ids. Explicit per-model config
 * (`AnthropicModelConfig.thinkingMode`) always wins over this inference.
 *
 * @example
 * inferThinkingMode("claude-opus-5-5");  // → "adaptive"
 * inferThinkingMode("claude-haiku-4-5"); // → "budget"
 * inferThinkingMode("claude-opus-9");    // → "adaptive" (unknown → current API direction)
 */
export function inferThinkingMode(modelName: string): AnthropicThinkingMode {
  const normalized = modelName.toLowerCase();

  if (BUDGET_FAMILY_PREFIXES.some((prefix) => normalized.startsWith(prefix))) {
    return "budget";
  }

  if (BUDGET_RELEASE_ROOTS.some((root) => matchesReleaseRoot(normalized, root))) {
    return "budget";
  }

  return "adaptive";
}

/**
 * Whether a model of this thinking mode accepts an explicit `temperature`.
 *
 * Installed SDK, `MessageCreateParamsBase.temperature`
 * (`messages.d.ts:2036-2040`): "Models released after Claude Opus 4.6 do
 * not support setting temperature. A value of 1.0 will be accepted for
 * backwards compatibility, all other values will be rejected with a 400
 * error." The claude-api ref's per-model table agrees: sampling is
 * "Removed - 400" on Fable 5/5.1, Opus 5.5/5/4.8/4.7 and Sonnet 5, and
 * "Non-default values - 400" on Sonnet 5.5; "Allowed" on Opus 4.6,
 * Sonnet 4.6, Haiku 4.5 and older.
 *
 * That split is exactly the adaptive/budget split in the table above
 * (adaptive = released after Opus 4.6; budget = Opus 4.6 and earlier), so
 * the thinking mode doubles as the sampling-parameter classification.
 */
export function acceptsTemperature(mode: AnthropicThinkingMode): boolean {
  return mode === "budget";
}
