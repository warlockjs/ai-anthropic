import type { ModelConfig, ModelPricing } from "@warlock.js/ai";
import type { ClientOptions } from "@anthropic-ai/sdk";

/**
 * Configuration for the Anthropic SDK adapter.
 *
 * `provider` labels the SDK with the upstream you're pointing at —
 * useful when the same Anthropic-compatible client is wired to a
 * gateway or proxy. The label flows through to `ModelContract.provider`
 * and surfaces on `AgentReport.model`, logs, and any middleware that
 * branches on provider identity. Defaults to `"anthropic"` when omitted.
 *
 * `pricing` is an optional SDK-level registry keyed by model name. One
 * source of truth per provider; matches how providers publish pricing
 * tables. Resolution at `model()` call time: per-model `pricing` (on
 * `AnthropicModelConfig`) > this SDK registry > `undefined`. When
 * neither is set, `Usage.cost` stays `undefined` on every report this
 * SDK's models produce.
 *
 * @example
 * new AnthropicSDK({ apiKey: process.env.ANTHROPIC_API_KEY! });
 *
 * @example
 * // SDK-level pricing registry — pricing in USD per million tokens.
 * new AnthropicSDK({
 *   apiKey,
 *   pricing: {
 *     "claude-haiku-4-5":  { input: 1,  output: 5,  cachedInput: 0.1 },
 *     "claude-sonnet-4-6": { input: 3,  output: 15, cachedInput: 0.3 },
 *   },
 * });
 */
export type AnthropicSDKConfig = ClientOptions & {
  provider?: string;
  /**
   * Per-model USD pricing registry, keyed by model name. Surfaced onto
   * every `AnthropicModel` produced by `model()`; per-model
   * `AnthropicModelConfig.pricing` still wins when both are set.
   */
  pricing?: Record<string, ModelPricing>;
};

/**
 * Per-model configuration for `AnthropicSDK.model()`. Extends the
 * neutral `ModelConfig` with Anthropic-specific capability overrides.
 *
 * @example
 * anthropic.model({ name: "claude-sonnet-4-6" });               // vision auto-true
 * anthropic.model({ name: "custom-proxy-model", vision: false }); // dev override
 */
export type AnthropicModelConfig = ModelConfig & {
  /**
   * Override the auto-inferred vision capability. When omitted, the
   * adapter checks the model name against the known Claude vision
   * families (see `known-vision-models.ts`). Setting `true` or `false`
   * explicitly always wins over inference — useful for proxied
   * deployments or capability-degraded testing.
   */
  vision?: boolean;
  /**
   * Override the inferred `structuredOutput` capability. When omitted,
   * the adapter treats the model as capable and forwards
   * `responseSchema` via Anthropic's native `output_config.format`
   * (JSON-schema structured outputs). Set to `false` for models or
   * proxies that don't support native structured output — the agent
   * then re-injects a soft schema hint into the system prompt instead.
   */
  structuredOutput?: boolean;
  /**
   * Opt into Anthropic prompt caching for the (static) tool definitions.
   * When `true`, the adapter marks the tools block with
   * `cache_control: { type: "ephemeral" }` so multi-trip / multi-turn
   * agents reuse the tool schemas at the ~0.1x cache-read rate instead
   * of re-sending them at full price every trip.
   *
   * Off by default: a cache *write* costs ~1.25x, so it only pays off
   * when the cached prefix is read at least twice (multi-trip agents).
   * A one-shot agent with no tools is unaffected — no tools, no marker,
   * no surcharge. The system prompt is intentionally NOT cached here:
   * it carries per-turn placeholders, so a breakpoint there would pay
   * the write surcharge every turn with no reads.
   *
   * Independent of this flag, a per-call
   * `ModelCallOptions.cacheControl.breakpoints` still places a cache
   * breakpoint on the system prompt — that hint is opt-in per request.
   */
  promptCaching?: boolean;
  /**
   * Override the inferred `reasoning` capability. When omitted, the
   * adapter advertises reasoning as supported (every modern Claude model
   * accepts Anthropic extended thinking via `thinking`), so a per-call
   * `ModelCallOptions.reasoning` is forwarded as
   * `thinking: { type: "enabled", budget_tokens }`. Set to `false` for
   * proxied deployments or older targets that reject the `thinking`
   * param — the adapter then ignores reasoning options rather than
   * sending an unsupported field.
   */
  reasoning?: boolean;
};
