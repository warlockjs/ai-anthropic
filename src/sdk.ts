import Anthropic from "@anthropic-ai/sdk";
import type { ModelContract, ModelPricing, SDKAdapterContract } from "@warlock.js/ai";
import { approximateTokenCount } from "@warlock.js/ai";
import type { AnthropicModelConfig, AnthropicSDKConfig } from "./config.type";
import { AnthropicModel } from "./model";

/**
 * Anthropic-backed implementation of `SDKAdapterContract`.
 *
 * **Role.** The package entry point for Claude models via the official
 * `@anthropic-ai/sdk`. A single `AnthropicSDK` instance holds one live
 * `Anthropic` client, shared by every `ModelContract` it produces via
 * `model()`. Users construct one SDK per account and reuse it across
 * all agents, workflows, and supervisors that target Anthropic.
 *
 * **Responsibility.**
 * - Owns: a long-lived `Anthropic` client (authentication, base URL)
 *   and its lifetime scope. Factory for `AnthropicModel` instances —
 *   each model call gets a reference to the same client.
 * - Does NOT own: anything per-call (tool execution, message history,
 *   streaming loop) — those live in `AnthropicModel` and the agent
 *   runtime. Does NOT implement `embedder()`: Anthropic ships no
 *   first-party embeddings API (the contract marks it optional).
 *
 * Modeled as a class (see §4.2 of code-style.md — "long-lived state
 * across many calls"): the `Anthropic` client is heavy to construct
 * and designed to be reused; keeping it on `this` makes that reuse
 * explicit and aligns with the `new Anthropic(...)` upstream
 * convention.
 *
 * @example
 * const anthropic = new AnthropicSDK({ apiKey: process.env.ANTHROPIC_API_KEY! });
 * const model = anthropic.model({ name: "claude-sonnet-4-6", temperature: 0.7 });
 * const tokens = await anthropic.count("Hello world");
 *
 * @example
 * // Compose into an `ai.anthropic` namespace for ergonomic agent wiring
 * const ai = { agent, tool, systemPrompt, anthropic: new AnthropicSDK({ apiKey }) };
 * const myAgent = ai.agent({ model: ai.anthropic.model({ name: "claude-haiku-4-5" }) });
 */
export class AnthropicSDK implements SDKAdapterContract {
  private readonly client: Anthropic;
  private readonly provider: string;
  private readonly pricing?: Record<string, ModelPricing>;

  public constructor(config: AnthropicSDKConfig) {
    // Peel off the framework-only keys and forward every other upstream
    // `ClientOptions` (timeout, maxRetries, defaultHeaders, fetch, …) verbatim
    // — they type-check, so dropping them is a silent footgun. Mirrors the
    // Bedrock/Google/Ollama adapters.
    const { provider, pricing, ...clientOptions } = config;

    this.client = new Anthropic(clientOptions);
    this.provider = provider ?? "anthropic";
    this.pricing = pricing;
  }

  /**
   * Build an `AnthropicModel` bound to this SDK's client. Each call
   * returns a fresh model instance, but all instances share the
   * underlying `Anthropic` client — connection pools, rate limits, and
   * authentication stay unified across every model produced here. The
   * SDK's `provider` label is forwarded so every model self-identifies
   * as coming from the same upstream.
   *
   * Pricing resolution: per-model `config.pricing` wins; otherwise the
   * SDK-level registry entry keyed by `config.name`; otherwise
   * `undefined` (no cost computed).
   */
  public model(config: AnthropicModelConfig): ModelContract {
    const resolvedPricing = config.pricing ?? this.pricing?.[config.name];
    const resolvedConfig: AnthropicModelConfig =
      resolvedPricing === config.pricing ? config : { ...config, pricing: resolvedPricing };

    return new AnthropicModel(this.client, resolvedConfig, this.provider);
  }

  /**
   * Rough token-count estimate for a given text. Uses the
   * character-heuristic (`approximateTokenCount`) from the core package
   * — good enough for budgeting and quota guards, not for billing.
   * Anthropic does expose a `messages.countTokens` endpoint, but that
   * is a network round-trip; `count()` is intentionally offline and
   * synchronous-cost. The optional model id is reserved for future
   * per-model tokenizer dispatch; currently ignored.
   */
  public async count(text: string, _model?: string): Promise<number> {
    return approximateTokenCount(text);
  }
}
