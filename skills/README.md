# `@warlock.js/ai-anthropic` — skills index

Per-task skills. All cross-references use the form `@warlock.js/<pkg>/<skill>/SKILL.md`.

## Skills

### [`setup-anthropic/`](./setup-anthropic/SKILL.md)

Wire @warlock.js/ai-anthropic — new AnthropicSDK({apiKey, baseURL?, provider?}) for Claude, .model({name, vision?, structuredOutput?, reasoning?, thinkingMode?, promptCaching?, maxTokens?}). System-prompt hoisting, max_tokens required (default 4096), extended thinking via options.reasoning → adaptive thinking + output_config.effort (current/unknown models) or budget_tokens (Haiku 4.5, 4.6 and older; override with thinkingMode), prompt caching via promptCaching + options.cacheControl with cost-truth usage (cachedTokens/cacheWriteTokens), no first-party embeddings. Load when wiring a Claude-backed model into a @warlock.js agent.
