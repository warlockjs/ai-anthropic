# `@warlock.js/ai-anthropic` — skills index

Per-task skills. All cross-references use the form `@warlock.js/<pkg>/<skill>/SKILL.md`.

## Skills

### [`setup-anthropic/`](./setup-anthropic/SKILL.md)

Wire @warlock.js/ai-anthropic — new AnthropicSDK({apiKey, baseURL?, provider?}) for Claude, .model({name, vision?, structuredOutput?, reasoning?, promptCaching?, maxTokens?}). System-prompt hoisting, max_tokens required (default 4096), extended thinking via options.reasoning → thinking budget_tokens, prompt caching via promptCaching + options.cacheControl with cost-truth usage (cachedTokens/cacheWriteTokens), no first-party embeddings. Load when wiring a Claude-backed model into a @warlock.js agent.
