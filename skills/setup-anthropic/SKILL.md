---
name: setup-anthropic
description: 'Wire @warlock.js/ai-anthropic — new AnthropicSDK({apiKey, baseURL?, provider?}) for Claude, .model({name, vision?, structuredOutput?, reasoning?, thinkingMode?, promptCaching?, maxTokens?}). System-prompt hoisting, max_tokens required (default 4096), extended thinking via options.reasoning → adaptive thinking + output_config.effort (current/unknown models) or budget_tokens (Haiku 4.5, 4.6 and older; override with thinkingMode), prompt caching via promptCaching + options.cacheControl, cost-truth usage (cachedTokens/cacheWriteTokens), no first-party embeddings. Triggers: `AnthropicSDK`, `anthropic.model`, `anthropic.count`, `maxTokens`, `reasoning`, `thinking`, `budget_tokens`, `thinkingMode`, `adaptive`, `cacheControl`, `promptCaching`, `cachedTokens`, `cacheWriteTokens`, `claude-sonnet-4-6`, `claude-haiku-4-5`, `claude-opus-4-7`; "wire claude into warlock agent", "configure anthropic provider", "use claude sonnet", "anthropic gateway baseURL", "claude extended thinking", "anthropic prompt caching cost"; typical import `import { AnthropicSDK } from "@warlock.js/ai-anthropic"`. Skip: embeddings — the `setup-openai` topic of the `warlock-js-ai-openai` skill; sibling adapters `@warlock.js/ai-openai`, `@warlock.js/ai-bedrock`, `@warlock.js/ai-google`, `@warlock.js/ai-ollama`; raw `@anthropic-ai/sdk`; Vercel `@ai-sdk/anthropic`.'
---

# `@warlock.js/ai-anthropic`

Provider adapter that turns Anthropic's Messages API into a vendor-neutral `ModelContract`. Mirrors `@warlock.js/ai-openai` — same shape, Claude-specific wire mapping.

## Construction

```ts
import { AnthropicSDK } from "@warlock.js/ai-anthropic";

const anthropic = new AnthropicSDK({ apiKey: process.env.ANTHROPIC_API_KEY! });

// Proxied / gateway endpoints speaking the Anthropic protocol:
const proxied = new AnthropicSDK({
  apiKey: process.env.GATEWAY_KEY!,
  baseURL: "https://gateway.internal/anthropic",
  provider: "anthropic-proxy",
});
```

`AnthropicSDK` is a class with a long-lived `Anthropic` client. `provider` defaults to `"anthropic"`.

## Producing a model

```ts
anthropic.model({ name: "claude-sonnet-4-6" }); // common case
anthropic.model({ name: "claude-haiku-4-5", temperature: 0.2 }); // sampling controls
anthropic.model({ name: "claude-opus-4-7", maxTokens: 8192 }); // raise the cap
```

## Capabilities — what's auto-set

| Flag               | Default                                                                                                                                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `structuredOutput` | `true` (via Anthropic's native `output_config.format`)                                                                                                                                                   |
| `vision`           | Inferred from model name. `true` for Claude 3 / 3.5 / 3.7 / 4 and Claude 5-series (`claude-opus-5*`, `claude-sonnet-5*`, `claude-fable-5*`, `claude-mythos-5*`); `false` for pre-3 and unknown ids.      |
| `reasoning`        | `true` — forwarded in the shape the model accepts (see [Extended thinking](#extended-thinking-reasoning)). Override with `reasoning: false` for legacy/proxied targets that reject the `thinking` param. |
| `promptCaching`    | `true` — the adapter places `cache_control` breakpoints and reports both cache reads (`Usage.cachedTokens`) and writes (`Usage.cacheWriteTokens`).                                                       |
| `pdf`              | `true` — the Messages API accepts PDF/document content blocks on vision-capable models.                                                                                                                  |
| `audio`            | absent (`false`) — Anthropic has no audio-input block, so the agent rejects audio attachments upfront.                                                                                                   |

Explicit config always wins (`structuredOutput`, `vision`, `reasoning`, `thinkingMode`).

## Pricing & cost

Pricing is opt-in. Supply USD-per-million-token rates and every report this SDK's models produce carries `Usage.cost`; omit them and `cost` stays `undefined` (honest absence, never a false zero). Two declaration sites:

```ts
// SDK-level registry — one source of truth, keyed by model name:
const anthropic = new AnthropicSDK({
  apiKey,
  pricing: {
    "claude-haiku-4-5": { input: 1, output: 5, cachedInput: 0.1 },
    "claude-sonnet-4-6": { input: 3, output: 15, cachedInput: 0.3 },
  },
});

// Per-model override (multi-tenant / contract-specific rates):
anthropic.model({ name: "claude-sonnet-4-6", pricing: { input: 3, output: 15 } });
```

Resolution at `model()` time: per-model `pricing` > SDK registry entry for that name > `undefined`. `ModelPricing` is `{ input, output, cachedInput?, cachedOutput? }`.

## Tokens & usage accounting

Every `ModelResponse.usage` (and the streaming terminal `done`) is normalized to the neutral `Usage` shape. Anthropic reports `input_tokens` / `output_tokens` separately with no pre-summed total, so `total` is computed (`input + output`). The two cache channels are surfaced only when non-zero:

| Neutral field            | Anthropic source              | Meaning / billing                                                                                                                             |
| ------------------------ | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `usage.cachedTokens`     | `cache_read_input_tokens`     | Subset of input served from the prompt cache. Bills at `cachedInput` (falls back to `input` when unset).                                      |
| `usage.cacheWriteTokens` | `cache_creation_input_tokens` | Input tokens **written** to the cache on this call (the ~1.25x write surcharge). Bills at `cachedOutput` when set.                            |
| `usage.reasoningTokens`  | — _(not reported separately)_ | Anthropic bills extended-thinking tokens **inside** `output_tokens`, so this stays unset — populating it would double-count against `output`. |

In streaming, `cachedTokens` / `cacheWriteTokens` are seeded from `message_start` and then **overwritten by the cumulative counts on `message_delta`** when those are present and non-zero, so the terminal `done` carries the final tally.

## `max_tokens` is required

Unlike OpenAI, Anthropic **requires** `max_tokens` on every request. Resolution: per-call `options.maxTokens` > `config.maxTokens` > **default `4096`**. A caller who never sets a cap still gets a complete answer instead of a 400.

With extended thinking enabled and no explicit cap, the adapter reserves answer room in both modes: `max_tokens = thinking budget + 4096`. In budget mode an explicit `maxTokens` must be greater than `budget_tokens` or the call throws before it reaches Anthropic.

**Tiny caps + adaptive thinking:** thinking tokens count inside `max_tokens`, so a very small cap can be spent entirely on thinking. Live check: Sonnet 5.5 with `maxTokens: 60` returned `finishReason: "length"` and empty content. For tiny outputs give thinking room (raise `maxTokens`) or turn reasoning off (`reasoning: false` on the model, or no `options.reasoning`).

## System prompt

Anthropic has no `"system"` role inside `messages`. The adapter hoists every neutral `role: "system"` message into the top-level `system` parameter (multiple system messages join with a blank line). Transparent to the agent.

## Tool calls

- Outgoing: vendor-neutral tools → Anthropic `tools` with `input_schema` (non-object schemas degrade to a parameterless object so registration never fails).
- Assistant tool calls in history → `assistant` message with an optional leading `text` block + one `tool_use` block per call.
- Tool results (`role: "tool"`) → a `user` turn carrying a single `tool_result` block keyed by `tool_use_id`.
- Response `tool_use` blocks → neutral `toolCalls`; `stop_reason: "tool_use"` → `finishReason: "tool_calls"`.

## Structured output

When the agent passes `responseSchema` and the model is `structuredOutput`-capable, an **object-root** schema is forwarded as `output_config: { format: { type: "json_schema", schema } }` (Anthropic native). Non-object schemas or `structuredOutput: false` omit it; the agent's soft system-prompt hint + client-side `validate()` still enforce shape.

## Extended thinking (reasoning)

When the model is `reasoning`-capable (default `true`) and the agent passes `options.reasoning`, the adapter forwards Anthropic thinking in one of two shapes, chosen per model (`thinkingMode`):

| Mode         | Models (inferred from the name)                                                                                                 | Wire shape                                                |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `"adaptive"` | Current Claude models — Opus 4.7 / 4.8 / 5 / 5.5, Sonnet 5 / 5.5, Fable 5 / 5.1 — **and any id the adapter does not recognise** | `thinking: { type: "adaptive" }` + `output_config.effort` |
| `"budget"`   | Haiku 4.5, Opus 4.6, Sonnet 4.6, and every older model (Claude 3.x, 4.0 / 4.1 / 4.5, 2, instant)                                | `thinking: { type: "enabled", budget_tokens }`            |

Adaptive models reject `budget_tokens` with a 400, so the shape matters.

- **Adaptive:** `reasoning.effort` (`"low" | "medium" | "high"`) → `output_config.effort` of the same name. With only `reasoning.maxTokens` set, effort is `"medium"`; `maxTokens` then only sizes the reserved `max_tokens` room, it is not sent as a budget.
- **Budget:** `reasoning.maxTokens` → `budget_tokens` verbatim; `reasoning.effort` → a tiered budget (`1024` / `4096` / `12000`) when no `maxTokens` is given. Any resolved budget is floored at Anthropic's `1024` minimum.
- `reasoning` with neither `effort` nor `maxTokens`, or `effort: "none"`, emits nothing.

### `thinkingMode` override

`anthropic.model({ name, thinkingMode: "adaptive" | "budget" })` beats the name-based detection. Use it when the id does not look like an Anthropic id: region-prefixed / Bedrock-style ids (`us.anthropic.claude-…`) and custom or gateway ids are _unrecognised_, so they default to adaptive. If they front an older model (Haiku 4.5, 4.6 or earlier) set `thinkingMode: "budget"`. It can also move Opus/Sonnet 4.6 onto adaptive. The `AnthropicThinkingMode` type is exported.

```ts
import { AnthropicSDK, type AnthropicThinkingMode } from "@warlock.js/ai-anthropic";

const anthropic = new AnthropicSDK({ apiKey: process.env.ANTHROPIC_API_KEY! });

const mode: AnthropicThinkingMode = "budget";

anthropic.model({ name: "claude-opus-5-5" }); // adaptive (detected)
anthropic.model({ name: "claude-haiku-4-5" }); // budget (detected)
anthropic.model({ name: "us.anthropic.claude-haiku-4-5", thinkingMode: mode }); // region-prefixed id of an older model
anthropic.model({ name: "claude-sonnet-4-6", thinkingMode: "adaptive" }); // opt 4.6 into adaptive
```

### Temperature

`temperature` is **not sent** to models after Opus 4.6 (adaptive mode) — they reject any non-default value with a 400. It is also dropped on any request that enables thinking, since Anthropic 400s on thinking plus an explicit temperature. So it is forwarded only in budget mode on requests without thinking; `thinkingMode` decides this too. Set `reasoning: false` on the model config to suppress the thinking param entirely for proxied/legacy targets.

```ts
await agent.execute("Prove it.", { reasoning: { effort: "high" } });
// adaptive model: thinking { type: "adaptive" }, output_config.effort "high"
// budget model:   thinking { type: "enabled", budget_tokens: 12000 }
await model.complete(messages, { reasoning: { maxTokens: 8000 } });
// adaptive model: effort "medium"; budget model: budget_tokens 8000
```

When a thinking-enabled response also makes tool calls, its signed `thinking` and `redacted_thinking` blocks are retained in each call's `providerMetadata.anthropic.thinkingBlocks`. The adapter replays those opaque blocks automatically on the next assistant turn, preserving Anthropic's required signatures; application code should leave that metadata intact when persisting message history.

## Prompt caching (`cache_control`)

Caching is opt-in: `model({ promptCaching: true })` sends request-level
`cache_control: { type: "ephemeral" }` when a call has no explicit
`options.cacheControl.breakpoints`. Anthropic then marks the last cacheable
block itself, caching the longest reusable request prefix (including growing
message history). A caller-supplied breakpoint remains authoritative, so its
request has no top-level automatic marker.

Set `promptCacheTtl: "1h"` on the model to apply a one-hour TTL to automatic
caching and emitted explicit markers. Omit it (or use `"5m"`) for Anthropic's
default five-minute TTL; the one-hour cache has a higher write cost.

Two independent breakpoint sites, both reported back via `usage.cachedTokens` / `usage.cacheWriteTokens`:

- **Tools** — set `model({ promptCaching: true })` to mark the _last_ tool definition with `cache_control: ephemeral`. One breakpoint caches the whole tool prefix; off by default since a write costs ~1.25x and only pays off across multiple trips.
- **System prompt** — a per-call `options.cacheControl.breakpoints >= 1` emits the system prompt as a `TextBlockParam` carrying `cache_control: ephemeral` (the longest stable prefix on a turn). Without the hint the system prompt stays a plain string with no marker of its own, but with `promptCaching: true` the automatic request-level marker still caches the longest reusable prefix, system prompt included (a live check read back 10,090 cached tokens of a large system prompt on the second call). With `promptCaching` off and no hint, nothing is cached, so a one-shot call never pays the write surcharge. A system prompt with per-turn placeholders changes the prefix each turn: write surcharge, no reads. No system prompt → no block.

```ts
await agent.execute("…", { cacheControl: { breakpoints: 1 } }); // cache the system prefix
```

## Multipart messages (vision)

`ContentPart[]` user content maps to Anthropic content blocks:

- `{ type: "text", text }` → `{ type: "text", text }`
- `{ type: "image", source: { url } }` → `{ type: "image", source: { type: "url", url } }`
- `{ type: "image", source: { base64, mediaType } }` → `{ type: "image", source: { type: "base64", media_type, data } }`

## Streaming

`model.stream()` drains `messages.create({ stream: true })`:

- `content_block_delta` text → `{ type: "delta", content }`
- `tool_use` blocks accumulate `input_json_delta` fragments and emit one consolidated `{ type: "tool-call", ... }` at `content_block_stop` (no partial-arg `{}` artifact — input is parsed once, whole)
- terminal `{ type: "done", finishReason, usage }` — `usage.input` from `message_start`, `usage.output` from `message_delta`, `total` computed; `cachedTokens` / `cacheWriteTokens` from `message_start` then the cumulative `message_delta` counts (see [Tokens & usage accounting](#tokens--usage-accounting))

## Finish-reason mapping

`end_turn` / `stop_sequence` → `stop` · `max_tokens` → `length` · `tool_use` → `tool_calls` · `refusal` / `pause_turn` / unknown / null → `error`.

## Errors

Raw SDK errors are wrapped into the typed `@warlock.js/ai` `AIError` hierarchy. Dispatch keys on `error.type` (Anthropic has no per-error machine `code`), falling back to HTTP status:

- `authentication_error` / `permission_error` / 401 / 403 → `ProviderAuthError`
- `rate_limit_error` / 429 → `ProviderRateLimitError` (+`retryAfter` from `retry-after`)
- `billing_error` → `QuotaExceededError`
- `invalid_request_error` with "prompt is too long" → `ContextLengthExceededError`, else `InvalidRequestError`
- 5xx / overloaded → `ProviderError`
- Connection timeouts → `ProviderTimeoutError`

## Token counting

```ts
await anthropic.count("some text"); // approximate heuristic, offline
```

## No embeddings

Anthropic ships no first-party embeddings API, so `AnthropicSDK` intentionally does **not** implement `embedder()` (the `SDKAdapterContract.embedder` slot is optional). Use `@warlock.js/ai-openai` (or another embeddings-capable adapter; see the `setup-openai` topic of the `warlock-js-ai-openai` skill) for vectors.

## When NOT to use this skill

- Direct calls to `@anthropic-ai/sdk` without going through `@warlock.js/ai` agents.
- OpenAI / Bedrock / Gemini / Ollama models — those have their own adapter packages.

## See also

- The `run-ai-agent` topic of the `warlock-js-ai` skill
- The `pick-ai-provider` topic of the `warlock-js-ai` skill
- The `handle-ai-errors` topic of the `warlock-js-ai` skill
