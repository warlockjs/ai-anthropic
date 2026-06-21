/**
 * Model-name prefixes for Claude families that accept image input
 * (vision) on the Messages API.
 *
 * Every Claude 3, Claude 3.5/3.7, and Claude 4 family model is
 * multimodal, so the list covers both the dotted legacy naming
 * (`claude-3-haiku-...`, `claude-3-5-sonnet-...`) and the current
 * `claude-<tier>-4-*` naming (`claude-opus-4-7`, `claude-sonnet-4-6`,
 * `claude-haiku-4-5`). Pre-3 families (`claude-2`, `claude-instant`)
 * are text-only and intentionally absent.
 *
 * Matched as a prefix so dated variants (`claude-opus-4-20250514`) are
 * covered without listing every release tag. Devs can always override
 * per-model via `anthropic.model({ name, vision: true | false })` —
 * explicit config wins over inference in either direction.
 */
const VISION_CAPABLE_PREFIXES = [
  "claude-3",
  "claude-4",
  "claude-opus-4",
  "claude-sonnet-4",
  "claude-haiku-4",
];

/**
 * Infer whether a given Claude model name supports vision based on the
 * known-prefix list. Unknown models default to `false` so that passing
 * an image attachment to an unsupported model surfaces a clear,
 * agent-side capability error instead of an opaque Anthropic 400.
 *
 * @example
 * inferVisionCapability("claude-sonnet-4-6");      // → true
 * inferVisionCapability("claude-3-5-sonnet-latest"); // → true
 * inferVisionCapability("claude-2.1");             // → false
 * inferVisionCapability("custom-proxy-llm");       // → false
 */
export function inferVisionCapability(modelName: string): boolean {
  const normalized = modelName.toLowerCase();

  return VISION_CAPABLE_PREFIXES.some((prefix) => normalized.startsWith(prefix));
}
