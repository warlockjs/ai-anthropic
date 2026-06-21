import type { FinishReason } from "@warlock.js/ai";

const stopReasonMap: Record<string, FinishReason> = {
  end_turn: "stop",
  stop_sequence: "stop",
  max_tokens: "length",
  tool_use: "tool_calls",
};

/**
 * Map Anthropic's `stop_reason` to the normalized `FinishReason` union.
 *
 * `end_turn` / `stop_sequence` are both natural stops. `max_tokens`
 * maps to `length`. `tool_use` maps to `tool_calls`. Everything else —
 * `refusal` (policy intervention), `pause_turn` (incomplete
 * long-running turn), `null`, or any value a future API version adds —
 * falls through to `"error"` so the agent treats the trip as a
 * non-clean terminal rather than silently accepting a partial result.
 *
 * @example
 * mapStopReason("end_turn");  // "stop"
 * mapStopReason("tool_use");  // "tool_calls"
 * mapStopReason("refusal");   // "error"
 * mapStopReason(null);        // "error"
 */
export function mapStopReason(raw: string | null | undefined): FinishReason {
  return stopReasonMap[raw ?? ""] ?? "error";
}
