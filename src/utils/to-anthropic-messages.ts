import type { ContentPart, Message } from "@warlock.js/ai";
import type Anthropic from "@anthropic-ai/sdk";

/**
 * Result of splitting a vendor-neutral `Message[]` for the Anthropic
 * Messages API: the system prompt is hoisted to a top-level `system`
 * string (Anthropic has no `"system"` role inside `messages`), and the
 * remaining turns are mapped to `MessageParam[]`.
 */
export type AnthropicMessages = {
  system: string | undefined;
  messages: Anthropic.MessageParam[];
};

const ANTHROPIC_IMAGE_MEDIA_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"] as const;

type AnthropicImageMediaType = (typeof ANTHROPIC_IMAGE_MEDIA_TYPES)[number];

/**
 * Convert vendor-neutral `Message[]` into Anthropic's request shape.
 *
 * Anthropic differs from the OpenAI Chat protocol in three ways this
 * function absorbs:
 *
 * 1. **No `system` role.** System messages are concatenated (newline-
 *    separated) and returned separately as the top-level `system`
 *    parameter.
 * 2. **Tool results are `user` turns.** A neutral `tool` message becomes
 *    a `user` message whose content is a single `tool_result` block
 *    keyed by `tool_use_id`.
 * 3. **Tool calls are `tool_use` content blocks.** An assistant message
 *    carrying `toolCalls` becomes an `assistant` message whose content
 *    is an optional leading `text` block followed by one `tool_use`
 *    block per call.
 *
 * Consecutive same-role turns are left as-is — the Messages API merges
 * them server-side.
 *
 * @example
 * const { system, messages } = toAnthropicMessages([
 *   { role: "system", content: "Be concise." },
 *   { role: "user", content: "Hi" },
 * ]);
 * // system === "Be concise."
 * // messages === [{ role: "user", content: "Hi" }]
 */
export function toAnthropicMessages(messages: Message[]): AnthropicMessages {
  const systemParts: string[] = [];
  const mapped: Anthropic.MessageParam[] = [];

  for (const message of messages) {
    if (message.role === "system") {
      systemParts.push(stringifyContent(message.content));

      continue;
    }

    if (message.role === "tool") {
      mapped.push({
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: message.toolCallId ?? "",
            content: stringifyContent(message.content),
          },
        ],
      });

      continue;
    }

    if (message.role === "assistant" && message.toolCalls && message.toolCalls.length > 0) {
      const blocks: Anthropic.ContentBlockParam[] = [];
      const text = stringifyContent(message.content);

      if (text) {
        blocks.push({ type: "text", text });
      }

      for (const toolCall of message.toolCalls) {
        blocks.push({
          type: "tool_use",
          id: toolCall.id,
          name: toolCall.name,
          input: toolCall.input ?? {},
        });
      }

      mapped.push({ role: "assistant", content: blocks });

      continue;
    }

    if (message.role === "user" && Array.isArray(message.content)) {
      mapped.push({
        role: "user",
        content: message.content.map(toAnthropicContentBlock),
      });

      continue;
    }

    mapped.push({
      role: message.role === "assistant" ? "assistant" : "user",
      content: stringifyContent(message.content),
    });
  }

  return {
    system: systemParts.length > 0 ? systemParts.join("\n\n") : undefined,
    messages: mapped,
  };
}

/**
 * Multipart content is only meaningful on user messages — for any other
 * role collapse a `ContentPart[]` to its concatenated text so the wire
 * format stays valid. Plain strings pass through unchanged.
 */
function stringifyContent(content: string | ContentPart[]): string {
  if (typeof content === "string") {
    return content;
  }

  return content
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text)
    .join("");
}

/**
 * Map a single resolved `ContentPart` to an Anthropic content block.
 * Text passes straight through; images become an `image` block with a
 * `url` source (remote) or a `base64` source (inlined bytes). The
 * agent has already resolved every attachment before it reaches here,
 * so this never reads files or fetches URLs.
 */
function toAnthropicContentBlock(part: ContentPart): Anthropic.ContentBlockParam {
  if (part.type === "text") {
    return { type: "text", text: part.text };
  }

  if ("url" in part.source) {
    return { type: "image", source: { type: "url", url: part.source.url } };
  }

  return {
    type: "image",
    source: {
      type: "base64",
      media_type: part.source.mediaType as AnthropicImageMediaType,
      data: part.source.base64,
    },
  };
}
