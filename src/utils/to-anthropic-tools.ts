import { extractJsonSchema, type ToolConfig } from "@warlock.js/ai";
import type Anthropic from "@anthropic-ai/sdk";

/**
 * Convert vendor-neutral `ToolConfig[]` into Anthropic's `tools` array.
 * Uses the shared `extractJsonSchema` helper; Anthropic requires the
 * input schema to be a JSON-Schema object, so a non-object extraction
 * is coerced into an empty-object schema rather than rejected — the
 * tool still registers and the model simply sees no parameters.
 *
 * @example
 * const tools = toAnthropicTools([weatherTool, calculatorTool]);
 * await client.messages.create({ model, max_tokens, messages, tools });
 */
export function toAnthropicTools(
  tools: ToolConfig<unknown, unknown>[] | undefined,
): Anthropic.Tool[] | undefined {
  if (!tools || tools.length === 0) {
    return undefined;
  }

  return tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    input_schema: toInputSchema(tool.input),
  }));
}

/**
 * Coerce the extracted JSON Schema into Anthropic's `Tool.InputSchema`
 * shape (root must be `{ type: "object" }`). Anything that isn't an
 * object schema degrades to a parameterless object so registration
 * never fails on a malformed extractor result.
 */
function toInputSchema(input: ToolConfig<unknown, unknown>["input"]): Anthropic.Tool.InputSchema {
  const schema = extractJsonSchema(input);

  if (schema && schema.type === "object") {
    return schema as Anthropic.Tool.InputSchema;
  }

  return { type: "object" };
}
