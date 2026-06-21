import type { ToolConfig } from "@warlock.js/ai";
import { describe, expect, it } from "vitest";
import { toAnthropicTools } from "./to-anthropic-tools";

/**
 * Minimal Standard Schema whose `~standard.jsonSchema.input()` returns
 * a fixed JSON Schema — enough for `extractJsonSchema` to resolve a
 * real object schema without pulling in a schema library.
 */
function schemaTool(name: string, jsonSchema: Record<string, unknown>): ToolConfig<unknown, unknown> {
  return {
    name,
    description: `${name} tool`,
    input: {
      "~standard": {
        version: 1,
        vendor: "test",
        validate: (value: unknown) => ({ value }),
        jsonSchema: { input: () => jsonSchema },
      },
    } as unknown as ToolConfig<unknown, unknown>["input"],
    execute: async (value: unknown) => value,
  };
}

describe("toAnthropicTools", () => {
  it("returns undefined for empty or missing tool lists", () => {
    expect(toAnthropicTools(undefined)).toBeUndefined();
    expect(toAnthropicTools([])).toBeUndefined();
  });

  it("maps name, description, and object input_schema", () => {
    const objectSchema = {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    };

    const tools = toAnthropicTools([schemaTool("getWeather", objectSchema)]);

    expect(tools).toEqual([
      {
        name: "getWeather",
        description: "getWeather tool",
        input_schema: objectSchema,
      },
    ]);
  });

  it("degrades a non-object schema to a parameterless object schema", () => {
    const tools = toAnthropicTools([
      schemaTool("listAll", { type: "array", items: { type: "string" } }),
    ]);

    expect(tools?.[0].input_schema).toEqual({ type: "object" });
  });

  it("maps multiple tools preserving order", () => {
    const objectSchema = { type: "object", properties: {} };
    const tools = toAnthropicTools([
      schemaTool("first", objectSchema),
      schemaTool("second", objectSchema),
    ]);

    expect(tools?.map((tool) => tool.name)).toEqual(["first", "second"]);
  });

  it("degrades to an object schema when the extractor yields no JSON Schema", () => {
    const tool: ToolConfig<unknown, unknown> = {
      name: "noSchema",
      description: "no schema tool",
      // No `~standard` slot → extractJsonSchema returns undefined.
      input: {} as unknown as ToolConfig<unknown, unknown>["input"],
      execute: async (value: unknown) => value,
    };

    expect(toAnthropicTools([tool])?.[0].input_schema).toEqual({ type: "object" });
  });

  it("forwards an undefined description verbatim", () => {
    const tool: ToolConfig<unknown, unknown> = {
      name: "undocumented",
      description: undefined as unknown as string,
      input: {
        "~standard": {
          version: 1,
          vendor: "test",
          validate: (value: unknown) => ({ value }),
          jsonSchema: { input: () => ({ type: "object", properties: {} }) },
        },
      } as unknown as ToolConfig<unknown, unknown>["input"],
      execute: async (value: unknown) => value,
    };

    const mapped = toAnthropicTools([tool]);

    expect(mapped?.[0].name).toBe("undocumented");
    expect(mapped?.[0].description).toBeUndefined();
  });

  it("keeps an object schema with extra keywords intact (required, additionalProperties)", () => {
    const richSchema = {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
      additionalProperties: false,
    };

    const tools = toAnthropicTools([schemaTool("strict", richSchema)]);

    expect(tools?.[0].input_schema).toEqual(richSchema);
  });
});
