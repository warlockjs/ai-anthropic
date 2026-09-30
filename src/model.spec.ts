import type Anthropic from "@anthropic-ai/sdk";
import { describe, expect, it } from "vitest";
import { AnthropicModel } from "./model";

type CreateCall = { params: Anthropic.MessageCreateParams; requestOptions: unknown };

/**
 * Fake Anthropic client whose `messages.create()` records the params it
 * was called with and returns either a scripted `Message` (non-stream)
 * or an async iterable of raw stream events (stream). Lets us assert
 * the exact wire shape `AnthropicModel` sends without the network.
 *
 * `streamFactory` overrides `streamEvents` when the test needs the
 * stream itself (not the request) to throw mid-iteration.
 */
function makeFakeClient(options: {
  message?: Anthropic.Message;
  streamEvents?: Anthropic.RawMessageStreamEvent[];
  streamFactory?: () => AsyncIterable<Anthropic.RawMessageStreamEvent>;
  throws?: unknown;
}) {
  const calls: CreateCall[] = [];

  const create = async (params: Anthropic.MessageCreateParams, requestOptions?: unknown) => {
    calls.push({ params, requestOptions });

    if (options.throws) {
      throw options.throws;
    }

    if (params.stream) {
      if (options.streamFactory) {
        return options.streamFactory();
      }

      return (async function* () {
        for (const event of options.streamEvents ?? []) {
          yield event;
        }
      })();
    }

    return options.message;
  };

  const client = { messages: { create } } as unknown as Anthropic;

  return { client, calls };
}

/**
 * Drain a stream into an ordered list of every emitted chunk so a test
 * can assert on the full sequence rather than just the terminal `done`.
 */
async function collectStream(
  iterable: AsyncIterable<{ type: string }>,
): Promise<Array<Record<string, unknown>>> {
  const chunks: Array<Record<string, unknown>> = [];

  for await (const chunk of iterable) {
    chunks.push(chunk as unknown as Record<string, unknown>);
  }

  return chunks;
}

function message(overrides: Partial<Anthropic.Message> = {}): Anthropic.Message {
  return {
    id: "msg_1",
    container: null,
    model: "claude-sonnet-4-6",
    role: "assistant",
    stop_details: null,
    stop_reason: "end_turn",
    stop_sequence: null,
    type: "message",
    content: [{ type: "text", text: "hello", citations: null }],
    usage: {
      input_tokens: 5,
      output_tokens: 3,
      cache_creation_input_tokens: null,
      cache_read_input_tokens: null,
    },
    ...overrides,
  } as Anthropic.Message;
}

describe("AnthropicModel.complete()", () => {
  it("forwards model, messages, default max_tokens, temperature, and hoisted system", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6", temperature: 0.4 });

    await model.complete([
      { role: "system", content: "Be concise." },
      { role: "user", content: "hi" },
    ]);

    expect(calls[0].params.model).toBe("claude-sonnet-4-6");
    expect(calls[0].params.max_tokens).toBe(4096);
    expect(calls[0].params.temperature).toBe(0.4);
    expect(calls[0].params.system).toBe("Be concise.");
    expect(calls[0].params.messages).toEqual([{ role: "user", content: "hi" }]);
    expect(calls[0].params.stream).toBe(false);
  });

  it("config maxTokens overrides the default and per-call options override config", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, {
      name: "claude-sonnet-4-6",
      maxTokens: 512,
      temperature: 0.4,
    });

    await model.complete([{ role: "user", content: "hi" }]);
    expect(calls[0].params.max_tokens).toBe(512);

    await model.complete([{ role: "user", content: "hi" }], { maxTokens: 64, temperature: 0.9 });
    expect(calls[1].params.max_tokens).toBe(64);
    expect(calls[1].params.temperature).toBe(0.9);
  });

  it("normalizes a text response into ModelResponse shape", async () => {
    const { client } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.content).toBe("hello");
    expect(result.finishReason).toBe("stop");
    expect(result.usage).toEqual({ input: 5, output: 3, total: 8 });
    expect(result.toolCalls).toBeUndefined();
  });

  it("extracts tool_use blocks into neutral tool calls", async () => {
    const { client } = makeFakeClient({
      message: message({
        stop_reason: "tool_use",
        content: [
          { type: "text", text: "let me check", citations: null },
          { type: "tool_use", id: "tu_1", name: "getWeather", input: { city: "Cairo" } },
        ] as Anthropic.ContentBlock[],
      }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.finishReason).toBe("tool_calls");
    expect(result.content).toBe("let me check");
    expect(result.toolCalls).toEqual([
      { id: "tu_1", name: "getWeather", input: { city: "Cairo" } },
    ]);
  });

  it("round-trips signed thinking blocks before a tool-use follow-up", async () => {
    const { client, calls } = makeFakeClient({
      message: message({
        stop_reason: "tool_use",
        content: [
          { type: "thinking", thinking: "I should look this up.", signature: "sig_1" },
          { type: "tool_use", id: "tu_1", name: "getWeather", input: { city: "Cairo" } },
        ] as unknown as Anthropic.ContentBlock[],
      }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const response = await model.complete([{ role: "user", content: "weather?" }]);
    await model.complete([
      { role: "user", content: "weather?" },
      { role: "assistant", content: response.content, toolCalls: response.toolCalls },
      { role: "tool", toolCallId: "tu_1", content: "Sunny" },
    ]);

    expect(calls[1]?.params.messages).toEqual([
      { role: "user", content: "weather?" },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "I should look this up.", signature: "sig_1" },
          { type: "tool_use", id: "tu_1", name: "getWeather", input: { city: "Cairo" } },
        ],
      },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "tu_1", content: "Sunny" }],
      },
    ]);
  });

  it("replays redacted thinking data before a tool-use follow-up", async () => {
    const { client, calls } = makeFakeClient({
      message: message({
        stop_reason: "tool_use",
        content: [
          { type: "redacted_thinking", data: "encrypted-reasoning" },
          { type: "tool_use", id: "tu_1", name: "getWeather", input: {} },
        ] as unknown as Anthropic.ContentBlock[],
      }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "weather?" }]);

    expect(result.toolCalls?.[0]?.providerMetadata).toEqual({
      anthropic: { thinkingBlocks: [{ type: "redacted_thinking", data: "encrypted-reasoning" }] },
    });

    await model.complete([
      { role: "user", content: "weather?" },
      { role: "assistant", content: result.content, toolCalls: result.toolCalls },
      { role: "tool", toolCallId: "tu_1", content: "Sunny" },
    ]);

    expect(calls[1]?.params.messages?.[1]).toEqual({
      role: "assistant",
      content: [
        { type: "redacted_thinking", data: "encrypted-reasoning" },
        { type: "tool_use", id: "tu_1", name: "getWeather", input: {} },
      ],
    });
  });

  it("surfaces cache_read_input_tokens as cachedTokens when non-zero", async () => {
    const { client } = makeFakeClient({
      message: message({
        usage: {
          input_tokens: 10,
          output_tokens: 4,
          cache_read_input_tokens: 6,
          cache_creation_input_tokens: null,
        } as Anthropic.Usage,
      }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.usage).toEqual({ input: 10, output: 4, total: 14, cachedTokens: 6 });
  });

  it("emits native output_config for an object responseSchema when capable", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const schema = { type: "object", properties: { summary: { type: "string" } } };

    await model.complete([{ role: "user", content: "hi" }], { responseSchema: schema });

    expect((calls[0].params as { output_config?: unknown }).output_config).toEqual({
      format: { type: "json_schema", schema },
    });
  });

  it("omits output_config for non-object schemas and when structuredOutput is off", async () => {
    const { client, calls } = makeFakeClient({ message: message() });

    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });
    await model.complete([{ role: "user", content: "hi" }], {
      responseSchema: { type: "array", items: { type: "string" } },
    });
    expect((calls[0].params as { output_config?: unknown }).output_config).toBeUndefined();

    const noStruct = new AnthropicModel(client, {
      name: "claude-sonnet-4-6",
      structuredOutput: false,
    });
    await noStruct.complete([{ role: "user", content: "hi" }], {
      responseSchema: { type: "object", properties: {} },
    });
    expect((calls[1].params as { output_config?: unknown }).output_config).toBeUndefined();
  });

  it("rethrows a wrapped typed error on failure", async () => {
    const { client } = makeFakeClient({ throws: { status: 500, message: "boom" } });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await expect(model.complete([{ role: "user", content: "hi" }])).rejects.toMatchObject({
      code: "PROVIDER_ERROR",
    });
  });

  it("omits temperature and system when neither is configured nor supplied", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await model.complete([{ role: "user", content: "hi" }]);

    expect("temperature" in calls[0].params).toBe(false);
    expect("system" in calls[0].params).toBe(false);
    expect("tools" in calls[0].params).toBe(false);
    expect("output_config" in calls[0].params).toBe(false);
  });

  it("treats a configured temperature of 0 as present (not omitted by falsy check)", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6", temperature: 0 });

    await model.complete([{ role: "user", content: "hi" }]);

    expect(calls[0].params.temperature).toBe(0);
  });

  it("treats a per-call maxTokens of 0 as present (overrides config and default)", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6", maxTokens: 100 });

    await model.complete([{ role: "user", content: "hi" }], { maxTokens: 0 });

    expect(calls[0].params.max_tokens).toBe(0);
  });

  it("forwards the AbortSignal as the request's second argument", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });
    const controller = new AbortController();

    await model.complete([{ role: "user", content: "hi" }], { signal: controller.signal });

    expect(calls[0].requestOptions).toEqual({ signal: controller.signal });
  });

  it("passes undefined request options when no signal is supplied", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await model.complete([{ role: "user", content: "hi" }]);

    expect(calls[0].requestOptions).toBeUndefined();
  });

  it("forwards mapped tools onto the request body", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await model.complete([{ role: "user", content: "hi" }], {
      tools: [
        {
          name: "ping",
          description: "ping tool",
          input: {
            "~standard": {
              version: 1,
              vendor: "test",
              validate: (value: unknown) => ({ value }),
              jsonSchema: { input: () => ({ type: "object", properties: {} }) },
            },
          },
          execute: async (value: unknown) => value,
        },
      ] as never,
    });

    expect(calls[0].params.tools).toEqual([
      { name: "ping", description: "ping tool", input_schema: { type: "object", properties: {} } },
    ]);
  });

  it("concatenates multiple text blocks into one content string", async () => {
    const { client } = makeFakeClient({
      message: message({
        content: [
          { type: "text", text: "Hello ", citations: null },
          { type: "text", text: "world", citations: null },
        ] as Anthropic.ContentBlock[],
      }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.content).toBe("Hello world");
  });

  it("returns an empty content string when the response has no text blocks", async () => {
    const { client } = makeFakeClient({
      message: message({
        stop_reason: "tool_use",
        content: [
          { type: "tool_use", id: "tu_1", name: "noop", input: { a: 1 } },
        ] as Anthropic.ContentBlock[],
      }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.content).toBe("");
    expect(result.toolCalls).toEqual([{ id: "tu_1", name: "noop", input: { a: 1 } }]);
  });

  it("extracts multiple tool_use blocks in order", async () => {
    const { client } = makeFakeClient({
      message: message({
        stop_reason: "tool_use",
        content: [
          { type: "tool_use", id: "tu_1", name: "getWeather", input: { city: "Cairo" } },
          { type: "tool_use", id: "tu_2", name: "getTime", input: { tz: "UTC" } },
        ] as Anthropic.ContentBlock[],
      }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.toolCalls).toEqual([
      { id: "tu_1", name: "getWeather", input: { city: "Cairo" } },
      { id: "tu_2", name: "getTime", input: { tz: "UTC" } },
    ]);
  });

  it("defaults a tool_use block with null input to an empty object", async () => {
    const { client } = makeFakeClient({
      message: message({
        stop_reason: "tool_use",
        content: [
          { type: "tool_use", id: "tu_1", name: "noop", input: null },
        ] as unknown as Anthropic.ContentBlock[],
      }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.toolCalls).toEqual([{ id: "tu_1", name: "noop", input: {} }]);
  });

  it("does not surface cachedTokens when cache_read_input_tokens is 0", async () => {
    const { client } = makeFakeClient({
      message: message({
        usage: {
          input_tokens: 10,
          output_tokens: 4,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: null,
        } as Anthropic.Usage,
      }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.usage).toEqual({ input: 10, output: 4, total: 14 });
    expect(result.usage.cachedTokens).toBeUndefined();
  });

  it("treats missing input/output token counts as zero", async () => {
    const { client } = makeFakeClient({
      message: message({
        usage: {
          cache_read_input_tokens: null,
          cache_creation_input_tokens: null,
        } as unknown as Anthropic.Usage,
      }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.usage).toEqual({ input: 0, output: 0, total: 0 });
  });

  it("surfaces Anthropic thinking tokens as reasoningTokens when reported", async () => {
    const { client } = makeFakeClient({
      message: message({
        usage: {
          input_tokens: 10,
          output_tokens: 14,
          output_tokens_details: { thinking_tokens: 9 },
        } as Anthropic.Usage,
      }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.usage).toEqual({ input: 10, output: 14, total: 24, reasoningTokens: 9 });
  });

  it("maps max_tokens stop_reason to the 'length' finish reason", async () => {
    const { client } = makeFakeClient({ message: message({ stop_reason: "max_tokens" }) });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.finishReason).toBe("length");
  });

  it("maps model_context_window_exceeded stop_reason to the 'length' finish reason", async () => {
    const { client } = makeFakeClient({
      message: message({ stop_reason: "model_context_window_exceeded" }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.finishReason).toBe("length");
  });

  it("maps an unknown/null stop_reason to the 'error' finish reason", async () => {
    const { client } = makeFakeClient({
      message: message({ stop_reason: null as unknown as Anthropic.Message["stop_reason"] }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.finishReason).toBe("error");
  });

  it("uses the instance provider label and exposes name/capabilities", () => {
    const { client } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" }, "bedrock-anthropic");

    expect(model.name).toBe("claude-sonnet-4-6");
    expect(model.provider).toBe("bedrock-anthropic");
    expect(model.capabilities).toEqual({
      structuredOutput: true,
      vision: true,
      reasoning: true,
      promptCaching: true,
      pdf: true,
    });
  });

  it("honors an explicit reasoning capability override", () => {
    const { client } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "custom-proxy", reasoning: false });

    expect(model.capabilities.reasoning).toBe(false);
  });

  it("does not advertise audio input as a capability", () => {
    const { client } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    expect(model.capabilities.audio).toBeUndefined();
  });

  it("surfaces cache_creation_input_tokens as cacheWriteTokens when non-zero", async () => {
    const { client } = makeFakeClient({
      message: message({
        usage: {
          input_tokens: 12,
          output_tokens: 4,
          cache_read_input_tokens: null,
          cache_creation_input_tokens: 64,
        } as Anthropic.Usage,
      }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.usage).toEqual({ input: 12, output: 4, total: 16, cacheWriteTokens: 64 });
  });

  it("surfaces both cache read and write tokens together", async () => {
    const { client } = makeFakeClient({
      message: message({
        usage: {
          input_tokens: 20,
          output_tokens: 5,
          cache_read_input_tokens: 10,
          cache_creation_input_tokens: 8,
        } as Anthropic.Usage,
      }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.usage).toEqual({
      input: 20,
      output: 5,
      total: 25,
      cachedTokens: 10,
      cacheWriteTokens: 8,
    });
  });

  it("does not surface cacheWriteTokens when cache_creation_input_tokens is 0", async () => {
    const { client } = makeFakeClient({
      message: message({
        usage: {
          input_tokens: 10,
          output_tokens: 4,
          cache_read_input_tokens: null,
          cache_creation_input_tokens: 0,
        } as Anthropic.Usage,
      }),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.usage).toEqual({ input: 10, output: 4, total: 14 });
    expect(result.usage.cacheWriteTokens).toBeUndefined();
  });

  it("maps reasoning.maxTokens to an enabled thinking budget", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await model.complete([{ role: "user", content: "hi" }], {
      reasoning: { maxTokens: 8000 },
    });

    expect((calls[0].params as { thinking?: unknown }).thinking).toEqual({
      type: "enabled",
      budget_tokens: 8000,
    });
    expect(calls[0]?.params.max_tokens).toBe(12096);
  });

  it("rejects an explicit maxTokens that is not greater than the thinking budget", async () => {
    const { client } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await expect(
      model.complete([{ role: "user", content: "hi" }], {
        maxTokens: 8000,
        reasoning: { maxTokens: 8000 },
      }),
    ).rejects.toThrow("maxTokens (8000) must be greater than thinking budget_tokens (8000)");
  });

  it("maps reasoning.effort levels to tiered thinking budgets", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await model.complete([{ role: "user", content: "hi" }], { reasoning: { effort: "low" } });
    await model.complete([{ role: "user", content: "hi" }], { reasoning: { effort: "high" } });

    expect((calls[0].params as { thinking?: { budget_tokens?: number } }).thinking).toEqual({
      type: "enabled",
      budget_tokens: 1024,
    });
    expect((calls[1].params as { thinking?: { budget_tokens?: number } }).thinking).toEqual({
      type: "enabled",
      budget_tokens: 12000,
    });
  });

  it("floors a sub-minimum thinking budget at 1024 tokens", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await model.complete([{ role: "user", content: "hi" }], { reasoning: { maxTokens: 200 } });

    expect((calls[0].params as { thinking?: { budget_tokens?: number } }).thinking).toEqual({
      type: "enabled",
      budget_tokens: 1024,
    });
  });

  it("drops temperature when extended thinking is enabled", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6", temperature: 0.7 });

    await model.complete([{ role: "user", content: "hi" }], { reasoning: { maxTokens: 2000 } });

    expect("temperature" in calls[0].params).toBe(false);
    expect((calls[0].params as { thinking?: unknown }).thinking).toBeDefined();
  });

  it("omits thinking when reasoning capability is disabled", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "custom-proxy", reasoning: false });

    await model.complete([{ role: "user", content: "hi" }], { reasoning: { maxTokens: 4000 } });

    expect("thinking" in calls[0].params).toBe(false);
  });

  it("omits thinking when reasoning has neither effort nor maxTokens", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await model.complete([{ role: "user", content: "hi" }], { reasoning: {} });

    expect("thinking" in calls[0].params).toBe(false);
    expect(calls[0].params.temperature).toBeUndefined();
  });

  it("omits thinking when effort is 'none' (explicit reasoning-off)", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await model.complete([{ role: "user", content: "hi" }], {
      reasoning: { effort: "none" },
    });

    expect("thinking" in calls[0].params).toBe(false);
  });

  it("treats effort 'none' as authoritative over an explicit maxTokens", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await model.complete([{ role: "user", content: "hi" }], {
      reasoning: { effort: "none", maxTokens: 8000 },
    });

    expect("thinking" in calls[0].params).toBe(false);
  });

  it("places a cache_control breakpoint on the system prompt when breakpoints >= 1", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await model.complete(
      [
        { role: "system", content: "Be concise." },
        { role: "user", content: "hi" },
      ],
      { cacheControl: { breakpoints: 1 } },
    );

    expect(calls[0].params.system).toEqual([
      { type: "text", text: "Be concise.", cache_control: { type: "ephemeral" } },
    ]);
  });

  it("automatically caches the longest reusable request prefix when prompt caching is enabled", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6", promptCaching: true });

    await model.complete([{ role: "user", content: "hi" }]);

    expect(calls[0].params.cache_control).toEqual({ type: "ephemeral" });
  });

  it("uses the configured cache TTL for automatic and explicit cache breakpoints", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, {
      name: "claude-sonnet-4-6",
      promptCaching: true,
      promptCacheTtl: "1h",
    });

    await model.complete([{ role: "user", content: "hi" }]);
    await model.complete(
      [
        { role: "system", content: "Be concise." },
        { role: "user", content: "hi" },
      ],
      { cacheControl: { breakpoints: 1 } },
    );

    expect(calls[0].params.cache_control).toEqual({ type: "ephemeral", ttl: "1h" });
    expect(calls[1].params.cache_control).toBeUndefined();
    expect(calls[1].params.system).toEqual([
      { type: "text", text: "Be concise.", cache_control: { type: "ephemeral", ttl: "1h" } },
    ]);
  });

  it("does not emit automatic cache control when prompt caching is disabled", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await model.complete([{ role: "user", content: "hi" }]);

    expect(calls[0].params.cache_control).toBeUndefined();
  });

  it("leaves the system prompt a plain string when no cache breakpoint is requested", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await model.complete([
      { role: "system", content: "Be concise." },
      { role: "user", content: "hi" },
    ]);

    expect(calls[0].params.system).toBe("Be concise.");
  });

  it("does not emit a system block for a cache breakpoint when there is no system prompt", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await model.complete([{ role: "user", content: "hi" }], { cacheControl: { breakpoints: 2 } });

    expect("system" in calls[0].params).toBe(false);
  });
});

describe("AnthropicModel.stream()", () => {
  it("yields text deltas then a terminal done with mapped finish reason + usage", async () => {
    const { client } = makeFakeClient({
      streamEvents: [
        { type: "message_start", message: { usage: { input_tokens: 9 } } },
        { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hel" } },
        { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "lo" } },
        { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 4 } },
        { type: "message_stop" },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const types: string[] = [];
    let done: { finishReason: string; usage: unknown } | undefined;

    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      types.push(event.type);

      if (event.type === "done") {
        done = { finishReason: event.finishReason, usage: event.usage };
      }
    }

    expect(types).toEqual(["delta", "delta", "done"]);
    expect(done).toEqual({
      finishReason: "stop",
      usage: { input: 9, output: 4, total: 13 },
    });
  });

  it("surfaces cumulative thinking tokens as reasoningTokens in done", async () => {
    const { client } = makeFakeClient({
      streamEvents: [
        {
          type: "message_start",
          message: {
            usage: {
              input_tokens: 9,
              output_tokens_details: { thinking_tokens: 2 },
            },
          },
        },
        {
          type: "message_delta",
          delta: { stop_reason: "end_turn" },
          usage: { output_tokens: 8, output_tokens_details: { thinking_tokens: 5 } },
        },
        { type: "message_stop" },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const chunks = await collectStream(model.stream([{ role: "user", content: "hi" }]));

    expect(chunks.at(-1)).toEqual({
      type: "done",
      finishReason: "stop",
      usage: { input: 9, output: 8, total: 17, reasoningTokens: 5 },
    });
  });

  it("accumulates input_json_delta and emits one tool-call at content_block_stop", async () => {
    const { client } = makeFakeClient({
      streamEvents: [
        { type: "message_start", message: { usage: { input_tokens: 2 } } },
        {
          type: "content_block_start",
          index: 0,
          content_block: { type: "tool_use", id: "tu_1", name: "getWeather", input: {} },
        },
        {
          type: "content_block_delta",
          index: 0,
          delta: { type: "input_json_delta", partial_json: '{"city":' },
        },
        {
          type: "content_block_delta",
          index: 0,
          delta: { type: "input_json_delta", partial_json: '"Cairo"}' },
        },
        { type: "content_block_stop", index: 0 },
        { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 7 } },
        { type: "message_stop" },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const toolCalls: Array<{ id: string; name: string; input: unknown }> = [];
    let finishReason = "";

    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      if (event.type === "tool-call") {
        toolCalls.push({ id: event.id, name: event.name, input: event.input });
      } else if (event.type === "done") {
        finishReason = event.finishReason;
      }
    }

    expect(toolCalls).toEqual([{ id: "tu_1", name: "getWeather", input: { city: "Cairo" } }]);
    expect(finishReason).toBe("tool_calls");
  });

  it("round-trips streamed signed thinking blocks before the tool use", async () => {
    const { client, calls } = makeFakeClient({
      message: message(),
      streamEvents: [
        { type: "message_start", message: { usage: { input_tokens: 2 } } },
        {
          type: "content_block_start",
          index: 0,
          content_block: { type: "thinking", thinking: "", signature: "" },
        },
        {
          type: "content_block_delta",
          index: 0,
          delta: { type: "thinking_delta", thinking: "Check the weather." },
        },
        {
          type: "content_block_delta",
          index: 0,
          delta: { type: "signature_delta", signature: "sig_stream" },
        },
        { type: "content_block_stop", index: 0 },
        {
          type: "content_block_start",
          index: 1,
          content_block: { type: "tool_use", id: "tu_1", name: "getWeather", input: {} },
        },
        {
          type: "content_block_delta",
          index: 1,
          delta: { type: "input_json_delta", partial_json: "{}" },
        },
        { type: "content_block_stop", index: 1 },
        { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 3 } },
        { type: "message_stop" },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const chunks = await collectStream(model.stream([{ role: "user", content: "weather?" }]));
    const toolCall = chunks.find((chunk) => chunk.type === "tool-call");

    await model.complete([
      { role: "user", content: "weather?" },
      {
        role: "assistant",
        content: "",
        toolCalls: [
          {
            id: toolCall?.id as string,
            name: toolCall?.name as string,
            input: toolCall?.input,
            providerMetadata: toolCall?.providerMetadata as Record<string, unknown>,
          },
        ],
      },
      { role: "tool", toolCallId: "tu_1", content: "Sunny" },
    ]);

    expect(calls[1]?.params.messages).toEqual([
      { role: "user", content: "weather?" },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "Check the weather.", signature: "sig_stream" },
          { type: "tool_use", id: "tu_1", name: "getWeather", input: {} },
        ],
      },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "tu_1", content: "Sunny" }],
      },
    ]);
  });

  it("rethrows a wrapped typed error when the stream request fails", async () => {
    const { client } = makeFakeClient({ throws: { status: 429, type: "rate_limit_error" } });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await expect(async () => {
      for await (const _event of model.stream([{ role: "user", content: "hi" }])) {
        void _event;
      }
    }).rejects.toMatchObject({ code: "PROVIDER_RATE_LIMIT" });
  });

  it("rethrows a wrapped typed error raised mid-iteration", async () => {
    const { client } = makeFakeClient({
      streamFactory: () =>
        (async function* () {
          yield {
            type: "message_start",
            message: { usage: { input_tokens: 1 } },
          } as unknown as Anthropic.RawMessageStreamEvent;
          yield {
            type: "content_block_delta",
            index: 0,
            delta: { type: "text_delta", text: "partial" },
          } as unknown as Anthropic.RawMessageStreamEvent;

          throw { status: 503, message: "stream dropped" };
        })(),
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const seen: string[] = [];

    await expect(async () => {
      for await (const event of model.stream([{ role: "user", content: "hi" }])) {
        seen.push(event.type);
      }
    }).rejects.toMatchObject({ code: "PROVIDER_ERROR" });

    // The text delta before the failure is still surfaced to the caller.
    expect(seen).toEqual(["delta"]);
  });

  it("surfaces cache_read_input_tokens from message_start as cachedTokens in done", async () => {
    const { client } = makeFakeClient({
      streamEvents: [
        {
          type: "message_start",
          message: { usage: { input_tokens: 8, cache_read_input_tokens: 5 } },
        },
        { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "ok" } },
        { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 2 } },
        { type: "message_stop" },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const chunks = await collectStream(model.stream([{ role: "user", content: "hi" }]));
    const done = chunks.find((chunk) => chunk.type === "done");

    expect(done?.usage).toEqual({ input: 8, output: 2, total: 10, cachedTokens: 5 });
  });

  it("does not set cachedTokens when message_start reports zero cache reads", async () => {
    const { client } = makeFakeClient({
      streamEvents: [
        {
          type: "message_start",
          message: { usage: { input_tokens: 8, cache_read_input_tokens: 0 } },
        },
        { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 2 } },
        { type: "message_stop" },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const chunks = await collectStream(model.stream([{ role: "user", content: "hi" }]));
    const done = chunks.find((chunk) => chunk.type === "done");

    expect(done?.usage).toEqual({ input: 8, output: 2, total: 10 });
    expect((done?.usage as { cachedTokens?: number }).cachedTokens).toBeUndefined();
  });

  it("surfaces cache_creation_input_tokens from message_start as cacheWriteTokens in done", async () => {
    const { client } = makeFakeClient({
      streamEvents: [
        {
          type: "message_start",
          message: { usage: { input_tokens: 8, cache_creation_input_tokens: 12 } },
        },
        { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "ok" } },
        { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 2 } },
        { type: "message_stop" },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const chunks = await collectStream(model.stream([{ role: "user", content: "hi" }]));
    const done = chunks.find((chunk) => chunk.type === "done");

    expect(done?.usage).toEqual({ input: 8, output: 2, total: 10, cacheWriteTokens: 12 });
  });

  it("prefers the cumulative cache counts reported on message_delta", async () => {
    const { client } = makeFakeClient({
      streamEvents: [
        {
          type: "message_start",
          message: {
            usage: {
              input_tokens: 8,
              cache_read_input_tokens: 3,
              cache_creation_input_tokens: 4,
            },
          },
        },
        { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "ok" } },
        {
          type: "message_delta",
          delta: { stop_reason: "end_turn" },
          usage: {
            output_tokens: 2,
            cache_read_input_tokens: 9,
            cache_creation_input_tokens: 7,
          },
        },
        { type: "message_stop" },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const chunks = await collectStream(model.stream([{ role: "user", content: "hi" }]));
    const done = chunks.find((chunk) => chunk.type === "done");

    expect(done?.usage).toEqual({
      input: 8,
      output: 2,
      total: 10,
      cachedTokens: 9,
      cacheWriteTokens: 7,
    });
  });

  it("forwards thinking and a cache breakpoint on a streaming request", async () => {
    const { client, calls } = makeFakeClient({
      streamEvents: [
        { type: "message_start", message: { usage: { input_tokens: 1 } } },
        { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    await collectStream(
      model.stream(
        [
          { role: "system", content: "Stay terse." },
          { role: "user", content: "hi" },
        ],
        { reasoning: { effort: "medium" }, cacheControl: { breakpoints: 1 } },
      ),
    );

    expect((calls[0].params as { thinking?: unknown }).thinking).toEqual({
      type: "enabled",
      budget_tokens: 4096,
    });
    expect(calls[0].params.system).toEqual([
      { type: "text", text: "Stay terse.", cache_control: { type: "ephemeral" } },
    ]);
    expect(calls[0].params.stream).toBe(true);
  });

  it("forwards the AbortSignal to the streaming request", async () => {
    const { client, calls } = makeFakeClient({
      streamEvents: [
        { type: "message_start", message: { usage: { input_tokens: 1 } } },
        { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });
    const controller = new AbortController();

    await collectStream(
      model.stream([{ role: "user", content: "hi" }], { signal: controller.signal }),
    );

    expect(calls[0].params.stream).toBe(true);
    expect(calls[0].requestOptions).toEqual({ signal: controller.signal });
  });

  it("interleaves two tool_use blocks at different indexes into two tool-calls", async () => {
    const { client } = makeFakeClient({
      streamEvents: [
        { type: "message_start", message: { usage: { input_tokens: 3 } } },
        {
          type: "content_block_start",
          index: 0,
          content_block: { type: "tool_use", id: "tu_1", name: "a", input: {} },
        },
        {
          type: "content_block_start",
          index: 1,
          content_block: { type: "tool_use", id: "tu_2", name: "b", input: {} },
        },
        {
          type: "content_block_delta",
          index: 1,
          delta: { type: "input_json_delta", partial_json: '{"y":2}' },
        },
        {
          type: "content_block_delta",
          index: 0,
          delta: { type: "input_json_delta", partial_json: '{"x":1}' },
        },
        { type: "content_block_stop", index: 1 },
        { type: "content_block_stop", index: 0 },
        { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 9 } },
        { type: "message_stop" },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const chunks = await collectStream(model.stream([{ role: "user", content: "hi" }]));
    const toolCalls = chunks.filter((chunk) => chunk.type === "tool-call");

    // Each block emits at its own content_block_stop, in stop order (tu_2 first).
    expect(toolCalls).toEqual([
      { type: "tool-call", id: "tu_2", name: "b", input: { y: 2 } },
      { type: "tool-call", id: "tu_1", name: "a", input: { x: 1 } },
    ]);
  });

  it("falls back to an empty object when a tool block's JSON never parses", async () => {
    const { client } = makeFakeClient({
      streamEvents: [
        { type: "message_start", message: { usage: { input_tokens: 1 } } },
        {
          type: "content_block_start",
          index: 0,
          content_block: { type: "tool_use", id: "tu_1", name: "broken", input: {} },
        },
        {
          type: "content_block_delta",
          index: 0,
          delta: { type: "input_json_delta", partial_json: "{not valid" },
        },
        { type: "content_block_stop", index: 0 },
        { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 2 } },
        { type: "message_stop" },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const chunks = await collectStream(model.stream([{ role: "user", content: "hi" }]));
    const toolCall = chunks.find((chunk) => chunk.type === "tool-call");

    expect(toolCall).toEqual({ type: "tool-call", id: "tu_1", name: "broken", input: {} });
  });

  it("ignores input_json_delta for an index with no open tool block", async () => {
    const { client } = makeFakeClient({
      streamEvents: [
        { type: "message_start", message: { usage: { input_tokens: 1 } } },
        {
          type: "content_block_delta",
          index: 7,
          delta: { type: "input_json_delta", partial_json: '{"orphan":true}' },
        },
        { type: "content_block_stop", index: 7 },
        { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } },
        { type: "message_stop" },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const chunks = await collectStream(model.stream([{ role: "user", content: "hi" }]));

    // No accumulator was ever created → no tool-call is emitted, stream still completes.
    expect(chunks.map((chunk) => chunk.type)).toEqual(["done"]);
  });

  it("retains the prior stop_reason when a later message_delta omits it", async () => {
    const { client } = makeFakeClient({
      streamEvents: [
        { type: "message_start", message: { usage: { input_tokens: 1 } } },
        { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 3 } },
        { type: "message_delta", delta: {}, usage: { output_tokens: 5 } },
        { type: "message_stop" },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const chunks = await collectStream(model.stream([{ role: "user", content: "hi" }]));
    const done = chunks.find((chunk) => chunk.type === "done");

    // stop_reason sticks from the first delta; output_tokens advances to the latest.
    expect(done?.finishReason).toBe("stop");
    expect(done?.usage).toEqual({ input: 1, output: 5, total: 6 });
  });

  it("yields a terminal done with zeroed usage and 'error' reason for an empty stream", async () => {
    const { client } = makeFakeClient({ streamEvents: [] });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const chunks = await collectStream(model.stream([{ role: "user", content: "hi" }]));

    expect(chunks).toEqual([
      { type: "done", finishReason: "error", usage: { input: 0, output: 0, total: 0 } },
    ]);
  });

  it("does not emit a tool-call at a text block's content_block_stop", async () => {
    const { client } = makeFakeClient({
      streamEvents: [
        { type: "message_start", message: { usage: { input_tokens: 2 } } },
        { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
        { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hi" } },
        { type: "content_block_stop", index: 0 },
        { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } },
        { type: "message_stop" },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-4-6" });

    const chunks = await collectStream(model.stream([{ role: "user", content: "hi" }]));

    // A text content_block_stop must NOT emit a tool-call (no accumulator registered).
    expect(chunks).toEqual([
      { type: "delta", content: "Hi" },
      { type: "done", finishReason: "stop", usage: { input: 2, output: 1, total: 3 } },
    ]);
  });
});

/**
 * Per-model thinking + sampling rules (5.26 fixes A and B). Adaptive-mode
 * models (Opus 4.7+, Sonnet 5+, Fable 5+, unknown/future ids) must get
 * `thinking: { type: "adaptive" }` + `output_config.effort` and never
 * `budget_tokens` or `temperature`; budget-mode models (Haiku 4.5, the
 * 4.6 pair, older) keep the legacy `budget_tokens` shape and temperature.
 */
describe("AnthropicModel per-model thinking and temperature rules", () => {
  type WireParams = {
    thinking?: Record<string, unknown>;
    output_config?: Record<string, unknown>;
    temperature?: number;
    max_tokens?: number;
    stream?: boolean;
  };

  const ADAPTIVE_MODELS = [
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-opus-5-5",
    "claude-sonnet-5",
    "claude-sonnet-5-5",
    "claude-fable-5",
    "claude-fable-5-1",
    "claude-some-future-model",
  ];

  it("sends adaptive thinking plus output_config.effort for an adaptive model", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-opus-5-5" });

    await model.complete([{ role: "user", content: "hi" }], { reasoning: { effort: "high" } });

    const params = calls[0].params as WireParams;

    expect(params.thinking).toEqual({ type: "adaptive" });
    expect(params.output_config).toEqual({ effort: "high" });
    // Same answer headroom as the budget path today: high budget (12000) + 4096.
    expect(params.max_tokens).toBe(16096);
  });

  it("defaults the effort to medium when an adaptive model gets only reasoning.maxTokens", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-sonnet-5-5" });

    await model.complete([{ role: "user", content: "hi" }], { reasoning: { maxTokens: 8000 } });

    const params = calls[0].params as WireParams;

    expect(params.thinking).toEqual({ type: "adaptive" });
    expect(params.output_config).toEqual({ effort: "medium" });
    expect(params.max_tokens).toBe(12096);
  });

  it("merges the effort into output_config alongside a structured-output format", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-opus-4-8" });
    const schema = { type: "object", properties: { summary: { type: "string" } } };

    await model.complete([{ role: "user", content: "hi" }], {
      reasoning: { effort: "low" },
      responseSchema: schema,
    });

    expect((calls[0].params as WireParams).output_config).toEqual({
      format: { type: "json_schema", schema },
      effort: "low",
    });
  });

  it("never sends budget_tokens to an adaptive model", async () => {
    for (const name of ADAPTIVE_MODELS) {
      const { client, calls } = makeFakeClient({ message: message() });
      const model = new AnthropicModel(client, { name });

      await model.complete([{ role: "user", content: "hi" }], { reasoning: { maxTokens: 9000 } });
      await model.complete([{ role: "user", content: "hi" }], { reasoning: { effort: "high" } });
      await model.complete([{ role: "user", content: "hi" }], {
        reasoning: { effort: "low", maxTokens: 2000 },
      });

      for (const call of calls) {
        expect(JSON.stringify(call.params), name).not.toContain("budget_tokens");
        expect((call.params as WireParams).thinking, name).toEqual({ type: "adaptive" });
      }
    }
  });

  it("does not reject a maxTokens below reasoning.maxTokens on an adaptive model", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-opus-5-5" });

    await model.complete([{ role: "user", content: "hi" }], {
      maxTokens: 2000,
      reasoning: { maxTokens: 8000 },
    });

    expect(calls[0].params.max_tokens).toBe(2000);
  });

  it("keeps budget_tokens and no output_config for a budget-mode model", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-haiku-4-5" });

    await model.complete([{ role: "user", content: "hi" }], { reasoning: { effort: "high" } });

    const params = calls[0].params as WireParams;

    expect(params.thinking).toEqual({ type: "enabled", budget_tokens: 12000 });
    expect("output_config" in calls[0].params).toBe(false);
  });

  it("omits thinking and output_config on an adaptive model for effort 'none'", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-opus-4-7" });

    await model.complete([{ role: "user", content: "hi" }], { reasoning: { effort: "none" } });

    expect("thinking" in calls[0].params).toBe(false);
    expect("output_config" in calls[0].params).toBe(false);
  });

  it("omits temperature for a post-4.6 model even without thinking", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const model = new AnthropicModel(client, { name: "claude-opus-4-7", temperature: 0.4 });

    await model.complete([{ role: "user", content: "hi" }]);
    await model.complete([{ role: "user", content: "hi" }], { temperature: 0.9 });

    expect("temperature" in calls[0].params).toBe(false);
    expect("temperature" in calls[1].params).toBe(false);
  });

  it("keeps temperature for Haiku 4.5 and the 4.6 pair", async () => {
    const { client, calls } = makeFakeClient({ message: message() });

    for (const name of ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6"]) {
      await new AnthropicModel(client, { name, temperature: 0.3 }).complete([
        { role: "user", content: "hi" },
      ]);
    }

    expect(calls.map((call) => call.params.temperature)).toEqual([0.3, 0.3, 0.3]);
  });

  it("honors an explicit thinkingMode override in either direction", async () => {
    const { client, calls } = makeFakeClient({ message: message() });
    const proxiedLegacy = new AnthropicModel(client, {
      name: "my-gateway-haiku",
      thinkingMode: "budget",
      temperature: 0.2,
    });
    const forcedAdaptive = new AnthropicModel(client, {
      name: "claude-sonnet-4-6",
      thinkingMode: "adaptive",
    });

    await proxiedLegacy.complete([{ role: "user", content: "hi" }], {
      reasoning: { effort: "low" },
    });
    await proxiedLegacy.complete([{ role: "user", content: "hi" }]);
    await forcedAdaptive.complete([{ role: "user", content: "hi" }], {
      reasoning: { effort: "high" },
    });

    expect((calls[0].params as WireParams).thinking).toEqual({
      type: "enabled",
      budget_tokens: 1024,
    });
    expect(calls[1].params.temperature).toBe(0.2);
    expect((calls[2].params as WireParams).thinking).toEqual({ type: "adaptive" });
    expect((calls[2].params as WireParams).output_config).toEqual({ effort: "high" });
  });

  it("replays signed thinking blocks on an adaptive model's tool-use follow-up", async () => {
    const { client, calls } = makeFakeClient({
      message: message({
        model: "claude-opus-5-5",
        stop_reason: "tool_use",
        content: [
          { type: "thinking", thinking: "", signature: "sig_adaptive" },
          { type: "tool_use", id: "tu_1", name: "getWeather", input: { city: "Cairo" } },
        ] as unknown as Anthropic.ContentBlock[],
      }),
    });
    const model = new AnthropicModel(client, { name: "claude-opus-5-5" });

    const response = await model.complete([{ role: "user", content: "weather?" }], {
      reasoning: { effort: "medium" },
    });
    await model.complete(
      [
        { role: "user", content: "weather?" },
        { role: "assistant", content: response.content, toolCalls: response.toolCalls },
        { role: "tool", toolCallId: "tu_1", content: "Sunny" },
      ],
      { reasoning: { effort: "medium" } },
    );

    expect((calls[1].params as WireParams).thinking).toEqual({ type: "adaptive" });
    expect(calls[1]?.params.messages?.[1]).toEqual({
      role: "assistant",
      content: [
        { type: "thinking", thinking: "", signature: "sig_adaptive" },
        { type: "tool_use", id: "tu_1", name: "getWeather", input: { city: "Cairo" } },
      ],
    });
  });

  it("applies the adaptive shape and drops temperature on a streaming request", async () => {
    const { client, calls } = makeFakeClient({
      streamEvents: [
        { type: "message_start", message: { usage: { input_tokens: 1 } } },
        { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-sonnet-5-5", temperature: 0.5 });

    await collectStream(
      model.stream([{ role: "user", content: "hi" }], { reasoning: { effort: "low" } }),
    );
    await collectStream(model.stream([{ role: "user", content: "hi" }]));

    const first = calls[0].params as WireParams;

    expect(first.stream).toBe(true);
    expect(first.thinking).toEqual({ type: "adaptive" });
    expect(first.output_config).toEqual({ effort: "low" });
    expect(JSON.stringify(first)).not.toContain("budget_tokens");
    expect("temperature" in calls[0].params).toBe(false);
    expect("temperature" in calls[1].params).toBe(false);
  });

  it("keeps budget_tokens and temperature on a budget-mode streaming request", async () => {
    const { client, calls } = makeFakeClient({
      streamEvents: [
        { type: "message_start", message: { usage: { input_tokens: 1 } } },
        { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } },
      ] as unknown as Anthropic.RawMessageStreamEvent[],
    });
    const model = new AnthropicModel(client, { name: "claude-haiku-4-5", temperature: 0.5 });

    await collectStream(
      model.stream([{ role: "user", content: "hi" }], { reasoning: { effort: "medium" } }),
    );
    await collectStream(model.stream([{ role: "user", content: "hi" }]));

    expect((calls[0].params as WireParams).thinking).toEqual({
      type: "enabled",
      budget_tokens: 4096,
    });
    expect("output_config" in calls[0].params).toBe(false);
    expect(calls[1].params.temperature).toBe(0.5);
  });

  it("infers vision for the Claude 5-series models", () => {
    const { client } = makeFakeClient({ message: message() });

    for (const name of ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"]) {
      expect(new AnthropicModel(client, { name }).capabilities.vision, name).toBe(true);
    }
  });
});
