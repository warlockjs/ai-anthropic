import type { Message } from "@warlock.js/ai";
import { describe, expect, it } from "vitest";
import { toAnthropicMessages } from "./to-anthropic-messages";

describe("toAnthropicMessages", () => {
  it("hoists system messages into the top-level system string", () => {
    const messages: Message[] = [
      { role: "system", content: "Be concise." },
      { role: "user", content: "Hi" },
    ];

    expect(toAnthropicMessages(messages)).toEqual({
      system: "Be concise.",
      messages: [{ role: "user", content: "Hi" }],
    });
  });

  it("joins multiple system messages with a blank line", () => {
    const messages: Message[] = [
      { role: "system", content: "First." },
      { role: "system", content: "Second." },
      { role: "user", content: "Hi" },
    ];

    expect(toAnthropicMessages(messages).system).toBe("First.\n\nSecond.");
  });

  it("leaves system undefined when there is no system message", () => {
    const result = toAnthropicMessages([{ role: "user", content: "Hi" }]);

    expect(result.system).toBeUndefined();
  });

  it("converts tool messages into a user turn with a tool_result block", () => {
    const messages: Message[] = [{ role: "tool", toolCallId: "tu_1", content: '{"ok":true}' }];

    expect(toAnthropicMessages(messages).messages).toEqual([
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "tu_1", content: '{"ok":true}' }],
      },
    ]);
  });

  it("falls back to empty tool_use_id when missing", () => {
    const messages: Message[] = [{ role: "tool", content: "{}" }];

    expect(toAnthropicMessages(messages).messages).toEqual([
      { role: "user", content: [{ type: "tool_result", tool_use_id: "", content: "{}" }] },
    ]);
  });

  it("emits assistant tool calls as text + tool_use blocks", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        content: "Let me check.",
        toolCalls: [{ id: "tu_1", name: "getWeather", input: { city: "Cairo" } }],
      },
    ];

    expect(toAnthropicMessages(messages).messages).toEqual([
      {
        role: "assistant",
        content: [
          { type: "text", text: "Let me check." },
          { type: "tool_use", id: "tu_1", name: "getWeather", input: { city: "Cairo" } },
        ],
      },
    ]);
  });

  it("omits the leading text block when assistant content is empty", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        content: "",
        toolCalls: [{ id: "tu_1", name: "noop", input: {} }],
      },
    ];

    expect(toAnthropicMessages(messages).messages).toEqual([
      { role: "assistant", content: [{ type: "tool_use", id: "tu_1", name: "noop", input: {} }] },
    ]);
  });

  it("maps a multipart user message into Anthropic image blocks", () => {
    const messages: Message[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          { type: "image", source: { url: "https://example.com/cat.jpg" } },
        ],
      },
    ];

    expect(toAnthropicMessages(messages).messages).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          { type: "image", source: { type: "url", url: "https://example.com/cat.jpg" } },
        ],
      },
    ]);
  });

  it("maps a pdf part into an Anthropic document block (A2)", () => {
    const messages: Message[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "Summarize" },
          { type: "pdf", source: { base64: "JVBERi0=", mediaType: "application/pdf" } },
        ],
      },
    ];

    expect(toAnthropicMessages(messages).messages).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "Summarize" },
          {
            type: "document",
            source: { type: "base64", media_type: "application/pdf", data: "JVBERi0=" },
          },
        ],
      },
    ]);
  });

  it("maps a pdf URL into a document url source (A2)", () => {
    const messages: Message[] = [
      { role: "user", content: [{ type: "pdf", source: { url: "https://x/r.pdf" } }] },
    ];

    expect(toAnthropicMessages(messages).messages[0].content).toEqual([
      { type: "document", source: { type: "url", url: "https://x/r.pdf" } },
    ]);
  });

  it("renders inline base64 image sources as base64 blocks", () => {
    const messages: Message[] = [
      {
        role: "user",
        content: [{ type: "image", source: { base64: "iVBORw0KGgo=", mediaType: "image/png" } }],
      },
    ];

    expect(toAnthropicMessages(messages).messages).toEqual([
      {
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" } },
        ],
      },
    ]);
  });

  it("collapses multipart content to text on non-user roles", () => {
    const messages: Message[] = [
      {
        role: "system",
        content: [
          { type: "text", text: "Hello " },
          { type: "text", text: "world" },
        ],
      },
    ];

    expect(toAnthropicMessages(messages).system).toBe("Hello world");
  });

  it("passes a plain assistant string message straight through as an assistant turn", () => {
    const messages: Message[] = [{ role: "assistant", content: "It's 82°F in Cairo." }];

    expect(toAnthropicMessages(messages).messages).toEqual([
      { role: "assistant", content: "It's 82°F in Cairo." },
    ]);
  });

  it("treats an assistant message with an empty toolCalls array as a plain assistant turn", () => {
    const messages: Message[] = [{ role: "assistant", content: "done", toolCalls: [] }];

    expect(toAnthropicMessages(messages).messages).toEqual([
      { role: "assistant", content: "done" },
    ]);
  });

  it("emits multiple tool_use blocks on a single assistant message", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        content: "",
        toolCalls: [
          { id: "tu_1", name: "getWeather", input: { city: "Cairo" } },
          { id: "tu_2", name: "getTime", input: { tz: "UTC" } },
        ],
      },
    ];

    expect(toAnthropicMessages(messages).messages).toEqual([
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "tu_1", name: "getWeather", input: { city: "Cairo" } },
          { type: "tool_use", id: "tu_2", name: "getTime", input: { tz: "UTC" } },
        ],
      },
    ]);
  });

  it("defaults a tool_use block input to an empty object when undefined", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        content: "",
        toolCalls: [{ id: "tu_1", name: "noop" } as never],
      },
    ];

    expect(toAnthropicMessages(messages).messages).toEqual([
      { role: "assistant", content: [{ type: "tool_use", id: "tu_1", name: "noop", input: {} }] },
    ]);
  });

  it("collapses multipart content to text on the tool role", () => {
    const messages: Message[] = [
      {
        role: "tool",
        toolCallId: "tu_1",
        content: [
          { type: "text", text: "part-a " },
          { type: "text", text: "part-b" },
        ],
      },
    ];

    expect(toAnthropicMessages(messages).messages).toEqual([
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "tu_1", content: "part-a part-b" }],
      },
    ]);
  });

  it("drops non-text parts when stringifying multipart content for a non-user role", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "see image " },
          { type: "image", source: { url: "https://example.com/x.png" } },
          { type: "text", text: "now" },
        ],
      },
    ];

    expect(toAnthropicMessages(messages).messages).toEqual([
      { role: "assistant", content: "see image now" },
    ]);
  });

  it("preserves a mixed text + image + base64 multipart user message in order", () => {
    const messages: Message[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "compare" },
          { type: "image", source: { url: "https://example.com/a.jpg" } },
          { type: "image", source: { base64: "AAAA", mediaType: "image/webp" } },
        ],
      },
    ];

    expect(toAnthropicMessages(messages).messages).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "compare" },
          { type: "image", source: { type: "url", url: "https://example.com/a.jpg" } },
          { type: "image", source: { type: "base64", media_type: "image/webp", data: "AAAA" } },
        ],
      },
    ]);
  });

  it("preserves order across a full system / user / assistant-tool-call / tool / assistant round-trip", () => {
    const messages: Message[] = [
      { role: "system", content: "Be terse." },
      { role: "user", content: "Weather in Cairo?" },
      {
        role: "assistant",
        content: "checking",
        toolCalls: [{ id: "tu_1", name: "getWeather", input: { city: "Cairo" } }],
      },
      { role: "tool", toolCallId: "tu_1", content: '{"temp":82}' },
      { role: "assistant", content: "It's 82°F." },
    ];

    expect(toAnthropicMessages(messages)).toEqual({
      system: "Be terse.",
      messages: [
        { role: "user", content: "Weather in Cairo?" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "checking" },
            { type: "tool_use", id: "tu_1", name: "getWeather", input: { city: "Cairo" } },
          ],
        },
        {
          role: "user",
          content: [{ type: "tool_result", tool_use_id: "tu_1", content: '{"temp":82}' }],
        },
        { role: "assistant", content: "It's 82°F." },
      ],
    });
  });

  it("returns empty messages and undefined system for an empty input", () => {
    expect(toAnthropicMessages([])).toEqual({ system: undefined, messages: [] });
  });
});
