import { describe, expect, it } from "vitest";
import { AnthropicSDK } from "./sdk";

describe("AnthropicSDK", () => {
  it("constructs successfully with an apiKey", () => {
    expect(new AnthropicSDK({ apiKey: "test-key" })).toBeInstanceOf(AnthropicSDK);
  });

  it("constructs successfully with apiKey + baseURL (proxied endpoints)", () => {
    const sdk = new AnthropicSDK({
      apiKey: "test-key",
      baseURL: "https://gateway.internal/anthropic",
    });

    expect(sdk).toBeInstanceOf(AnthropicSDK);
  });

  it("forwards upstream ClientOptions (timeout, maxRetries) to the Anthropic client — C1 regression", () => {
    // Previously only apiKey + baseURL were forwarded, so every other
    // type-checked ClientOptions value was silently dropped.
    const sdk = new AnthropicSDK({ apiKey: "test-key", maxRetries: 7, timeout: 1234 });
    const client = (sdk as unknown as { client: { maxRetries: number; timeout: number } }).client;

    expect(client.maxRetries).toBe(7);
    expect(client.timeout).toBe(1234);
  });

  it("model() returns a fresh ModelContract bound to this SDK each call", () => {
    const sdk = new AnthropicSDK({ apiKey: "test-key" });
    const a = sdk.model({ name: "claude-haiku-4-5" });
    const b = sdk.model({ name: "claude-haiku-4-5" });

    expect(a).not.toBe(b);
    expect(a.name).toBe("claude-haiku-4-5");
    expect(a.provider).toBe("anthropic");
  });

  it("model() honors a custom provider label", () => {
    const sdk = new AnthropicSDK({ apiKey: "test-key", provider: "bedrock-anthropic" });

    expect(sdk.model({ name: "claude-sonnet-4-6" }).provider).toBe("bedrock-anthropic");
  });

  it("model() forwards vision capability inference from the model name", () => {
    const sdk = new AnthropicSDK({ apiKey: "test-key" });

    expect(sdk.model({ name: "claude-sonnet-4-6" }).capabilities?.vision).toBe(true);
    expect(sdk.model({ name: "claude-2.1" }).capabilities?.vision).toBe(false);
  });

  it("model() honors explicit vision override over inference", () => {
    const sdk = new AnthropicSDK({ apiKey: "test-key" });

    expect(sdk.model({ name: "claude-2.1", vision: true }).capabilities?.vision).toBe(true);
    expect(sdk.model({ name: "claude-sonnet-4-6", vision: false }).capabilities?.vision).toBe(
      false,
    );
  });

  it("model() defaults structuredOutput to true and honors the override", () => {
    const sdk = new AnthropicSDK({ apiKey: "test-key" });

    expect(sdk.model({ name: "claude-sonnet-4-6" }).capabilities?.structuredOutput).toBe(true);
    expect(
      sdk.model({ name: "claude-sonnet-4-6", structuredOutput: false }).capabilities
        ?.structuredOutput,
    ).toBe(false);
  });

  it("model() resolves SDK-level pricing by model name, per-model wins", () => {
    const sdk = new AnthropicSDK({
      apiKey: "test-key",
      pricing: { "claude-sonnet-4-6": { input: 3, output: 15 } },
    });

    expect(sdk.model({ name: "claude-sonnet-4-6" }).pricing).toEqual({ input: 3, output: 15 });
    expect(
      sdk.model({ name: "claude-sonnet-4-6", pricing: { input: 1, output: 2 } }).pricing,
    ).toEqual({ input: 1, output: 2 });
    expect(sdk.model({ name: "claude-haiku-4-5" }).pricing).toBeUndefined();
  });

  it("count() returns an approximate token count using the core heuristic", async () => {
    const sdk = new AnthropicSDK({ apiKey: "test-key" });

    expect(await sdk.count("")).toBe(0);
    expect(await sdk.count("Hello, world!")).toBe(4);
    expect(await sdk.count("a".repeat(400))).toBe(100);
  });

  it("count() ignores the optional model argument (offline heuristic only)", async () => {
    const sdk = new AnthropicSDK({ apiKey: "test-key" });

    expect(await sdk.count("Hello, world!", "claude-sonnet-4-6")).toBe(4);
    expect(await sdk.count("Hello, world!")).toBe(
      await sdk.count("Hello, world!", "any-other-model"),
    );
  });

  it("count() rounds partial tokens up (ceil of length / 4)", async () => {
    const sdk = new AnthropicSDK({ apiKey: "test-key" });

    // 5 chars / 4 = 1.25 → ceils to 2.
    expect(await sdk.count("abcde")).toBe(2);
  });

  it("model() does not re-wrap config when no SDK pricing applies", () => {
    const sdk = new AnthropicSDK({ apiKey: "test-key" });

    // Without any pricing, pricing resolves to undefined on the produced model.
    expect(sdk.model({ name: "claude-sonnet-4-6" }).pricing).toBeUndefined();
  });

  it("model() defaults the provider label to 'anthropic'", () => {
    const sdk = new AnthropicSDK({ apiKey: "test-key" });

    expect(sdk.model({ name: "claude-sonnet-4-6" }).provider).toBe("anthropic");
  });

  it("does not expose an embedder() method (Anthropic has no embeddings API)", () => {
    const sdk = new AnthropicSDK({ apiKey: "test-key" });

    expect((sdk as { embedder?: unknown }).embedder).toBeUndefined();
  });
});
