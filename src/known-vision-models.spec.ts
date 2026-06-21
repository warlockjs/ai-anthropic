import { describe, expect, it } from "vitest";
import { inferVisionCapability } from "./known-vision-models";

describe("inferVisionCapability", () => {
  it("returns true for current claude-<tier>-4 families", () => {
    expect(inferVisionCapability("claude-opus-4-7")).toBe(true);
    expect(inferVisionCapability("claude-sonnet-4-6")).toBe(true);
    expect(inferVisionCapability("claude-haiku-4-5")).toBe(true);
    expect(inferVisionCapability("claude-opus-4-20250514")).toBe(true);
  });

  it("returns true for claude-3 / 3.5 / 3.7 dotted naming", () => {
    expect(inferVisionCapability("claude-3-haiku-20240307")).toBe(true);
    expect(inferVisionCapability("claude-3-5-sonnet-latest")).toBe(true);
  });

  it("is case-insensitive", () => {
    expect(inferVisionCapability("CLAUDE-SONNET-4-6")).toBe(true);
  });

  it("returns true for the bare tier-4 family prefixes", () => {
    expect(inferVisionCapability("claude-4-something")).toBe(true);
    expect(inferVisionCapability("claude-opus-4")).toBe(true);
    expect(inferVisionCapability("claude-sonnet-4")).toBe(true);
    expect(inferVisionCapability("claude-haiku-4")).toBe(true);
  });

  it("returns false for pre-3 and unknown models", () => {
    expect(inferVisionCapability("claude-2.1")).toBe(false);
    expect(inferVisionCapability("claude-instant-1.2")).toBe(false);
    expect(inferVisionCapability("custom-proxy-llm")).toBe(false);
    expect(inferVisionCapability("")).toBe(false);
  });

  it("matches as a prefix, not a substring (no mid-string match)", () => {
    // The vision prefix must be at the START — embedding it later does not match.
    expect(inferVisionCapability("my-gateway/claude-sonnet-4-6")).toBe(false);
    expect(inferVisionCapability("anthropic.claude-3-haiku")).toBe(false);
  });

  it("returns false for a non-claude vendor model", () => {
    expect(inferVisionCapability("gpt-4o")).toBe(false);
    expect(inferVisionCapability("gemini-1.5-pro")).toBe(false);
  });
});
