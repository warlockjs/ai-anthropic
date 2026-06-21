import { describe, expect, it } from "vitest";
import { mapStopReason } from "./map-stop-reason";

describe("mapStopReason", () => {
  it("maps natural stops to 'stop'", () => {
    expect(mapStopReason("end_turn")).toBe("stop");
    expect(mapStopReason("stop_sequence")).toBe("stop");
  });

  it("maps max_tokens to 'length'", () => {
    expect(mapStopReason("max_tokens")).toBe("length");
  });

  it("maps tool_use to 'tool_calls'", () => {
    expect(mapStopReason("tool_use")).toBe("tool_calls");
  });

  it("treats refusal / pause_turn as 'error' (non-clean terminals)", () => {
    expect(mapStopReason("refusal")).toBe("error");
    expect(mapStopReason("pause_turn")).toBe("error");
  });

  it("falls back to 'error' for null, undefined, empty, and unknown values", () => {
    expect(mapStopReason(null)).toBe("error");
    expect(mapStopReason(undefined)).toBe("error");
    expect(mapStopReason("")).toBe("error");
    expect(mapStopReason("something-new")).toBe("error");
  });
});
