import {
  AIError,
  ContextLengthExceededError,
  InvalidRequestError,
  ProviderAuthError,
  ProviderError,
  ProviderRateLimitError,
  ProviderTimeoutError,
  QuotaExceededError,
} from "@warlock.js/ai";
import { APIConnectionTimeoutError, APIError } from "@anthropic-ai/sdk";
import { describe, expect, it } from "vitest";
import { wrapAnthropicError } from "./wrap-anthropic-error";

/**
 * Build a duck-typed object that looks like an Anthropic `APIError`
 * without going through the real constructor (which wants a Headers
 * instance and a private response body we don't control in tests).
 */
function fakeAnthropicError(shape: {
  status?: number;
  type?: string;
  message?: string;
  headers?: Record<string, string>;
  name?: string;
  requestID?: string;
}): unknown {
  return {
    name: shape.name ?? "APIError",
    status: shape.status,
    type: shape.type,
    message: shape.message ?? "failed",
    headers: shape.headers,
    requestID: shape.requestID,
  };
}

describe("wrapAnthropicError", () => {
  it("passes AIError through untouched", () => {
    const original = new ProviderRateLimitError("slow", { retryAfter: 1000 });

    expect(wrapAnthropicError(original)).toBe(original);
  });

  it("maps authentication_error type to ProviderAuthError", () => {
    const wrapped = wrapAnthropicError(
      fakeAnthropicError({ status: 401, type: "authentication_error", message: "bad key" }),
    );

    expect(wrapped).toBeInstanceOf(ProviderAuthError);
    expect(wrapped.message).toBe("bad key");
  });

  it("maps permission_error type to ProviderAuthError", () => {
    const wrapped = wrapAnthropicError(
      fakeAnthropicError({ status: 403, type: "permission_error" }),
    );

    expect(wrapped).toBeInstanceOf(ProviderAuthError);
  });

  it("maps bare 401 / 403 (body stripped) to ProviderAuthError", () => {
    expect(wrapAnthropicError(fakeAnthropicError({ status: 401 }))).toBeInstanceOf(
      ProviderAuthError,
    );
    expect(wrapAnthropicError(fakeAnthropicError({ status: 403 }))).toBeInstanceOf(
      ProviderAuthError,
    );
  });

  it("maps billing_error to QuotaExceededError (still an AIError/ProviderError-free quota)", () => {
    const wrapped = wrapAnthropicError(
      fakeAnthropicError({ status: 400, type: "billing_error" }),
    );

    expect(wrapped).toBeInstanceOf(QuotaExceededError);
  });

  it("maps rate_limit_error to ProviderRateLimitError and parses retry-after", () => {
    const wrapped = wrapAnthropicError(
      fakeAnthropicError({
        status: 429,
        type: "rate_limit_error",
        headers: { "retry-after": "2" },
      }),
    );

    expect(wrapped).toBeInstanceOf(ProviderRateLimitError);
    expect((wrapped as ProviderRateLimitError).retryAfter).toBe(2000);
  });

  it("maps bare 429 to ProviderRateLimitError", () => {
    expect(wrapAnthropicError(fakeAnthropicError({ status: 429 }))).toBeInstanceOf(
      ProviderRateLimitError,
    );
  });

  it("reads retry-after from a real Headers instance", () => {
    const wrapped = wrapAnthropicError({
      name: "APIError",
      status: 429,
      type: "rate_limit_error",
      message: "slow down",
      headers: new Headers({ "retry-after": "5" }),
    });

    expect((wrapped as ProviderRateLimitError).retryAfter).toBe(5000);
  });

  it("leaves retryAfter undefined when header missing or unparseable", () => {
    expect(
      (wrapAnthropicError(fakeAnthropicError({ status: 429 })) as ProviderRateLimitError)
        .retryAfter,
    ).toBeUndefined();
    expect(
      (
        wrapAnthropicError(
          fakeAnthropicError({ status: 429, headers: { "retry-after": "soon" } }),
        ) as ProviderRateLimitError
      ).retryAfter,
    ).toBeUndefined();
  });

  it("maps invalid_request_error to InvalidRequestError", () => {
    const wrapped = wrapAnthropicError(
      fakeAnthropicError({ status: 400, type: "invalid_request_error", message: "bad input" }),
    );

    expect(wrapped).toBeInstanceOf(InvalidRequestError);
  });

  it("maps a 'prompt is too long' 400 to ContextLengthExceededError", () => {
    const wrapped = wrapAnthropicError(
      fakeAnthropicError({
        status: 400,
        type: "invalid_request_error",
        message: "prompt is too long: 250000 tokens > 200000 maximum",
      }),
    );

    expect(wrapped).toBeInstanceOf(ContextLengthExceededError);
    expect(wrapped.code).toBe("CONTEXT_LENGTH_EXCEEDED");
  });

  it("maps a generic 4xx (no type) to InvalidRequestError", () => {
    expect(wrapAnthropicError(fakeAnthropicError({ status: 404 }))).toBeInstanceOf(
      InvalidRequestError,
    );
  });

  it("maps 500 / overloaded to plain ProviderError", () => {
    const wrapped = wrapAnthropicError(
      fakeAnthropicError({ status: 500, message: "server err" }),
    );

    expect(wrapped).toBeInstanceOf(ProviderError);
    expect(wrapped).not.toBeInstanceOf(InvalidRequestError);
    expect(wrapped.code).toBe("PROVIDER_ERROR");

    expect(
      wrapAnthropicError(fakeAnthropicError({ status: 529, type: "overloaded_error" })),
    ).toBeInstanceOf(ProviderError);
  });

  it("maps APIConnectionTimeoutError-by-name to ProviderTimeoutError", () => {
    const wrapped = wrapAnthropicError(
      fakeAnthropicError({ name: "APIConnectionTimeoutError", message: "timeout" }),
    );

    expect(wrapped).toBeInstanceOf(ProviderTimeoutError);
  });

  it("maps ETIMEDOUT / ECONNABORTED to ProviderTimeoutError", () => {
    expect(wrapAnthropicError({ code: "ETIMEDOUT", message: "socket" })).toBeInstanceOf(
      ProviderTimeoutError,
    );
    expect(wrapAnthropicError({ code: "ECONNABORTED", message: "aborted" })).toBeInstanceOf(
      ProviderTimeoutError,
    );
  });

  it("categorizes a real APIConnectionTimeoutError instance as timeout", () => {
    const wrapped = wrapAnthropicError(new APIConnectionTimeoutError({ message: "timed out" }));

    expect(wrapped).toBeInstanceOf(ProviderTimeoutError);
  });

  it("preserves cause and attaches status/type/requestId to context", () => {
    const raw = fakeAnthropicError({
      status: 429,
      type: "rate_limit_error",
      requestID: "req_abc",
    });

    const wrapped = wrapAnthropicError(raw);

    expect((wrapped as unknown as { cause: unknown }).cause).toBe(raw);
    expect(wrapped.context).toMatchObject({
      status: 429,
      type: "rate_limit_error",
      requestId: "req_abc",
    });
  });

  it("wraps non-Error, numeric, and plain Error values into ProviderError", () => {
    expect(wrapAnthropicError("random string").message).toBe("random string");
    expect(wrapAnthropicError(42).message).toBe("42");
    expect(wrapAnthropicError(new Error("boom"))).toBeInstanceOf(ProviderError);
  });

  it("maps permission_error to ProviderAuthError on type alone (body stripped of status)", () => {
    const wrapped = wrapAnthropicError(fakeAnthropicError({ type: "permission_error" }));

    expect(wrapped).toBeInstanceOf(ProviderAuthError);
  });

  it("maps authentication_error to ProviderAuthError even with a non-401 status", () => {
    const wrapped = wrapAnthropicError(
      fakeAnthropicError({ status: 400, type: "authentication_error" }),
    );

    expect(wrapped).toBeInstanceOf(ProviderAuthError);
  });

  it("prefers the rate-limit type over a 4xx status when both are present", () => {
    const wrapped = wrapAnthropicError(
      fakeAnthropicError({ status: 400, type: "rate_limit_error" }),
    );

    expect(wrapped).toBeInstanceOf(ProviderRateLimitError);
  });

  it("classifies a real Anthropic APIError instance by its body type (instanceof path)", () => {
    const real = APIError.generate(
      429,
      { type: "error", error: { type: "rate_limit_error", message: "slow down" } },
      "slow down",
      new Headers({ "retry-after": "4", "request-id": "req_real" }),
    );

    const wrapped = wrapAnthropicError(real);

    expect(wrapped).toBeInstanceOf(ProviderRateLimitError);
    expect((wrapped as ProviderRateLimitError).retryAfter).toBe(4000);
    expect(wrapped.context).toMatchObject({
      status: 429,
      type: "rate_limit_error",
      requestId: "req_real",
    });
  });

  it("reads the request id from a snake_case request_id on a plain object", () => {
    const wrapped = wrapAnthropicError({
      status: 500,
      message: "boom",
      request_id: "req_snake",
    });

    expect(wrapped.context).toMatchObject({ requestId: "req_snake" });
  });

  it("reads retry-after from a capitalized Retry-After record key", () => {
    const wrapped = wrapAnthropicError(
      fakeAnthropicError({ status: 429, headers: { "Retry-After": "3" } }),
    );

    expect((wrapped as ProviderRateLimitError).retryAfter).toBe(3000);
  });

  it("rounds a fractional retry-after to the nearest millisecond", () => {
    const wrapped = wrapAnthropicError(
      fakeAnthropicError({ status: 429, headers: { "retry-after": "1.5" } }),
    );

    expect((wrapped as ProviderRateLimitError).retryAfter).toBe(1500);
  });

  it("ignores a negative retry-after value", () => {
    const wrapped = wrapAnthropicError(
      fakeAnthropicError({ status: 429, headers: { "retry-after": "-5" } }),
    );

    expect((wrapped as ProviderRateLimitError).retryAfter).toBeUndefined();
  });

  it("leaves context empty when status, type, and request id are all absent", () => {
    const wrapped = wrapAnthropicError(new Error("plain"));

    expect(wrapped.context).toEqual({});
  });

  it("includes only the populated diagnostic fields in context", () => {
    const wrapped = wrapAnthropicError(fakeAnthropicError({ status: 500 }));

    expect(wrapped.context).toEqual({ status: 500 });
  });

  it("falls back to thrown.message for an Error with no Anthropic-shaped message field", () => {
    const wrapped = wrapAnthropicError(new Error("native boom"));

    expect(wrapped.message).toBe("native boom");
  });

  it("uses the default 'failed' message duck-typed errors carry when no message exists", () => {
    // A plain object with no `message` key → falls through to String(thrown).
    const wrapped = wrapAnthropicError({ status: 500 });

    expect(wrapped.message).toBe("[object Object]");
  });

  it("does not attach a retryAfter for a bare 429 with no headers at all", () => {
    const wrapped = wrapAnthropicError({ status: 429 });

    expect(wrapped).toBeInstanceOf(ProviderRateLimitError);
    expect((wrapped as ProviderRateLimitError).retryAfter).toBeUndefined();
  });

  it("every wrapped error is an AIError", () => {
    const samples = [
      fakeAnthropicError({ status: 401, type: "authentication_error" }),
      fakeAnthropicError({ status: 429, type: "rate_limit_error" }),
      fakeAnthropicError({ status: 400, type: "billing_error" }),
      fakeAnthropicError({ status: 400, type: "invalid_request_error" }),
      fakeAnthropicError({ status: 500 }),
      fakeAnthropicError({ name: "APIConnectionTimeoutError" }),
      "plain string",
      new Error("plain error"),
    ];

    for (const sample of samples) {
      expect(wrapAnthropicError(sample)).toBeInstanceOf(AIError);
    }
  });
});
