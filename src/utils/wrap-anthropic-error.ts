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

/**
 * Raw-error fields the wrapper reads off an Anthropic SDK error.
 *
 * `APIError` exposes `status`, `type` (the `error.type` from the
 * response body, e.g. `"rate_limit_error"`), `message`, `headers`, and
 * `requestID`. We duck-type because wrapped retries, proxied errors,
 * and custom subclasses sometimes lose the `instanceof` relationship.
 */
type AnthropicErrorShape = {
  status?: number;
  type?: string | null;
  message?: string;
  headers?: HeaderBag | undefined;
  name?: string;
  requestId?: string;
};

/** Either a fetch `Headers` instance or a plain record (duck-typed tests). */
type HeaderBag = Headers | Record<string, string>;

/**
 * Wrap any thrown value caught inside the Anthropic adapter into the
 * appropriate `@warlock.js/ai` `AIError` subclass.
 *
 * **Dispatch strategy.** Anthropic has no per-error machine `code`; the
 * stable identifier is `error.type` on the response body (surfaced as
 * `APIError.type`). Dispatch prefers `type`, falls back to `status`
 * when the body was stripped (common with proxies). Name-based
 * detection catches transport-layer timeouts that never produced an
 * HTTP response. The `invalid_request_error` branch additionally
 * sniffs the message for Anthropic's "prompt is too long" phrasing,
 * which is the only signal that a 400 was a context-length overflow.
 *
 * `AIError` instances pass through unchanged so `catch/throw wrap(e)`
 * pipelines never double-wrap.
 *
 * @example
 * try {
 *   return await this.client.messages.create(...);
 * } catch (thrown) {
 *   throw wrapAnthropicError(thrown);
 * }
 */
export function wrapAnthropicError(thrown: unknown): AIError {
  if (thrown instanceof AIError) {
    return thrown;
  }

  const shape = toShape(thrown);
  const context = buildContext(shape);
  const message = shape.message ?? (thrown instanceof Error ? thrown.message : String(thrown));

  if (isTimeout(thrown, shape)) {
    return new ProviderTimeoutError(message, { cause: thrown, context });
  }

  if (shape.type === "authentication_error" || shape.type === "permission_error") {
    return new ProviderAuthError(message, { cause: thrown, context });
  }

  if (shape.status === 401 || shape.status === 403) {
    return new ProviderAuthError(message, { cause: thrown, context });
  }

  if (shape.type === "billing_error") {
    return new QuotaExceededError(message, { cause: thrown, context });
  }

  if (shape.type === "rate_limit_error" || shape.status === 429) {
    return new ProviderRateLimitError(message, {
      cause: thrown,
      context,
      retryAfter: parseRetryAfter(shape.headers),
    });
  }

  if (shape.type === "invalid_request_error" || isClientStatus(shape.status)) {
    if (/prompt is too long/i.test(message)) {
      return new ContextLengthExceededError(message, { cause: thrown, context });
    }

    return new InvalidRequestError(message, { cause: thrown, context });
  }

  return new ProviderError(message, { cause: thrown, context });
}

/**
 * Read the raw error shape without depending on `instanceof APIError`
 * — proxies and re-wrappers strip the prototype chain. Duck-typing on
 * the visible fields is resilient to both.
 */
function toShape(thrown: unknown): AnthropicErrorShape {
  if (thrown instanceof APIError) {
    return {
      status: typeof thrown.status === "number" ? thrown.status : undefined,
      type: thrown.type,
      message: thrown.message,
      headers: thrown.headers,
      name: thrown.name,
      requestId: thrown.requestID ?? undefined,
    };
  }

  if (typeof thrown === "object" && thrown !== null) {
    const raw = thrown as Record<string, unknown>;

    return {
      status: typeof raw.status === "number" ? raw.status : undefined,
      type: typeof raw.type === "string" ? raw.type : undefined,
      message: typeof raw.message === "string" ? raw.message : undefined,
      headers: isHeaderBag(raw.headers) ? raw.headers : undefined,
      name: typeof raw.name === "string" ? raw.name : undefined,
      requestId: readRequestId(raw),
    };
  }

  return {};
}

/**
 * Decide whether the thrown value represents a timeout. The Anthropic
 * SDK throws `APIConnectionTimeoutError` for transport-level timeouts;
 * Node surfaces `ETIMEDOUT` / `ECONNABORTED` on the socket layer.
 * Either signal counts.
 */
function isTimeout(thrown: unknown, shape: AnthropicErrorShape): boolean {
  if (thrown instanceof APIConnectionTimeoutError) {
    return true;
  }

  if (shape.name === "APIConnectionTimeoutError") {
    return true;
  }

  if (typeof thrown === "object" && thrown !== null) {
    const code = (thrown as Record<string, unknown>).code;

    if (code === "ETIMEDOUT" || code === "ECONNABORTED") {
      return true;
    }
  }

  return false;
}

/** True for HTTP 4xx — a client-side request problem, not a server fault. */
function isClientStatus(status: number | undefined): boolean {
  return typeof status === "number" && status >= 400 && status < 500;
}

/**
 * Attach the raw diagnostic fields to `error.context` so consumers
 * have everything the provider surfaced without each subclass having
 * to redeclare them. Never includes `cause` — that lives on
 * `error.cause`.
 */
function buildContext(shape: AnthropicErrorShape): Record<string, unknown> {
  const context: Record<string, unknown> = {};

  if (shape.status !== undefined) {
    context.status = shape.status;
  }

  if (shape.type) {
    context.type = shape.type;
  }

  if (shape.requestId) {
    context.requestId = shape.requestId;
  }

  return context;
}

/** Anthropic exposes the request id as `requestID`; some proxies use `request_id`. */
function readRequestId(raw: Record<string, unknown>): string | undefined {
  if (typeof raw.requestID === "string") {
    return raw.requestID;
  }

  if (typeof raw.request_id === "string") {
    return raw.request_id;
  }

  return undefined;
}

function isHeaderBag(value: unknown): value is HeaderBag {
  return typeof value === "object" && value !== null;
}

/** Read a header value from either a `Headers` instance or a plain record. */
function readHeader(headers: HeaderBag, name: string): string | undefined {
  if (typeof (headers as Headers).get === "function") {
    return (headers as Headers).get(name) ?? undefined;
  }

  const record = headers as Record<string, string>;

  return record[name] ?? record[name.toLowerCase()];
}

/**
 * Parse the `Retry-After` response header (seconds per HTTP spec) into
 * milliseconds so consumers can feed it straight to `setTimeout`.
 * Returns `undefined` when missing or unparseable.
 */
function parseRetryAfter(headers: HeaderBag | undefined): number | undefined {
  if (!headers) {
    return undefined;
  }

  const raw = readHeader(headers, "retry-after") ?? readHeader(headers, "Retry-After");

  if (!raw) {
    return undefined;
  }

  const seconds = Number(raw);

  if (!Number.isFinite(seconds) || seconds < 0) {
    return undefined;
  }

  return Math.round(seconds * 1000);
}
