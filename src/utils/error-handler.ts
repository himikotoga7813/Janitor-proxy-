export class ProxyError extends Error {
  constructor(
    message: string,
    public statusCode: number = 500,
    public code?: string
  ) {
    super(message);
    this.name = 'ProxyError';
  }
}

export function createErrorResponse(error: unknown): Response {
  const isProxyError = error instanceof ProxyError;
  const statusCode = isProxyError
    ? error.statusCode
    : getErrorStatusCode(error) || 500;
  const message =
    error instanceof Error ? error.message : 'Unknown error occurred';

  return new Response(
    JSON.stringify({
      error: {
        message,
        type: isProxyError ? 'proxy_error' : 'internal_error',
        ...(isProxyError && error.code ? { code: error.code } : {}),
      },
    }),
    {
      status: statusCode,
      headers: { 'Content-Type': 'application/json' },
    }
  );
}

interface ErrorLike {
  status?: number;
  statusCode?: number;
  message?: string;
  code?: string;
}

export function getErrorStatusCode(error: unknown): number | undefined {
  if (!error || typeof error !== 'object') return undefined;
  const e = error as ErrorLike;
  const candidate = e.statusCode ?? e.status;
  return typeof candidate === 'number' && candidate >= 400 && candidate <= 599
    ? candidate
    : undefined;
}

export function isRateLimitError(error: unknown): boolean {
  const e = error as ErrorLike;
  return (
    e.status === 429 ||
    e.statusCode === 429 ||
    (typeof e.message === 'string' &&
      /rate.?limit|too many requests/i.test(e.message)) ||
    e.code === 'rate_limit_exceeded'
  );
}

export function isRetryableError(error: unknown): boolean {
  const e = error as ErrorLike;
  const status = getErrorStatusCode(error);
  return (
    isRateLimitError(error) ||
    [408, 425, 500, 502, 503, 504, 529].includes(status || 0) ||
    (typeof e.message === 'string' &&
      /timeout|timed out|overloaded|temporarily unavailable|network error|fetch failed|connection reset/i.test(
        e.message
      ))
  );
}

/** Default upstream connection/initial-response timeout (milliseconds). */
const DEFAULT_PROVIDER_TIMEOUT_MS = 120000;
const MAX_PROVIDER_TIMEOUT_MS = 600000;

/** Read an optional timeout override without allowing zero, NaN, or extreme values. */
export function getProviderTimeoutMs(env?: { PROVIDER_TIMEOUT_MS?: string }): number {
  const configured = Number(env?.PROVIDER_TIMEOUT_MS);
  if (!Number.isFinite(configured) || configured <= 0) {
    return DEFAULT_PROVIDER_TIMEOUT_MS;
  }
  return Math.min(Math.max(Math.floor(configured), 1000), MAX_PROVIDER_TIMEOUT_MS);
}

/**
 * Wrap the initial provider operation with a timeout.
 * For streaming APIs this covers connection/stream creation, not the stream's full lifetime.
 * The underlying operation may continue after timeout if the provider SDK offers no abort hook.
 */
export function withTimeout<T>(
  promise: Promise<T>,
  ms: number = DEFAULT_PROVIDER_TIMEOUT_MS
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Provider timeout after ${ms}ms`)), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      }
    );
  });
}
