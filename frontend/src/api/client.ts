/**
 * The single fetch wrapper every request goes through.
 *
 * Its job is to collapse the many ways an HTTP call can fail into one typed
 * `ApiError`, so components branch on `error.kind` instead of inspecting raw
 * responses. Without this, each page would re-derive "was that a 404, a dead
 * backend, or malformed JSON?" and would get it subtly wrong.
 */

import type { ApiErrorBody } from './types';

/**
 * Empty by default: the Vite dev server proxies `/api` to the backend, so
 * requests stay same-origin locally. In production (Vercel) this is set to the
 * deployed backend's origin.
 */
const BASE_URL = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '');

/** How a failure should be *presented*, which is not the same as its status. */
export type ApiErrorKind =
  | 'network' // backend unreachable — offer retry
  | 'not_found' // 404 — the entity genuinely doesn't exist
  | 'validation' // 422 — the request was wrong, retrying unchanged won't help
  | 'unavailable' // 503 — dependency (usually the model) missing, may recover
  | 'server' // 5xx — our bug
  | 'client' // other 4xx
  | 'parse'; // 2xx with a body we couldn't read

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number;
  /** Backend's machine-readable code, e.g. `CUSTOMER_NOT_FOUND`. */
  readonly code: string;
  readonly details?: unknown;

  constructor(args: {
    kind: ApiErrorKind;
    status: number;
    code: string;
    message: string;
    details?: unknown;
  }) {
    super(args.message);
    this.name = 'ApiError';
    this.kind = args.kind;
    this.status = args.status;
    this.code = args.code;
    this.details = args.details;
  }

  /** True when retrying the identical request could plausibly succeed.
   *  A 404 or 422 cannot, so those error states omit the retry button. */
  get isRetryable(): boolean {
    return this.kind === 'network' || this.kind === 'server' || this.kind === 'unavailable';
  }
}

function kindForStatus(status: number): ApiErrorKind {
  if (status === 404) return 'not_found';
  if (status === 422) return 'validation';
  if (status === 503) return 'unavailable';
  if (status >= 500) return 'server';
  return 'client';
}

/**
 * Pull the message out of the backend's `{error: {...}}` envelope.
 *
 * Also handles FastAPI's built-in 422 shape, which bypasses our handler and
 * arrives as `{detail: [{loc, msg}]}`. Those are flattened to
 * "field: reason" so the UI can name the offending field.
 */
async function toApiError(response: Response): Promise<ApiError> {
  const status = response.status;
  const kind = kindForStatus(status);

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    return new ApiError({
      kind,
      status,
      code: 'MALFORMED_ERROR_RESPONSE',
      message: `Request failed with status ${status}.`,
    });
  }

  const envelope = body as Partial<ApiErrorBody>;
  if (envelope?.error?.message) {
    return new ApiError({
      kind,
      status,
      code: envelope.error.code ?? 'UNKNOWN',
      message: envelope.error.message,
      details: envelope.error.details,
    });
  }

  const detail = (body as { detail?: unknown })?.detail;
  if (Array.isArray(detail)) {
    const fields = detail
      .map((item) => {
        const entry = item as { loc?: unknown[]; msg?: string };
        const path = Array.isArray(entry.loc)
          ? entry.loc.filter((p) => p !== 'body' && p !== 'query').join('.')
          : '';
        return path ? `${path}: ${entry.msg ?? 'invalid'}` : (entry.msg ?? 'invalid');
      })
      .join('; ');
    return new ApiError({
      kind: 'validation',
      status,
      code: 'VALIDATION_ERROR',
      message: fields || 'The request was rejected as invalid.',
      details: detail,
    });
  }

  return new ApiError({
    kind,
    status,
    code: 'UNKNOWN',
    message: typeof detail === 'string' ? detail : `Request failed with status ${status}.`,
  });
}

export type QueryParams = Record<
  string,
  string | number | boolean | null | undefined
>;

/** Drop null/undefined so optional filters don't become `?segment=undefined`. */
function buildQuery(params?: QueryParams): string {
  if (!params) return '';
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === '') continue;
    search.append(key, String(value));
  }
  const query = search.toString();
  return query ? `?${query}` : '';
}

export interface RequestOptions {
  params?: QueryParams;
  signal?: AbortSignal;
  method?: 'GET' | 'POST';
  body?: unknown;
  /** Aborts slow requests so a hung backend surfaces as a retryable error
   *  rather than an indefinite skeleton. */
  timeoutMs?: number;
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { params, signal, method = 'GET', body, timeoutMs = 30_000 } = options;
  const url = `${BASE_URL}${path}${buildQuery(params)}`;

  // Combine the caller's abort signal (React Query cancelling on unmount) with
  // our own timeout, so whichever fires first cancels the request.
  const timeoutController = new AbortController();
  const timer = setTimeout(() => timeoutController.abort(), timeoutMs);
  const onExternalAbort = () => timeoutController.abort();
  signal?.addEventListener('abort', onExternalAbort);

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      headers: {
        Accept: 'application/json',
        ...(body ? { 'Content-Type': 'application/json' } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: timeoutController.signal,
    });
  } catch (cause) {
    // A caller-initiated abort is not an error worth showing; rethrow so React
    // Query treats it as a cancellation rather than a failure.
    if (signal?.aborted) throw cause;
    throw new ApiError({
      kind: 'network',
      status: 0,
      code: 'NETWORK_ERROR',
      message:
        'Could not reach the API. Check that the backend is running and try again.',
      details: cause,
    });
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onExternalAbort);
  }

  if (!response.ok) throw await toApiError(response);

  if (response.status === 204) return undefined as T;

  try {
    return (await response.json()) as T;
  } catch (cause) {
    throw new ApiError({
      kind: 'parse',
      status: response.status,
      code: 'MALFORMED_RESPONSE',
      message: 'The API returned a response that could not be read.',
      details: cause,
    });
  }
}
