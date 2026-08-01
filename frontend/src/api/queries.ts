/**
 * Every API call in one place.
 *
 * Each function calls `request<T>()` and returns a plain Promise. Hooks
 * compose these with React Query in `hooks/`. Keeping them separate means:
 *   a) query functions are testable in isolation.
 *   b) the same function can be used in `queryClient.prefetchQuery()` without
 *      importing a hook.
 */

import { request } from './client';
import type {
  AtRiskResponse,
  ChurnPredictionRequest,
  ChurnPredictionResponse,
  CustomerProfile,
  Granularity,
  HealthResponse,
  InsightsResponse,
  KPISummary,
  ModelPerformance,
  SegmentDistribution,
  SalesByCountry,
  SalesTrend,
} from './types';

// ---------------------------------------------------------------- health

export function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request('/api/health', { signal });
}

// ------------------------------------------------------------------ kpis

export function fetchKPISummary(
  params: { start_date?: string; end_date?: string } = {},
  signal?: AbortSignal,
): Promise<KPISummary> {
  return request('/api/kpis/summary', { params, signal });
}

// ----------------------------------------------------------------- sales

export function fetchSalesTrend(
  params: {
    granularity?: Granularity;
    start_date?: string;
    end_date?: string;
  } = {},
  signal?: AbortSignal,
): Promise<SalesTrend> {
  return request('/api/sales/trend', { params, signal });
}

export function fetchSalesByCountry(
  params: { limit?: number; start_date?: string; end_date?: string } = {},
  signal?: AbortSignal,
): Promise<SalesByCountry> {
  return request('/api/sales/by-country', { params, signal });
}

// ------------------------------------------------------------- customers

export function fetchSegments(signal?: AbortSignal): Promise<SegmentDistribution> {
  return request('/api/customers/segments', { signal });
}

export function fetchCustomer(
  customerId: number | string,
  params: { history_limit?: number } = {},
  signal?: AbortSignal,
): Promise<CustomerProfile> {
  return request(`/api/customers/${customerId}`, { params, signal });
}

// ----------------------------------------------------------------- churn

export function fetchAtRisk(
  params: {
    threshold?: number;
    limit?: number;
    offset?: number;
    segment?: string;
    country?: string;
  } = {},
  signal?: AbortSignal,
): Promise<AtRiskResponse> {
  return request('/api/churn/at-risk', { params, signal });
}

export function fetchModelPerformance(signal?: AbortSignal): Promise<ModelPerformance> {
  return request('/api/churn/model-performance', { signal });
}

export function predictChurn(
  payload: ChurnPredictionRequest,
  signal?: AbortSignal,
): Promise<ChurnPredictionResponse> {
  return request('/api/predict/churn', {
    method: 'POST',
    body: payload,
    signal,
  });
}

// ------------------------------------------------------------------ stats

export function fetchInsights(
  params: { alpha?: number; top_n?: number } = {},
  signal?: AbortSignal,
): Promise<InsightsResponse> {
  return request('/api/stats/insights', { params, signal });
}
