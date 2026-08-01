/**
 * React Query wrappers for every API endpoint.
 *
 * One hook per resource, named after what it fetches. Each hook:
 *   - Defines a stable, predictable cache key so invalidations are surgical.
 *   - Sets sensible defaults (staleTime, retry) without repeating them in
 *     every component.
 *   - Passes the AbortSignal so unmounting a component cancels the fetch.
 *
 * Stale-time rationale: analytics data is refreshed by an overnight pipeline,
 * not by user actions. A 5-minute stale time means a reviewer clicking between
 * pages doesn't hammer the DB; navigating back within a session is instant.
 */

import {
  useQuery,
  useMutation,
  useQueryClient,
  type UseQueryOptions,
} from '@tanstack/react-query';
import {
  fetchHealth,
  fetchKPISummary,
  fetchSalesTrend,
  fetchSalesByCountry,
  fetchSegments,
  fetchCustomer,
  fetchAtRisk,
  fetchModelPerformance,
  fetchInsights,
  predictChurn,
} from '@/api/queries';
import type { ApiError } from '@/api/client';
import type {
  Granularity,
  ChurnPredictionRequest,
  ChurnPredictionResponse,
} from '@/api/types';

const STALE_5M = 5 * 60 * 1000;
const STALE_1M = 60 * 1000;

// Typed shorthand so callers don't write <SomeType, ApiError> on every hook.
type QO<T> = Omit<UseQueryOptions<T, ApiError>, 'queryKey' | 'queryFn'>;

// ---------------------------------------------------------------- health

export function useHealth(options?: QO<Awaited<ReturnType<typeof fetchHealth>>>) {
  return useQuery({
    queryKey: ['health'],
    queryFn: ({ signal }) => fetchHealth(signal),
    staleTime: STALE_1M,
    retry: 1,
    ...options,
  });
}

// ------------------------------------------------------------------ kpis

export function useKPISummary(
  params: Parameters<typeof fetchKPISummary>[0] = {},
  options?: QO<Awaited<ReturnType<typeof fetchKPISummary>>>,
) {
  return useQuery({
    queryKey: ['kpis', 'summary', params],
    queryFn: ({ signal }) => fetchKPISummary(params, signal),
    staleTime: STALE_5M,
    ...options,
  });
}

// ----------------------------------------------------------------- sales

export function useSalesTrend(
  params: Parameters<typeof fetchSalesTrend>[0] = {},
  options?: QO<Awaited<ReturnType<typeof fetchSalesTrend>>>,
) {
  return useQuery({
    queryKey: ['sales', 'trend', params],
    queryFn: ({ signal }) => fetchSalesTrend(params, signal),
    staleTime: STALE_5M,
    ...options,
  });
}

export function useSalesByCountry(
  params: Parameters<typeof fetchSalesByCountry>[0] = {},
  options?: QO<Awaited<ReturnType<typeof fetchSalesByCountry>>>,
) {
  return useQuery({
    queryKey: ['sales', 'by-country', params],
    queryFn: ({ signal }) => fetchSalesByCountry(params, signal),
    staleTime: STALE_5M,
    ...options,
  });
}

// ------------------------------------------------------------- customers

export function useSegments(
  options?: QO<Awaited<ReturnType<typeof fetchSegments>>>,
) {
  return useQuery({
    queryKey: ['customers', 'segments'],
    queryFn: ({ signal }) => fetchSegments(signal),
    staleTime: STALE_5M,
    ...options,
  });
}

export function useCustomer(
  customerId: number | string | undefined,
  params: Parameters<typeof fetchCustomer>[1] = {},
  options?: QO<Awaited<ReturnType<typeof fetchCustomer>>>,
) {
  return useQuery({
    queryKey: ['customers', 'profile', customerId, params],
    queryFn: ({ signal }) => fetchCustomer(customerId!, params, signal),
    enabled: customerId != null && customerId !== '',
    staleTime: STALE_5M,
    ...options,
  });
}

// ----------------------------------------------------------------- churn

export function useAtRisk(
  params: Parameters<typeof fetchAtRisk>[0] = {},
  options?: QO<Awaited<ReturnType<typeof fetchAtRisk>>>,
) {
  return useQuery({
    queryKey: ['churn', 'at-risk', params],
    queryFn: ({ signal }) => fetchAtRisk(params, signal),
    staleTime: STALE_5M,
    ...options,
  });
}

export function useModelPerformance(
  options?: QO<Awaited<ReturnType<typeof fetchModelPerformance>>>,
) {
  return useQuery({
    queryKey: ['churn', 'model-performance'],
    queryFn: ({ signal }) => fetchModelPerformance(signal),
    staleTime: STALE_5M,
    ...options,
  });
}

/** Mutation because prediction is POST and should not be cached globally. */
export function usePredictChurn() {
  const queryClient = useQueryClient();
  return useMutation<ChurnPredictionResponse, ApiError, ChurnPredictionRequest>({
    mutationFn: predictChurn,
    onSuccess: () => {
      // Invalidate the at-risk list in case a reviewer submits updated values
      // for a customer that was already in the list.
      void queryClient.invalidateQueries({ queryKey: ['churn', 'at-risk'] });
    },
  });
}

// ------------------------------------------------------------------ stats

export function useInsights(
  params: Parameters<typeof fetchInsights>[0] = {},
  options?: QO<Awaited<ReturnType<typeof fetchInsights>>>,
) {
  return useQuery({
    queryKey: ['stats', 'insights', params],
    queryFn: ({ signal }) => fetchInsights(params, signal),
    staleTime: STALE_5M,
    ...options,
  });
}

// ----------------------------------------------------------------- utils

/** Re-export these so callers get the typed version, not the raw react-query one. */
export { type Granularity };
