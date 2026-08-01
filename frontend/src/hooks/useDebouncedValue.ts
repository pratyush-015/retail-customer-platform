import { useEffect, useState } from 'react';

/**
 * Trails a rapidly-changing value so it can be used as a query key.
 *
 * Needed by the churn threshold slider: dragging it fires a change per pixel,
 * and keying a query directly off that would queue dozens of requests for
 * thresholds the user never stopped on. The slider stays responsive because the
 * raw value still drives the thumb; only the fetch waits.
 */
export function useDebouncedValue<T>(value: T, delayMs = 300): T {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);

  return debounced;
}
