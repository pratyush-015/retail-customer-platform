/**
 * Client-side sorting for tables whose full result set is already in memory.
 *
 * Appropriate here because the segment table is ~10 rows and the at-risk table
 * is capped by a `limit` param. Anything unbounded would need server-side
 * sorting instead, which is why this is a hook and not baked into a table
 * component: the churn table can opt out later without a rewrite.
 */

import { useMemo, useState } from 'react';

export type SortDirection = 'asc' | 'desc';

export interface SortState<K extends string> {
  key: K;
  direction: SortDirection;
}

export function useSortable<T, K extends string>(
  rows: T[] | undefined,
  accessors: Record<K, (row: T) => number | string | null | undefined>,
  initial: SortState<K>,
) {
  const [sort, setSort] = useState<SortState<K>>(initial);

  /** Clicking the active column flips direction; a new column starts descending,
   *  which is what you want for revenue and risk (the interesting end first). */
  const toggle = (key: K) => {
    setSort((current) =>
      current.key === key
        ? { key, direction: current.direction === 'asc' ? 'desc' : 'asc' }
        : { key, direction: 'desc' },
    );
  };

  const sorted = useMemo(() => {
    if (!rows) return [];
    const accessor = accessors[sort.key];
    const factor = sort.direction === 'asc' ? 1 : -1;

    return [...rows].sort((a, b) => {
      const left = accessor(a);
      const right = accessor(b);

      // Nulls sort last in both directions. An un-scored customer is not
      // "lowest risk", so it must not lead a descending sort.
      const leftMissing = left === null || left === undefined;
      const rightMissing = right === null || right === undefined;
      if (leftMissing && rightMissing) return 0;
      if (leftMissing) return 1;
      if (rightMissing) return -1;

      if (typeof left === 'string' || typeof right === 'string') {
        return String(left).localeCompare(String(right)) * factor;
      }
      return (Number(left) - Number(right)) * factor;
    });
  }, [rows, sort, accessors]);

  return { sorted, sort, toggle };
}
