/**
 * The three states every data-driven component must handle: loading, error and
 * empty. Centralising them is what makes "no blank screens, no spinner-forever"
 * enforceable rather than aspirational.
 *
 * The key component is `AsyncBoundary`, which takes a React Query result and
 * renders exactly one of skeleton / error / empty / content. Pages therefore
 * cannot accidentally forget a state — the boundary owns the decision.
 */

import { AlertTriangle, Inbox, RefreshCw, ServerCrash, SearchX, WifiOff } from 'lucide-react';
import type { CSSProperties, ReactNode } from 'react';
import { ApiError } from '@/api/client';

// ------------------------------------------------------------- skeletons

/** A single shimmering block. `.skeleton` carries the animation (see index.css).
 *  Accepts `style` because bar heights and row opacity are computed, not fixed
 *  to a Tailwind step. */
export function Skeleton({
  className = '',
  style,
}: {
  className?: string;
  style?: CSSProperties;
}) {
  return <div className={`skeleton ${className}`} style={style} aria-hidden="true" />;
}

/**
 * Skeletons mirror the shape of the content they replace, so the layout doesn't
 * jump when real data arrives. A generic spinner cannot do that.
 */
export function SkeletonCard() {
  return (
    <div className="card" aria-busy="true">
      <Skeleton className="h-3 w-24" />
      <Skeleton className="mt-3 h-8 w-32" />
      <Skeleton className="mt-3 h-3 w-20" />
    </div>
  );
}

export function SkeletonChart({ height = 280 }: { height?: number }) {
  return (
    <div className="card" aria-busy="true">
      <Skeleton className="h-4 w-40" />
      <Skeleton className="mt-2 h-3 w-56" />
      <div className="mt-6 flex items-end gap-2" style={{ height }}>
        {/* Varied bar heights read as a chart rather than a loading block. */}
        {[45, 62, 38, 78, 55, 88, 42, 70, 58, 92, 48, 66].map((pct, index) => (
          <Skeleton key={index} className="flex-1" style={{ height: `${pct}%` }} />
        ))}
      </div>
    </div>
  );
}

export function SkeletonTable({ rows = 6, columns = 5 }: { rows?: number; columns?: number }) {
  return (
    <div className="card overflow-hidden" aria-busy="true">
      <Skeleton className="h-4 w-40" />
      <div className="mt-5 space-y-3">
        <div className="flex gap-4 border-b border-navy-100 pb-3">
          {Array.from({ length: columns }).map((_, index) => (
            <Skeleton key={index} className="h-3 flex-1" />
          ))}
        </div>
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <div key={rowIndex} className="flex gap-4 py-1">
            {Array.from({ length: columns }).map((_, columnIndex) => (
              <Skeleton
                key={columnIndex}
                className="h-3 flex-1"
                // Taper opacity down the rows so the eye reads it as a list
                // fading out, not a solid block.
                style={{ opacity: 1 - rowIndex * 0.1 }}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

// ----------------------------------------------------------------- errors

/** Icon and copy chosen from the error's kind, so a missing model reads
 *  differently from a dead backend. */
function errorPresentation(error: unknown): {
  Icon: typeof AlertTriangle;
  title: string;
  hint: string;
  tone: 'warning' | 'danger';
} {
  if (error instanceof ApiError) {
    switch (error.kind) {
      case 'network':
        return {
          Icon: WifiOff,
          title: 'Cannot reach the API',
          hint: 'The backend may still be starting up. Check it is running on port 8000.',
          tone: 'warning',
        };
      case 'unavailable':
        return {
          Icon: ServerCrash,
          title: 'Feature temporarily unavailable',
          // Almost always the churn model artifact being absent, which is
          // fixable by running the training script — so say that.
          hint: 'This usually means the churn model has not been trained yet. Run `python scripts/train_churn_model.py`.',
          tone: 'warning',
        };
      case 'not_found':
        return {
          Icon: SearchX,
          title: 'Not found',
          hint: 'Check the identifier and try again.',
          tone: 'warning',
        };
      case 'validation':
        return {
          Icon: AlertTriangle,
          title: 'Invalid request',
          hint: 'Adjust the filters and try again.',
          tone: 'warning',
        };
      default:
        return {
          Icon: ServerCrash,
          title: 'Something went wrong',
          hint: 'The API returned an unexpected error.',
          tone: 'danger',
        };
    }
  }
  return {
    Icon: AlertTriangle,
    title: 'Something went wrong',
    hint: 'An unexpected error occurred while rendering this view.',
    tone: 'danger',
  };
}

export function ErrorState({
  error,
  onRetry,
  compact = false,
}: {
  error: unknown;
  onRetry?: () => void;
  compact?: boolean;
}) {
  const { Icon, title, hint, tone } = errorPresentation(error);
  const message = error instanceof Error ? error.message : String(error);
  // Retrying a 404 or 422 unchanged cannot succeed, so the button is withheld
  // rather than inviting a pointless click.
  const canRetry = Boolean(onRetry) && (!(error instanceof ApiError) || error.isRetryable);
  const code = error instanceof ApiError ? error.code : null;

  return (
    <div
      role="alert"
      className={[
        'flex flex-col items-center justify-center rounded-card border text-center',
        compact ? 'gap-2 p-5' : 'gap-3 p-8',
        tone === 'danger'
          ? 'border-risk-highBorder bg-risk-highBg'
          : 'border-risk-mediumBorder bg-risk-mediumBg',
      ].join(' ')}
    >
      <Icon
        className={tone === 'danger' ? 'text-risk-high' : 'text-risk-medium'}
        size={compact ? 20 : 26}
        aria-hidden="true"
      />
      <div>
        <p className="text-heading text-navy-900">{title}</p>
        <p className="mt-1 max-w-md text-caption text-navy-600">{message}</p>
        {!compact && <p className="mt-2 max-w-md text-caption text-navy-400">{hint}</p>}
      </div>
      {canRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-1 inline-flex items-center gap-1.5 rounded-control border border-navy-200 bg-surface px-3 py-1.5 text-caption font-medium text-navy-700 transition-colors hover:bg-navy-50"
        >
          <RefreshCw size={13} aria-hidden="true" />
          Retry
        </button>
      )}
      {code && <p className="font-mono text-micro text-navy-400">{code}</p>}
    </div>
  );
}

// ------------------------------------------------------------------ empty

/** A designed empty state: icon, what happened, and what to do about it. */
export function EmptyState({
  title,
  message,
  icon: Icon = Inbox,
  action,
  compact = false,
}: {
  title: string;
  message?: string;
  icon?: typeof Inbox;
  action?: ReactNode;
  compact?: boolean;
}) {
  return (
    <div
      className={[
        'flex flex-col items-center justify-center rounded-card border border-dashed border-navy-200 bg-surface-sunken/50 text-center',
        compact ? 'gap-2 p-6' : 'gap-3 p-10',
      ].join(' ')}
    >
      <div className="rounded-full bg-navy-100 p-2.5">
        <Icon className="text-navy-400" size={compact ? 18 : 22} aria-hidden="true" />
      </div>
      <div>
        <p className="text-heading text-navy-700">{title}</p>
        {message && <p className="mt-1 max-w-sm text-caption text-navy-500">{message}</p>}
      </div>
      {action}
    </div>
  );
}

// -------------------------------------------------------- async boundary

/** The subset of a React Query result the boundary needs. Structural typing
 *  keeps it usable with both queries and mutations. */
export interface AsyncState<T> {
  data: T | undefined;
  isPending: boolean;
  isError: boolean;
  error: unknown;
  refetch?: () => void;
}

/**
 * Renders exactly one of: skeleton, error, empty, or content.
 *
 * Order matters. Error is checked before data because React Query keeps stale
 * data on a failed refetch, and showing stale numbers under a silent failure is
 * worse than showing the failure.
 */
export function AsyncBoundary<T>({
  state,
  skeleton,
  children,
  isEmpty,
  empty,
  compactError = false,
}: {
  state: AsyncState<T>;
  skeleton: ReactNode;
  children: (data: T) => ReactNode;
  /** Distinguishes "loaded, genuinely nothing to show" from "still loading". */
  isEmpty?: (data: T) => boolean;
  empty?: ReactNode;
  compactError?: boolean;
}) {
  if (state.isPending) return <>{skeleton}</>;

  if (state.isError || state.data === undefined) {
    return (
      <ErrorState
        error={state.error ?? new Error('No data was returned.')}
        onRetry={state.refetch}
        compact={compactError}
      />
    );
  }

  if (isEmpty?.(state.data)) {
    return <>{empty ?? <EmptyState title="No data available" />}</>;
  }

  return <div className="animate-fade-in-up">{children(state.data)}</div>;
}
