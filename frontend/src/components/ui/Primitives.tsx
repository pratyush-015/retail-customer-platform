/**
 * Shared presentational primitives: panels, badges, KPI cards, tooltips.
 *
 * These exist so risk colours and card chrome are defined once. A `<Badge
 * tone="high">` is guaranteed to match the churn table's red everywhere it
 * appears, which hand-rolled classes per page would not be.
 */

import { HelpCircle, Minus, TrendingDown, TrendingUp } from 'lucide-react';
import type { ReactNode } from 'react';
import * as fmt from '@/lib/format';
import type { RiskBand, SegmentHealth } from '@/api/types';

// ------------------------------------------------------------------ panel

/** A titled container. `action` holds controls (e.g. a granularity toggle)
 *  aligned to the title row. */
export function Panel({
  title,
  subtitle,
  action,
  children,
  className = '',
}: {
  title?: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || action) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-heading-lg text-navy-900">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-caption text-navy-500">{subtitle}</p>}
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

// ----------------------------------------------------------------- badges

export type BadgeTone = 'neutral' | 'high' | 'medium' | 'low' | 'info';

const BADGE_TONES: Record<BadgeTone, string> = {
  neutral: 'bg-navy-100 text-navy-700 border-navy-200',
  high: 'bg-risk-highBg text-risk-high border-risk-highBorder',
  medium: 'bg-risk-mediumBg text-risk-medium border-risk-mediumBorder',
  low: 'bg-risk-lowBg text-risk-low border-risk-lowBorder',
  info: 'bg-teal-50 text-teal-700 border-teal-200',
};

export function Badge({
  children,
  tone = 'neutral',
  className = '',
}: {
  children: ReactNode;
  tone?: BadgeTone;
  className?: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-micro font-semibold uppercase tracking-wide ${BADGE_TONES[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

/** Risk band → tone. The band itself comes from the backend (`risk_band()`),
 *  so the thresholds are never duplicated here. */
export function riskTone(band: RiskBand | string | null | undefined): BadgeTone {
  if (band === 'high') return 'high';
  if (band === 'medium') return 'medium';
  if (band === 'low') return 'low';
  return 'neutral';
}

/** Segment health → tone, driven by the API's `health` field rather than by
 *  matching on segment names. */
export function healthTone(health: SegmentHealth | null | undefined): BadgeTone {
  switch (health) {
    case 'strong':
      return 'low';
    case 'stable':
      return 'info';
    case 'watch':
      return 'medium';
    case 'critical':
      return 'high';
    default:
      return 'neutral';
  }
}

export function RiskBadge({ band }: { band: RiskBand | string | null | undefined }) {
  if (!band) return <span className="text-navy-300">{fmt.EM_DASH}</span>;
  return <Badge tone={riskTone(band)}>{band}</Badge>;
}

// -------------------------------------------------------------- KPI card

/** Month-over-month delta. For revenue and orders, up is good; the caller can
 *  invert via `goodDirection` for metrics like return rate where it isn't. */
export function DeltaIndicator({
  value,
  goodDirection = 'up',
}: {
  value: number | null | undefined;
  goodDirection?: 'up' | 'down';
}) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    // No previous period to compare against — say so rather than showing 0%,
    // which would imply flat performance.
    return <span className="text-caption text-navy-400">No prior period</span>;
  }

  const isFlat = Math.abs(value) < 0.05;
  const isUp = value > 0;
  const isGood = goodDirection === 'up' ? isUp : !isUp;
  const Icon = isFlat ? Minus : isUp ? TrendingUp : TrendingDown;
  const colour = isFlat ? 'text-navy-400' : isGood ? 'text-positive' : 'text-negative';

  return (
    <span className={`inline-flex items-center gap-1 text-caption font-medium ${colour}`}>
      <Icon size={13} aria-hidden="true" />
      <span className="tabular">{fmt.delta(value)}</span>
      <span className="font-normal text-navy-400">vs last month</span>
    </span>
  );
}

export function KpiCard({
  label,
  value,
  delta,
  goodDirection = 'up',
  icon: Icon,
  hint,
  footer,
}: {
  label: string;
  value: string;
  delta?: number | null;
  goodDirection?: 'up' | 'down';
  icon?: typeof TrendingUp;
  hint?: string;
  footer?: ReactNode;
}) {
  return (
    <div className="card-interactive group">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <p className="label-micro">{label}</p>
          {hint && <InfoTip text={hint} />}
        </div>
        {Icon && (
          <Icon
            size={16}
            className="text-navy-300 transition-colors group-hover:text-teal-500"
            aria-hidden="true"
          />
        )}
      </div>
      <p className="tabular mt-2 text-display text-navy-900">{value}</p>
      <div className="mt-2 min-h-[1.125rem]">
        {delta !== undefined && <DeltaIndicator value={delta} goodDirection={goodDirection} />}
        {footer}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- infotip

/**
 * CSS-only tooltip for metric definitions ("AOV = revenue / orders").
 *
 * Deliberately not a JS popover library: these are short static strings, and a
 * hover/focus-driven CSS tooltip has no positioning state to get wrong.
 */
export function InfoTip({ text }: { text: string }) {
  return (
    <span className="group/tip relative inline-flex">
      <button
        type="button"
        className="text-navy-300 transition-colors hover:text-navy-500"
        aria-label={text}
      >
        <HelpCircle size={12} aria-hidden="true" />
      </button>
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-1.5 w-52 -translate-x-1/2 rounded-control bg-navy-900 px-2.5 py-1.5 text-caption font-normal normal-case tracking-normal text-white opacity-0 shadow-popover transition-opacity duration-150 group-hover/tip:opacity-100 group-focus-within/tip:opacity-100"
      >
        {text}
      </span>
    </span>
  );
}

// ---------------------------------------------------------------- toggle

/** Segmented control, used for the week/month granularity switch. */
export function SegmentedControl<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  ariaLabel: string;
}) {
  return (
    <div
      role="group"
      aria-label={ariaLabel}
      className="inline-flex rounded-control border border-navy-200 bg-surface-sunken p-0.5"
    >
      {options.map((option) => {
        const isActive = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            aria-pressed={isActive}
            className={[
              'rounded-[0.375rem] px-2.5 py-1 text-caption font-medium transition-all duration-150',
              isActive
                ? 'bg-surface text-navy-900 shadow-card'
                : 'text-navy-500 hover:text-navy-700',
            ].join(' ')}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

// ------------------------------------------------------------ page header

export function PageHeader({
  title,
  description,
  action,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <header className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-display text-navy-900">{title}</h1>
        {description && (
          <p className="mt-1 max-w-2xl text-body text-navy-500">{description}</p>
        )}
      </div>
      {action}
    </header>
  );
}

// --------------------------------------------------------------- chart tip

/** Recharts' default tooltip is unstyled and ignores the design system, so
 *  every chart passes this instead. */
export function ChartTooltip({
  active,
  payload,
  label,
  formatter,
}: {
  active?: boolean;
  payload?: { name?: string; value?: number | string; color?: string; dataKey?: string }[];
  label?: string | number;
  formatter?: (value: number | string, key: string) => string;
}) {
  if (!active || !payload?.length) return null;

  return (
    <div className="rounded-control border border-navy-200 bg-surface/97 px-3 py-2 shadow-popover backdrop-blur-sm">
      {label !== undefined && (
        <p className="mb-1 text-caption font-semibold text-navy-900">{String(label)}</p>
      )}
      <div className="space-y-0.5">
        {payload.map((entry, index) => (
          <div key={index} className="flex items-center gap-2 text-caption">
            {entry.color && (
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ backgroundColor: entry.color }}
                aria-hidden="true"
              />
            )}
            <span className="text-navy-500">{entry.name ?? entry.dataKey}</span>
            <span className="tabular ml-auto font-medium text-navy-900">
              {formatter && entry.value !== undefined
                ? formatter(entry.value, String(entry.dataKey ?? ''))
                : String(entry.value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
