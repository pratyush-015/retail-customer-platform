/**
 * Display formatting, centralised.
 *
 * The dataset is a UK retailer, so currency is GBP and dates are read as UK
 * order dates. Every formatter tolerates null/undefined and returns an em dash,
 * because the API legitimately returns null for un-computed analytics fields —
 * that is what stops "£undefined" or "NaN%" reaching the screen.
 */

const EM_DASH = '—';

const gbp0 = new Intl.NumberFormat('en-GB', {
  style: 'currency',
  currency: 'GBP',
  maximumFractionDigits: 0,
});

const gbp2 = new Intl.NumberFormat('en-GB', {
  style: 'currency',
  currency: 'GBP',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const int = new Intl.NumberFormat('en-GB', { maximumFractionDigits: 0 });

function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

/** Whole-pound currency, for totals where pennies are noise. */
export function currency(value: number | null | undefined): string {
  return isNumber(value) ? gbp0.format(value) : EM_DASH;
}

/** Two-decimal currency, for per-unit values like AOV where pennies matter. */
export function currencyPrecise(value: number | null | undefined): string {
  return isNumber(value) ? gbp2.format(value) : EM_DASH;
}

/**
 * Abbreviated currency for chart axes and KPI cards: £1.2M, £48.3K.
 *
 * Axis labels have no room for full numbers, and "£1,234,567" on a tick makes
 * charts unreadable. Thresholds are chosen so 3 significant figures survive.
 */
export function currencyCompact(value: number | null | undefined): string {
  if (!isNumber(value)) return EM_DASH;
  const abs = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  if (abs >= 1_000_000_000) return `${sign}£${(abs / 1_000_000_000).toFixed(1)}B`;
  if (abs >= 1_000_000) return `${sign}£${(abs / 1_000_000).toFixed(abs >= 10_000_000 ? 0 : 1)}M`;
  if (abs >= 1_000) return `${sign}£${(abs / 1_000).toFixed(abs >= 10_000 ? 0 : 1)}K`;
  return `${sign}£${abs.toFixed(0)}`;
}

export function number(value: number | null | undefined): string {
  return isNumber(value) ? int.format(value) : EM_DASH;
}

export function numberCompact(value: number | null | undefined): string {
  if (!isNumber(value)) return EM_DASH;
  const abs = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  if (abs >= 1_000_000) return `${sign}${(abs / 1_000_000).toFixed(1)}M`;
  if (abs >= 10_000) return `${sign}${(abs / 1_000).toFixed(0)}K`;
  return int.format(value);
}

/** A value already expressed as a percentage (e.g. 12.4 → "12.4%"). */
export function percent(
  value: number | null | undefined,
  digits = 1,
): string {
  return isNumber(value) ? `${value.toFixed(digits)}%` : EM_DASH;
}

/** A 0–1 ratio rendered as a percentage (e.g. 0.124 → "12.4%"). */
export function ratioAsPercent(
  value: number | null | undefined,
  digits = 1,
): string {
  return isNumber(value) ? `${(value * 100).toFixed(digits)}%` : EM_DASH;
}

/** Signed percentage for deltas, so +3.2% reads as growth at a glance. */
export function delta(value: number | null | undefined, digits = 1): string {
  if (!isNumber(value)) return EM_DASH;
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(digits)}%`;
}

/** Model scores and p-values: fixed decimals, never abbreviated. */
export function decimal(value: number | null | undefined, digits = 3): string {
  return isNumber(value) ? value.toFixed(digits) : EM_DASH;
}

/**
 * p-values below 0.001 are shown as "< 0.001" rather than "0.000", which would
 * wrongly imply the p-value is exactly zero.
 */
export function pValue(value: number | null | undefined): string {
  if (!isNumber(value)) return EM_DASH;
  if (value < 0.001) return '< 0.001';
  return value.toFixed(3);
}

/** ISO date → "12 Mar 2011". Date-only strings are parsed as UTC to avoid a
 *  local-timezone shift moving an order to the previous day. */
export function date(value: string | null | undefined): string {
  if (!value) return EM_DASH;
  const parsed = new Date(/^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T00:00:00Z` : value);
  if (Number.isNaN(parsed.getTime())) return EM_DASH;
  return parsed.toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    timeZone: 'UTC',
  });
}

/** "2011-03" or "2011-W12" → a human label for chart axes. */
export function periodLabel(value: string | null | undefined): string {
  if (!value) return EM_DASH;

  const month = /^(\d{4})-(\d{2})$/.exec(value);
  if (month) {
    const [, year, monthNum] = month;
    const monthIndex = Number(monthNum) - 1;
    const names = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    ];
    const name = names[monthIndex];
    if (name) return `${name} ${year?.slice(2)}`;
  }

  const week = /^(\d{4})-W(\d{1,2})$/.exec(value);
  if (week) return `W${week[2]} '${week[1]?.slice(2)}`;

  return value;
}

/** "recency_days" → "Recency days", for feature-importance labels. */
export function humanise(value: string | null | undefined): string {
  if (!value) return EM_DASH;
  const spaced = value.replace(/_/g, ' ').trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** Number of days rendered with its unit, singular where appropriate. */
export function days(value: number | null | undefined): string {
  if (!isNumber(value)) return EM_DASH;
  const rounded = Math.round(value);
  return `${int.format(rounded)} ${rounded === 1 ? 'day' : 'days'}`;
}

export { EM_DASH };
