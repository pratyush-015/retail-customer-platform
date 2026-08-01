/**
 * Customer Segments — RFM distribution and per-segment economics.
 *
 * The point of this page is the gap between headcount and revenue: Champions
 * are a small share of customers and a large share of money. The treemap is
 * sized by revenue and the table shows both percentages side by side, so that
 * asymmetry is the first thing you notice.
 */

import { ArrowDown, ArrowUp, Layers } from 'lucide-react';
import { useMemo } from 'react';
import { ResponsiveContainer, Tooltip, Treemap } from 'recharts';
import { useSegments } from '@/hooks/useApi';
import { useSortable } from '@/hooks/useSortable';
import { AsyncBoundary, EmptyState, SkeletonChart, SkeletonTable } from '@/components/ui/States';
import { Badge, healthTone, InfoTip, PageHeader, Panel } from '@/components/ui/Primitives';
import * as fmt from '@/lib/format';
import type { SegmentStats } from '@/api/types';

/** Health → fill colour. Same source of truth as the badges, so the treemap and
 *  the table can never disagree about which segments are in trouble. */
const HEALTH_FILL: Record<string, string> = {
  strong: 'var(--c-teal-500)',
  stable: 'var(--c-navy-400)',
  watch: 'var(--c-risk-medium)',
  critical: 'var(--c-risk-high)',
};

function fillFor(health: string | null | undefined): string {
  return HEALTH_FILL[health ?? ''] ?? 'var(--c-navy-200)';
}

type SortKey =
  | 'segment'
  | 'customer_count'
  | 'revenue'
  | 'avg_revenue_per_customer'
  | 'avg_recency_days'
  | 'avg_churn_risk';

const COLUMNS: { key: SortKey; label: string; numeric: boolean; hint?: string }[] = [
  { key: 'segment', label: 'Segment', numeric: false },
  { key: 'customer_count', label: 'Customers', numeric: true },
  { key: 'revenue', label: 'Revenue', numeric: true },
  {
    key: 'avg_revenue_per_customer',
    label: 'Rev / customer',
    numeric: true,
    hint: 'Average lifetime revenue per customer in this segment.',
  },
  { key: 'avg_recency_days', label: 'Avg recency', numeric: true },
  {
    key: 'avg_churn_risk',
    label: 'Avg churn risk',
    numeric: true,
    hint: 'Mean predicted churn probability. Blank until the churn model has scored customers.',
  },
];

function SegmentTable({ segments }: { segments: SegmentStats[] }) {
  // Memoised: a fresh accessors object each render would re-sort on every
  // parent update, since useSortable depends on its identity.
  const accessors = useMemo(
    () => ({
      segment: (row: SegmentStats) => row.segment,
      customer_count: (row: SegmentStats) => row.customer_count,
      revenue: (row: SegmentStats) => row.revenue,
      avg_revenue_per_customer: (row: SegmentStats) => row.avg_revenue_per_customer,
      avg_recency_days: (row: SegmentStats) => row.avg_recency_days,
      avg_churn_risk: (row: SegmentStats) => row.avg_churn_risk,
    }),
    [],
  );

  const { sorted, sort, toggle } = useSortable<SegmentStats, SortKey>(segments, accessors, {
    key: 'revenue',
    direction: 'desc',
  });

  return (
    <div className="-mx-card overflow-x-auto">
      <table className="w-full min-w-[720px] border-collapse">
        <thead>
          <tr className="border-b border-navy-100">
            {COLUMNS.map((column) => {
              const isActive = sort.key === column.key;
              const Arrow = sort.direction === 'asc' ? ArrowUp : ArrowDown;
              return (
                <th
                  key={column.key}
                  scope="col"
                  onClick={() => toggle(column.key)}
                  aria-sort={isActive ? (sort.direction === 'asc' ? 'ascending' : 'descending') : 'none'}
                  className={[
                    'th-sortable label-micro px-4 py-2.5',
                    column.numeric ? 'text-right' : 'text-left',
                    isActive ? 'text-navy-700' : '',
                  ].join(' ')}
                >
                  <span
                    className={`inline-flex items-center gap-1 ${column.numeric ? 'flex-row-reverse' : ''}`}
                  >
                    {column.label}
                    {column.hint && <InfoTip text={column.hint} />}
                    {isActive && <Arrow size={11} aria-hidden="true" />}
                  </span>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={row.segment} className="row-hover border-b border-navy-50 last:border-0">
              <td className="px-4 py-2.5">
                <div className="flex items-center gap-2">
                  <span
                    className="h-2.5 w-2.5 shrink-0 rounded-sm"
                    style={{ backgroundColor: fillFor(row.health) }}
                    aria-hidden="true"
                  />
                  <span className="text-body font-medium text-navy-900">{row.segment}</span>
                  <Badge tone={healthTone(row.health)}>{row.health}</Badge>
                </div>
              </td>
              <td className="tabular px-4 py-2.5 text-right">
                <span className="text-body text-navy-900">{fmt.number(row.customer_count)}</span>
                <span className="ml-1.5 text-caption text-navy-400">
                  {fmt.percent(row.customer_pct)}
                </span>
              </td>
              <td className="tabular px-4 py-2.5 text-right">
                <span className="text-body font-medium text-navy-900">
                  {fmt.currencyCompact(row.revenue)}
                </span>
                <span className="ml-1.5 text-caption text-navy-400">
                  {fmt.percent(row.revenue_pct)}
                </span>
              </td>
              <td className="tabular px-4 py-2.5 text-right text-body text-navy-700">
                {fmt.currency(row.avg_revenue_per_customer)}
              </td>
              <td className="tabular px-4 py-2.5 text-right text-body text-navy-700">
                {fmt.days(row.avg_recency_days)}
              </td>
              <td className="tabular px-4 py-2.5 text-right">
                {row.avg_churn_risk === null ? (
                  <span className="text-navy-300">{fmt.EM_DASH}</span>
                ) : (
                  <span
                    className={[
                      'text-body font-medium',
                      row.avg_churn_risk >= 0.7
                        ? 'text-risk-high'
                        : row.avg_churn_risk >= 0.4
                          ? 'text-risk-medium'
                          : 'text-risk-low',
                    ].join(' ')}
                  >
                    {fmt.ratioAsPercent(row.avg_churn_risk)}
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Recharts' Treemap gives content renderers raw geometry; this draws the cell
 *  and suppresses labels on tiles too small to hold them legibly. */
function TreemapCell(props: {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  name?: string;
  health?: string;
  revenue?: number;
}) {
  const { x = 0, y = 0, width = 0, height = 0, name, health, revenue } = props;
  const showLabel = width > 78 && height > 34;

  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={3}
        fill={fillFor(health)}
        fillOpacity={0.9}
        stroke="#fff"
        strokeWidth={2}
        className="transition-opacity duration-200 hover:fill-opacity-100"
      />
      {showLabel && (
        <>
          <text x={x + 8} y={y + 18} fill="#fff" fontSize={11} fontWeight={600}>
            {name}
          </text>
          <text x={x + 8} y={y + 32} fill="#fff" fontSize={10} fillOpacity={0.85}>
            {fmt.currencyCompact(revenue)}
          </text>
        </>
      )}
    </g>
  );
}

export default function Segments() {
  const query = useSegments();

  return (
    <>
      <PageHeader
        title="Customer Segments"
        description="RFM segmentation: customers scored 1–5 on recency, frequency and monetary value, then mapped to named behavioural segments."
      />

      <AsyncBoundary
        state={query}
        skeleton={
          <div className="space-y-6">
            <SkeletonChart height={300} />
            <SkeletonTable rows={8} columns={6} />
          </div>
        }
        // `computed: false` means the RFM job hasn't run — distinct from "ran and
        // found nothing", and fixable by a command, so the message names it.
        isEmpty={(data) => !data.computed || data.segments.length === 0}
        empty={
          <EmptyState
            icon={Layers}
            title="Segmentation not yet computed"
            message="RFM scores have not been calculated for this warehouse. Run `python scripts/run_rfm.py` to populate segments."
          />
        }
      >
        {(data) => {
          const treemapData = data.segments.map((segment) => ({
            name: segment.segment,
            size: segment.revenue,
            revenue: segment.revenue,
            health: segment.health,
          }));

          const topByRevenue = [...data.segments].sort((a, b) => b.revenue - a.revenue)[0];
          const largestByCount = [...data.segments].sort(
            (a, b) => b.customer_count - a.customer_count,
          )[0];

          return (
            <div className="space-y-6">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <div className="card">
                  <p className="label-micro">Segmented customers</p>
                  <p className="tabular mt-2 text-display text-navy-900">
                    {fmt.number(data.total_customers)}
                  </p>
                  <p className="mt-1 text-caption text-navy-400">
                    across {data.segments.length} segments
                  </p>
                </div>
                <div className="card">
                  <p className="label-micro">Attributed revenue</p>
                  <p className="tabular mt-2 text-display text-navy-900">
                    {fmt.currencyCompact(data.attributed_revenue)}
                  </p>
                  <p className="mt-1 text-caption text-navy-400">
                    Excludes guest checkouts with no customer ID
                  </p>
                </div>
                <div className="card">
                  <p className="label-micro">Top segment by revenue</p>
                  <p className="mt-2 text-heading-lg text-navy-900">
                    {topByRevenue?.segment ?? fmt.EM_DASH}
                  </p>
                  {topByRevenue && (
                    <p className="mt-1 text-caption text-navy-400">
                      {fmt.percent(topByRevenue.revenue_pct)} of revenue from{' '}
                      {fmt.percent(topByRevenue.customer_pct)} of customers
                    </p>
                  )}
                </div>
              </div>

              <Panel
                title="Revenue concentration"
                subtitle="Tile area is revenue; colour is segment health"
              >
                <ResponsiveContainer width="100%" height={320}>
                  <Treemap
                    data={treemapData}
                    dataKey="size"
                    stroke="#fff"
                    content={<TreemapCell />}
                    animationDuration={600}
                  >
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const row = payload[0]?.payload as { name: string; revenue: number };
                        return (
                          <div className="rounded-control border border-navy-200 bg-surface px-3 py-2 shadow-popover">
                            <p className="text-caption font-semibold text-navy-900">{row.name}</p>
                            <p className="tabular text-caption text-navy-600">
                              {fmt.currency(row.revenue)}
                            </p>
                          </div>
                        );
                      }}
                    />
                  </Treemap>
                </ResponsiveContainer>

                {largestByCount && topByRevenue && largestByCount.segment !== topByRevenue.segment && (
                  <p className="mt-3 border-t border-navy-100 pt-3 text-caption text-navy-500">
                    <span className="font-semibold text-navy-900">{largestByCount.segment}</span> is
                    the largest segment by headcount ({fmt.percent(largestByCount.customer_pct)} of
                    customers) but contributes {fmt.percent(largestByCount.revenue_pct)} of revenue.
                  </p>
                )}
              </Panel>

              <Panel title="Segment detail" subtitle="Sortable — click any column header">
                <SegmentTable segments={data.segments} />
              </Panel>
            </div>
          );
        }}
      </AsyncBoundary>
    </>
  );
}
