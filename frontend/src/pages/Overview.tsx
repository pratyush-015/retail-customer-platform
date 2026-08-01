/**
 * Executive Overview — the landing page.
 *
 * Answers, in order: how much did we make, is it growing, when did it happen,
 * and where did it come from. KPI cards first because they're the only thing
 * some viewers will read; charts below for the people who want the shape.
 *
 * Each section owns its own query and its own AsyncBoundary, so a failing
 * country query still leaves the KPI cards and trend chart on screen. One
 * page-level boundary would blank all three.
 */

import {
  Banknote,
  Globe2,
  Receipt,
  ShoppingCart,
  TrendingUp,
  Undo2,
  Users,
} from 'lucide-react';
import { useState } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useKPISummary, useSalesByCountry, useSalesTrend } from '@/hooks/useApi';
import {
  AsyncBoundary,
  EmptyState,
  SkeletonCard,
  SkeletonChart,
} from '@/components/ui/States';
import {
  ChartTooltip,
  KpiCard,
  Panel,
  PageHeader,
  SegmentedControl,
} from '@/components/ui/Primitives';
import * as fmt from '@/lib/format';
import type { Granularity } from '@/api/types';

// Axis/grid styling repeated across charts; declared once so every chart on the
// page shares identical tick treatment.
const AXIS = {
  tick: { fill: 'var(--c-navy-400)', fontSize: 11 },
  stroke: 'var(--c-navy-200)',
} as const;

function KpiRow() {
  const query = useKPISummary();

  return (
    <AsyncBoundary
      state={query}
      skeleton={
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => (
            <SkeletonCard key={index} />
          ))}
        </div>
      }
      // Zero orders means the warehouse is empty, which is a setup problem, not
      // a business result — so it gets an empty state pointing at the ETL.
      isEmpty={(data) => data.total_orders === 0}
      empty={
        <EmptyState
          title="No sales data loaded"
          message="The warehouse is empty. Run the ETL pipeline with `python scripts/seed_all.py` to load the dataset."
          icon={Receipt}
        />
      }
    >
      {(data) => (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <KpiCard
              label="Total revenue"
              value={fmt.currencyCompact(data.total_revenue)}
              delta={data.mom.revenue_pct}
              icon={Banknote}
              hint="Net revenue after returns and cancelled invoices."
            />
            <KpiCard
              label="Orders"
              value={fmt.number(data.total_orders)}
              delta={data.mom.orders_pct}
              icon={ShoppingCart}
              hint="Distinct invoices, excluding cancellations."
            />
            <KpiCard
              label="Avg order value"
              value={fmt.currencyPrecise(data.avg_order_value)}
              delta={data.mom.aov_pct}
              icon={TrendingUp}
              hint="Revenue divided by number of orders."
            />
            <KpiCard
              label="Active customers"
              value={fmt.number(data.total_customers)}
              delta={data.mom.customers_pct}
              icon={Users}
              hint="Identified customers with at least one purchase. Guest checkouts have no customer ID and are excluded."
            />
          </div>

          {/* Secondary metrics: real but not headline, so they get a quieter row. */}
          <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 rounded-card border border-navy-100 bg-surface px-card py-3">
            <span className="flex items-center gap-1.5 text-caption text-navy-500">
              <Undo2 size={13} className="text-navy-300" aria-hidden="true" />
              Return rate
              <span className="tabular font-semibold text-navy-900">
                {fmt.percent(data.return_rate_pct)}
              </span>
            </span>
            <span className="text-caption text-navy-500">
              Units sold{' '}
              <span className="tabular font-semibold text-navy-900">
                {fmt.numberCompact(data.total_units)}
              </span>
            </span>
            <span className="text-caption text-navy-500">
              Period{' '}
              <span className="font-medium text-navy-700">
                {fmt.date(data.period_start)} – {fmt.date(data.period_end)}
              </span>
            </span>
            <span className="ml-auto text-caption text-navy-400">
              MoM compares {data.mom.current_period}
              {data.mom.previous_period ? ` with ${data.mom.previous_period}` : ''}
            </span>
          </div>
        </>
      )}
    </AsyncBoundary>
  );
}

function RevenueTrend() {
  const [granularity, setGranularity] = useState<Granularity>('month');
  const query = useSalesTrend({ granularity });

  return (
    <Panel
      title="Revenue trend"
      subtitle={`Net revenue by ${granularity}, with order volume overlaid`}
      action={
        <SegmentedControl
          value={granularity}
          onChange={setGranularity}
          ariaLabel="Trend granularity"
          options={[
            { value: 'month', label: 'Monthly' },
            { value: 'week', label: 'Weekly' },
          ]}
        />
      }
    >
      <AsyncBoundary
        state={query}
        skeleton={<div className="h-[300px]"><SkeletonChart height={240} /></div>}
        isEmpty={(data) => data.points.length === 0}
        empty={
          <EmptyState
            compact
            title="No trend data"
            message="No sales fall within the selected period."
          />
        }
      >
        {(data) => (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart
              data={data.points.map((point) => ({
                ...point,
                label: fmt.periodLabel(point.period),
              }))}
              margin={{ top: 4, right: 8, left: 4, bottom: 0 }}
            >
              <defs>
                <linearGradient id="revenueFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--c-teal-500)" stopOpacity={0.28} />
                  <stop offset="100%" stopColor="var(--c-teal-500)" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--c-grid)" vertical={false} />
              <XAxis
                dataKey="label"
                tick={AXIS.tick}
                stroke={AXIS.stroke}
                // Weekly data has ~100 points; showing every label would be an
                // illegible smear, so Recharts thins them automatically.
                interval="preserveStartEnd"
                minTickGap={granularity === 'week' ? 40 : 12}
              />
              <YAxis
                tick={AXIS.tick}
                stroke={AXIS.stroke}
                tickFormatter={(value: number) => fmt.currencyCompact(value)}
                width={58}
              />
              <Tooltip
                content={
                  <ChartTooltip
                    formatter={(value, key) =>
                      key === 'revenue'
                        ? fmt.currency(Number(value))
                        : fmt.number(Number(value))
                    }
                  />
                }
                cursor={{ stroke: 'var(--c-navy-200)', strokeWidth: 1 }}
              />
              <Area
                type="monotone"
                dataKey="revenue"
                name="Revenue"
                stroke="var(--c-teal-500)"
                strokeWidth={2}
                fill="url(#revenueFill)"
                animationDuration={600}
                activeDot={{ r: 4, strokeWidth: 2, stroke: '#fff' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </AsyncBoundary>
    </Panel>
  );
}

function CountryBreakdown() {
  const query = useSalesByCountry({ limit: 10 });

  return (
    <Panel
      title="Revenue by country"
      subtitle="Top 10 markets by net revenue"
    >
      <AsyncBoundary
        state={query}
        skeleton={<div className="h-[340px]"><SkeletonChart height={280} /></div>}
        isEmpty={(data) => data.items.length === 0}
        empty={<EmptyState compact title="No country data" icon={Globe2} />}
      >
        {(data) => {
          const rows = data.items.map((item) => ({
            ...item,
            // Long names ("United Kingdom", "Republic of Ireland") would push the
            // plot area to nothing on a horizontal bar chart.
            short: item.country.length > 14 ? `${item.country.slice(0, 13)}…` : item.country,
          }));

          return (
            <>
              <ResponsiveContainer width="100%" height={340}>
                <BarChart
                  data={rows}
                  layout="vertical"
                  margin={{ top: 0, right: 16, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--c-grid)" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={AXIS.tick}
                    stroke={AXIS.stroke}
                    tickFormatter={(value: number) => fmt.currencyCompact(value)}
                  />
                  <YAxis
                    type="category"
                    dataKey="short"
                    tick={AXIS.tick}
                    stroke={AXIS.stroke}
                    width={96}
                  />
                  <Tooltip
                    content={
                      <ChartTooltip formatter={(value) => fmt.currency(Number(value))} />
                    }
                    cursor={{ fill: 'var(--c-navy-200)', fillOpacity: 0.25 }}
                  />
                  <Bar dataKey="revenue" name="Revenue" radius={[0, 3, 3, 0]} animationDuration={600}>
                    {rows.map((row, index) => (
                      <Cell
                        key={row.country_id}
                        // The UK dominates this dataset (~85% of revenue). Fading
                        // subsequent bars keeps the smaller markets legible
                        // instead of rendering ten identical teal bars.
                        fill={index === 0 ? 'var(--c-navy-700)' : 'var(--c-teal-500)'}
                        fillOpacity={index === 0 ? 1 : Math.max(0.4, 1 - index * 0.07)}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

              {rows[0] && (
                <p className="mt-3 border-t border-navy-100 pt-3 text-caption text-navy-500">
                  <span className="font-semibold text-navy-900">{rows[0].country}</span> accounts
                  for {fmt.percent(rows[0].revenue_share_pct)} of revenue across{' '}
                  {fmt.number(data.country_count)} markets.
                </p>
              )}
            </>
          );
        }}
      </AsyncBoundary>
    </Panel>
  );
}

export default function Overview() {
  return (
    <>
      <PageHeader
        title="Executive Overview"
        description="Trading performance across the Online Retail II dataset (Dec 2009 – Dec 2011)."
      />
      <div className="space-y-6">
        <KpiRow />
        <RevenueTrend />
        <CountryBreakdown />
      </div>
    </>
  );
}
