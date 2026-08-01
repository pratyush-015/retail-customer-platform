/**
 * Customer Detail — the drill-down for a single customer.
 *
 * Three questions: what have they bought, where do they sit on RFM, and how
 * likely are they to leave. The 404 path matters here more than anywhere else,
 * since the id comes from a URL a user can type by hand.
 */

import { ArrowLeft, Package, Receipt, ShoppingBag } from 'lucide-react';
import { Link, useParams } from 'react-router-dom';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  PolarAngleAxis,
  RadialBar,
  RadialBarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useCustomer } from '@/hooks/useApi';
import { AsyncBoundary, EmptyState, SkeletonCard, SkeletonChart, SkeletonTable } from '@/components/ui/States';
import { Badge, ChartTooltip, healthTone, InfoTip, PageHeader, Panel, RiskBadge } from '@/components/ui/Primitives';
import * as fmt from '@/lib/format';
import type { CustomerProfile, RFMDetail } from '@/api/types';

const AXIS = { tick: { fill: 'var(--c-navy-400)', fontSize: 11 }, stroke: 'var(--c-navy-200)' } as const;

/** Radial gauge for churn probability. Colour crosses the same thresholds the
 *  backend uses for risk bands, so the arc and the badge always agree. */
function ChurnGauge({ score, band }: { score: number | null; band: string | null }) {
  if (score === null) {
    return (
      <EmptyState
        compact
        title="Not scored"
        message="This customer has no churn score yet. Run the scoring script to populate it."
      />
    );
  }

  const colour = score >= 0.7 ? 'var(--c-risk-high)' : score >= 0.4 ? 'var(--c-risk-medium)' : 'var(--c-risk-low)';

  return (
    <div className="relative">
      <ResponsiveContainer width="100%" height={200}>
        <RadialBarChart
          data={[{ value: score * 100, fill: colour }]}
          innerRadius="72%"
          outerRadius="100%"
          // Sweeping 180°→-180° draws a full ring clockwise from the top, which
          // reads as a gauge rather than a pie slice.
          startAngle={180}
          endAngle={-180}
        >
          <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
          <RadialBar dataKey="value" cornerRadius={6} background={{ fill: 'var(--c-navy-100)' }} animationDuration={800} />
        </RadialBarChart>
      </ResponsiveContainer>
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
        <span className="tabular text-display text-navy-900">{fmt.ratioAsPercent(score)}</span>
        <span className="mt-1"><RiskBadge band={band} /></span>
      </div>
    </div>
  );
}

/** R, F and M as 1–5 bars with their raw values beneath, so the score is always
 *  traceable to the number that produced it. */
function RFMBreakdown({ rfm }: { rfm: RFMDetail }) {
  const dimensions = [
    { label: 'Recency', score: rfm.r_score, raw: fmt.days(rfm.recency_days), hint: 'Days since last purchase. Lower is better, so a recent buyer scores 5.' },
    { label: 'Frequency', score: rfm.f_score, raw: `${fmt.number(rfm.frequency)} orders`, hint: 'Total orders placed. Higher scores 5.' },
    { label: 'Monetary', score: rfm.m_score, raw: fmt.currency(rfm.monetary), hint: 'Lifetime revenue. Higher scores 5.' },
  ];

  return (
    <div className="space-y-4">
      {dimensions.map((dimension) => (
        <div key={dimension.label}>
          <div className="flex items-baseline justify-between">
            <span className="label-micro flex items-center gap-1.5">
              {dimension.label}
              <InfoTip text={dimension.hint} />
            </span>
            <span className="tabular text-caption text-navy-500">{dimension.raw}</span>
          </div>
          <div className="mt-1.5 flex gap-1" role="img" aria-label={`${dimension.label} score ${dimension.score ?? 'unknown'} out of 5`}>
            {[1, 2, 3, 4, 5].map((step) => (
              <span
                key={step}
                className={`h-1.5 flex-1 rounded-full transition-colors duration-300 ${
                  dimension.score !== null && step <= dimension.score ? 'bg-teal-500' : 'bg-navy-100'
                }`}
              />
            ))}
          </div>
        </div>
      ))}

      <div className="flex items-center justify-between border-t border-navy-100 pt-3">
        <span className="label-micro">RFM score</span>
        <span className="tabular font-mono text-heading text-navy-900">{rfm.rfm_score ?? fmt.EM_DASH}</span>
      </div>
      {rfm.segment && (
        <div className="flex items-center justify-between">
          <span className="label-micro">Segment</span>
          <Badge tone={healthTone(rfm.health)}>{rfm.segment}</Badge>
        </div>
      )}
    </div>
  );
}

function PurchaseTimeline({ profile }: { profile: CustomerProfile }) {
  // Oldest-first: a purchase history reads left to right in time.
  const history = [...profile.order_history]
    .sort((a, b) => a.order_date.localeCompare(b.order_date))
    .map((entry) => ({ ...entry, label: fmt.date(entry.order_date) }));

  if (history.length === 0) {
    return <EmptyState compact icon={Receipt} title="No orders on record" />;
  }

  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={history} margin={{ top: 4, right: 8, left: 4, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--c-grid)" vertical={false} />
        <XAxis dataKey="label" tick={AXIS.tick} stroke={AXIS.stroke} interval="preserveStartEnd" minTickGap={28} />
        <YAxis tick={AXIS.tick} stroke={AXIS.stroke} tickFormatter={(value: number) => fmt.currencyCompact(value)} width={58} />
        <Tooltip
          cursor={{ fill: 'var(--c-navy-200)', fillOpacity: 0.25 }}
          content={<ChartTooltip formatter={(value) => fmt.currency(Number(value))} />}
        />
        <Bar dataKey="revenue" name="Order value" radius={[3, 3, 0, 0]} animationDuration={600}>
          {history.map((entry) => (
            <Cell
              key={entry.invoice_no}
              // Cancellations carry negative revenue; colouring them red stops
              // a downward bar being read as a small sale.
              fill={entry.is_cancelled ? 'var(--c-risk-high)' : 'var(--c-teal-500)'}
              fillOpacity={entry.is_cancelled ? 0.75 : 0.9}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export default function CustomerDetail() {
  const { customerId } = useParams<{ customerId: string }>();
  const query = useCustomer(customerId);

  return (
    <>
      <Link
        to="/customers"
        className="mb-3 inline-flex items-center gap-1.5 text-caption font-medium text-navy-500 transition-colors hover:text-navy-900"
      >
        <ArrowLeft size={14} aria-hidden="true" />
        Back to lookup
      </Link>

      <AsyncBoundary
        state={query}
        skeleton={
          <div className="space-y-6">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {Array.from({ length: 4 }).map((_, index) => <SkeletonCard key={index} />)}
            </div>
            <SkeletonChart height={220} />
            <SkeletonTable rows={5} columns={4} />
          </div>
        }
      >
        {(profile) => (
          <>
            <PageHeader
              title={`Customer ${profile.customer_id}`}
              description={[
                profile.country ?? 'Unknown country',
                profile.first_purchase_date ? `first purchase ${fmt.date(profile.first_purchase_date)}` : null,
                profile.tenure_days !== null ? `${fmt.days(profile.tenure_days)} tenure` : null,
              ]
                .filter(Boolean)
                .join(' · ')}
              action={profile.churn_risk_band ? <RiskBadge band={profile.churn_risk_band} /> : undefined}
            />

            <div className="space-y-6">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <div className="card">
                  <p className="label-micro">Lifetime revenue</p>
                  <p className="tabular mt-2 text-display text-navy-900">{fmt.currency(profile.total_revenue)}</p>
                  {profile.revenue_percentile !== null && (
                    <p className="mt-1 text-caption text-navy-400">
                      Top {fmt.percent(100 - profile.revenue_percentile, 0)} of customers
                    </p>
                  )}
                </div>
                <div className="card">
                  <p className="label-micro">Orders</p>
                  <p className="tabular mt-2 text-display text-navy-900">{fmt.number(profile.total_orders)}</p>
                  <p className="mt-1 text-caption text-navy-400">
                    {fmt.currency(profile.avg_order_value)} average
                  </p>
                </div>
                <div className="card">
                  <p className="label-micro">Purchase cadence</p>
                  <p className="tabular mt-2 text-display text-navy-900">
                    {fmt.days(profile.avg_days_between_orders)}
                  </p>
                  <p className="mt-1 text-caption text-navy-400">between orders on average</p>
                </div>
                <div className="card">
                  <p className="label-micro">Return rate</p>
                  <p className="tabular mt-2 text-display text-navy-900">
                    {fmt.ratioAsPercent(profile.return_rate)}
                  </p>
                  <p className="mt-1 text-caption text-navy-400">
                    {fmt.number(profile.distinct_products)} distinct products
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
                <Panel title="Churn probability" className="lg:col-span-1">
                  <ChurnGauge score={profile.churn_risk_score} band={profile.churn_risk_band} />
                  {profile.is_churned !== null && (
                    <p className="mt-2 text-center text-caption text-navy-500">
                      Currently classified as{' '}
                      <span className="font-semibold text-navy-900">
                        {profile.is_churned ? 'churned' : 'active'}
                      </span>{' '}
                      under the 90-day definition.
                    </p>
                  )}
                </Panel>

                <Panel title="RFM breakdown" className="lg:col-span-2">
                  {profile.rfm_computed ? (
                    <RFMBreakdown rfm={profile.rfm} />
                  ) : (
                    <EmptyState
                      compact
                      title="RFM not computed"
                      message="Run `python scripts/run_rfm.py` to score this customer."
                    />
                  )}
                </Panel>
              </div>

              <Panel
                title="Purchase history"
                subtitle={`${fmt.number(profile.order_history.length)} most recent orders — red bars are cancellations`}
              >
                <PurchaseTimeline profile={profile} />
              </Panel>

              <Panel title="Top products" subtitle="Highest-revenue items bought by this customer">
                {profile.top_products.length === 0 ? (
                  <EmptyState compact icon={ShoppingBag} title="No product history" />
                ) : (
                  <div className="-mx-card overflow-x-auto">
                    <table className="w-full min-w-[560px] border-collapse">
                      <thead>
                        <tr className="border-b border-navy-100">
                          <th scope="col" className="label-micro px-4 py-2.5 text-left">Product</th>
                          <th scope="col" className="label-micro px-4 py-2.5 text-left">Category</th>
                          <th scope="col" className="label-micro px-4 py-2.5 text-right">Quantity</th>
                          <th scope="col" className="label-micro px-4 py-2.5 text-right">Revenue</th>
                        </tr>
                      </thead>
                      <tbody>
                        {profile.top_products.map((product) => (
                          <tr key={product.stock_code} className="row-hover border-b border-navy-50 last:border-0">
                            <td className="px-4 py-2.5">
                              <div className="flex items-center gap-2">
                                <Package size={14} className="shrink-0 text-navy-300" aria-hidden="true" />
                                <div>
                                  <p className="text-body text-navy-900">
                                    {product.description ?? 'Unnamed product'}
                                  </p>
                                  <p className="font-mono text-micro text-navy-400">{product.stock_code}</p>
                                </div>
                              </div>
                            </td>
                            <td className="px-4 py-2.5 text-caption text-navy-500">
                              {product.category ?? fmt.EM_DASH}
                            </td>
                            <td className="tabular px-4 py-2.5 text-right text-body text-navy-700">
                              {fmt.number(product.quantity)}
                            </td>
                            <td className="tabular px-4 py-2.5 text-right text-body font-medium text-navy-900">
                              {fmt.currency(product.revenue)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </Panel>
            </div>
          </>
        )}
      </AsyncBoundary>
    </>
  );
}
