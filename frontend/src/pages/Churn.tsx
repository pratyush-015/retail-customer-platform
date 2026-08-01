/**
 * Churn Risk — who is leaving, what it costs, and whether the model can be
 * trusted.
 *
 * The two halves are queried independently on purpose. `/churn/at-risk` reads
 * stored scores while `/churn/model-performance` reads the model artifact, so
 * a missing artifact must not blank the customer list. Two boundaries, two
 * failure modes.
 */

import { AlertTriangle, ArrowDown, ArrowUp, ShieldCheck, SlidersHorizontal, Users } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useAtRisk, useModelPerformance } from '@/hooks/useApi';
import { useDebouncedValue } from '@/hooks/useDebouncedValue';
import { useSortable } from '@/hooks/useSortable';
import { AsyncBoundary, EmptyState, SkeletonCard, SkeletonChart, SkeletonTable } from '@/components/ui/States';
import { Badge, InfoTip, PageHeader, Panel, RiskBadge } from '@/components/ui/Primitives';
import * as fmt from '@/lib/format';
import type { AtRiskCustomer, ConfusionMatrix, ModelPerformance } from '@/api/types';

const AXIS = { tick: { fill: 'var(--c-navy-400)', fontSize: 11 }, stroke: 'var(--c-navy-200)' } as const;

// --------------------------------------------------------- at-risk table

type SortKey = 'customer_id' | 'churn_risk_score' | 'total_revenue' | 'expected_revenue_loss' | 'recency_days' | 'frequency';

const COLUMNS: { key: SortKey; label: string; numeric: boolean; hint?: string }[] = [
  { key: 'customer_id', label: 'Customer', numeric: false },
  { key: 'churn_risk_score', label: 'Risk', numeric: true },
  { key: 'recency_days', label: 'Recency', numeric: true },
  { key: 'frequency', label: 'Orders', numeric: true },
  { key: 'total_revenue', label: 'Lifetime revenue', numeric: true },
  {
    key: 'expected_revenue_loss',
    label: 'Expected loss',
    numeric: true,
    hint: 'Churn probability × lifetime revenue. Ranks customers by money at stake rather than risk alone.',
  },
];

function AtRiskTable({ rows }: { rows: AtRiskCustomer[] }) {
  const accessors = useMemo(
    () => ({
      customer_id: (row: AtRiskCustomer) => row.customer_id,
      churn_risk_score: (row: AtRiskCustomer) => row.churn_risk_score,
      recency_days: (row: AtRiskCustomer) => row.recency_days,
      frequency: (row: AtRiskCustomer) => row.frequency,
      total_revenue: (row: AtRiskCustomer) => row.total_revenue,
      expected_revenue_loss: (row: AtRiskCustomer) => row.expected_revenue_loss,
    }),
    [],
  );

  const { sorted, sort, toggle } = useSortable<AtRiskCustomer, SortKey>(rows, accessors, {
    key: 'expected_revenue_loss',
    direction: 'desc',
  });

  return (
    <div className="-mx-card overflow-x-auto">
      <table className="w-full min-w-[760px] border-collapse">
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
                  className={`th-sortable label-micro px-4 py-2.5 ${column.numeric ? 'text-right' : 'text-left'} ${isActive ? 'text-navy-700' : ''}`}
                >
                  <span className={`inline-flex items-center gap-1 ${column.numeric ? 'flex-row-reverse' : ''}`}>
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
            <tr key={row.customer_id} className="row-hover border-b border-navy-50 last:border-0">
              <td className="px-4 py-2.5">
                <Link
                  to={`/customers/${row.customer_id}`}
                  className="text-body font-medium text-teal-700 hover:underline"
                >
                  {row.customer_id}
                </Link>
                <div className="mt-0.5 flex items-center gap-1.5 text-caption text-navy-400">
                  {row.country ?? 'Unknown'}
                  {row.segment && <span className="text-navy-300">·</span>}
                  {row.segment}
                </div>
              </td>
              <td className="px-4 py-2.5 text-right">
                <div className="flex items-center justify-end gap-2">
                  <span className="tabular text-body font-semibold text-navy-900">
                    {fmt.ratioAsPercent(row.churn_risk_score)}
                  </span>
                  <RiskBadge band={row.risk_band} />
                </div>
              </td>
              <td className="tabular px-4 py-2.5 text-right text-body text-navy-700">
                {fmt.days(row.recency_days)}
              </td>
              <td className="tabular px-4 py-2.5 text-right text-body text-navy-700">
                {fmt.number(row.frequency)}
              </td>
              <td className="tabular px-4 py-2.5 text-right text-body text-navy-700">
                {fmt.currency(row.total_revenue)}
              </td>
              <td className="tabular px-4 py-2.5 text-right text-body font-semibold text-risk-high">
                {fmt.currency(row.expected_revenue_loss)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ------------------------------------------------------ model performance

/** 2×2 grid. Colour encodes correct (teal) vs incorrect (red) rather than
 *  magnitude, so a glance tells you where the errors sit. */
function ConfusionGrid({ matrix }: { matrix: ConfusionMatrix }) {
  const cells = [
    { label: 'True negatives', value: matrix.true_negatives, good: true, note: 'Retained, predicted retained' },
    { label: 'False positives', value: matrix.false_positives, good: false, note: 'Retained, predicted churn' },
    { label: 'False negatives', value: matrix.false_negatives, good: false, note: 'Churned, predicted retained' },
    { label: 'True positives', value: matrix.true_positives, good: true, note: 'Churned, predicted churn' },
  ];

  return (
    <div className="grid grid-cols-2 gap-2">
      {cells.map((cell) => (
        <div
          key={cell.label}
          className={`rounded-control border p-3 ${cell.good ? 'border-risk-lowBorder bg-risk-lowBg' : 'border-risk-highBorder bg-risk-highBg'}`}
        >
          <p className="label-micro">{cell.label}</p>
          <p className="tabular mt-1 text-heading-lg text-navy-900">{fmt.number(cell.value)}</p>
          <p className="mt-0.5 text-micro text-navy-500">{cell.note}</p>
        </div>
      ))}
    </div>
  );
}

function ModelPanel({ model }: { model: ModelPerformance }) {
  // Ranked by |coefficient| server-side; reversed here because Recharts draws
  // horizontal bars bottom-up and the strongest driver belongs at the top.
  const importance = [...model.feature_importance]
    .slice(0, 8)
    .reverse()
    .map((item) => ({
      ...item,
      label: fmt.humanise(item.feature),
    }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {[
          { label: 'ROC-AUC', value: fmt.decimal(model.roc_auc), hint: 'Probability the model ranks a churner above a non-churner. 0.5 is a coin flip.' },
          { label: 'Precision', value: fmt.decimal(model.precision), hint: 'Of customers flagged as at-risk, the share that actually churned.' },
          { label: 'Recall', value: fmt.decimal(model.recall), hint: 'Of customers who actually churned, the share the model caught.' },
          { label: 'F1', value: fmt.decimal(model.f1), hint: 'Harmonic mean of precision and recall.' },
        ].map((metric) => (
          <div key={metric.label} className="card">
            <div className="flex items-center gap-1.5">
              <p className="label-micro">{metric.label}</p>
              <InfoTip text={metric.hint} />
            </div>
            <p className="tabular mt-2 text-display text-navy-900">{metric.value}</p>
          </div>
        ))}
      </div>

      {/* Accuracy alone flatters an imbalanced problem; the baseline is shown
          beside it so the comparison is unavoidable. */}
      {model.baseline_accuracy !== null && (
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-card border border-navy-100 bg-surface px-card py-3 text-caption">
          <span className="text-navy-500">
            Accuracy <span className="tabular font-semibold text-navy-900">{fmt.decimal(model.accuracy)}</span>
          </span>
          <span className="text-navy-500">
            Majority-class baseline{' '}
            <span className="tabular font-semibold text-navy-700">{fmt.decimal(model.baseline_accuracy)}</span>
          </span>
          <Badge tone={model.accuracy > model.baseline_accuracy ? 'low' : 'high'}>
            {model.accuracy > model.baseline_accuracy ? 'beats baseline' : 'at baseline'}
          </Badge>
          <span className="ml-auto text-navy-400">
            {model.model_type} · v{model.model_version} · trained {fmt.date(model.trained_at)}
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Panel title="ROC curve" subtitle={`AUC ${fmt.decimal(model.roc_auc)} — true positive rate against false positive rate`}>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={model.roc_curve} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="rocFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--c-teal-500)" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="var(--c-teal-500)" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--c-grid)" />
              <XAxis
                dataKey="fpr"
                type="number"
                domain={[0, 1]}
                tick={AXIS.tick}
                stroke={AXIS.stroke}
                tickFormatter={(value: number) => value.toFixed(1)}
                label={{ value: 'False positive rate', position: 'insideBottom', offset: -2, fill: 'var(--c-navy-400)', fontSize: 10 }}
              />
              <YAxis
                dataKey="tpr"
                type="number"
                domain={[0, 1]}
                tick={AXIS.tick}
                stroke={AXIS.stroke}
                tickFormatter={(value: number) => value.toFixed(1)}
                width={34}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const point = payload[0]?.payload as { fpr: number; tpr: number };
                  return (
                    <div className="rounded-control border border-navy-200 bg-surface px-3 py-2 shadow-popover">
                      <p className="tabular text-caption text-navy-700">TPR {fmt.decimal(point.tpr, 2)}</p>
                      <p className="tabular text-caption text-navy-500">FPR {fmt.decimal(point.fpr, 2)}</p>
                    </div>
                  );
                }}
              />
              {/* The chance diagonal: without it an AUC curve has no visual reference. */}
              <ReferenceLine
                segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                stroke="var(--c-navy-300)"
                strokeDasharray="4 4"
              />
              <Area type="monotone" dataKey="tpr" stroke="var(--c-teal-500)" strokeWidth={2} fill="url(#rocFill)" animationDuration={700} />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Confusion matrix" subtitle="Held-out test set at the default 0.5 threshold">
          <ConfusionGrid matrix={model.confusion_matrix} />
          <p className="mt-3 text-caption text-navy-500">
            Churn is defined as <span className="font-medium text-navy-700">{model.churn_definition}</span>.
          </p>
        </Panel>
      </div>

      <Panel
        title="Feature importance"
        subtitle="Logistic regression coefficients on standardised features — bar direction shows whether the feature raises or lowers churn risk"
      >
        <ResponsiveContainer width="100%" height={Math.max(220, importance.length * 34)}>
          <BarChart data={importance} layout="vertical" margin={{ top: 0, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--c-grid)" horizontal={false} />
            <XAxis type="number" tick={AXIS.tick} stroke={AXIS.stroke} tickFormatter={(value: number) => value.toFixed(2)} />
            <YAxis type="category" dataKey="label" tick={AXIS.tick} stroke={AXIS.stroke} width={132} />
            <Tooltip
              cursor={{ fill: 'var(--c-navy-200)', fillOpacity: 0.25 }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const row = payload[0]?.payload as { label: string; coefficient: number; odds_ratio: number; direction: string };
                return (
                  <div className="max-w-xs rounded-control border border-navy-200 bg-surface px-3 py-2 shadow-popover">
                    <p className="text-caption font-semibold text-navy-900">{row.label}</p>
                    <p className="tabular text-caption text-navy-600">coef {fmt.decimal(row.coefficient)}</p>
                    <p className="tabular text-caption text-navy-600">odds ratio {fmt.decimal(row.odds_ratio, 2)}</p>
                    <p className="mt-1 text-micro text-navy-500">{row.direction} churn risk</p>
                  </div>
                );
              }}
            />
            <ReferenceLine x={0} stroke="var(--c-navy-300)" />
            <Bar dataKey="coefficient" radius={[0, 3, 3, 0]} animationDuration={600}>
              {importance.map((item) => (
                <Cell
                  key={item.feature}
                  fill={item.coefficient >= 0 ? 'var(--c-risk-high)' : 'var(--c-risk-low)'}
                  fillOpacity={0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="mt-3 border-t border-navy-100 pt-3 text-caption text-navy-500">
          Red bars increase churn probability, green bars decrease it. Coefficients are on standardised
          features, so magnitudes are directly comparable.
        </p>
      </Panel>
    </div>
  );
}

// ------------------------------------------------------------------ page

export default function Churn() {
  const [threshold, setThreshold] = useState(0.5);
  // The slider drives the thumb immediately; the query waits for the pause.
  const debouncedThreshold = useDebouncedValue(threshold, 350);

  const atRiskQuery = useAtRisk({ threshold: debouncedThreshold, limit: 100 });
  const modelQuery = useModelPerformance();

  return (
    <>
      <PageHeader
        title="Churn Risk"
        description="Customers predicted to lapse, ranked by the revenue their departure would cost."
      />

      <div className="space-y-6">
        <AsyncBoundary
          state={atRiskQuery}
          skeleton={
            <div className="space-y-6">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                {Array.from({ length: 3 }).map((_, index) => <SkeletonCard key={index} />)}
              </div>
              <SkeletonTable rows={8} columns={6} />
            </div>
          }
          isEmpty={(data) => data.scored_customers === 0}
          empty={
            <EmptyState
              icon={ShieldCheck}
              title="No churn scores available"
              message="Customers have not been scored yet. Run `python scripts/score_customers.py` after training the model."
            />
          }
        >
          {(data) => (
            <>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <div className="card border-risk-highBorder bg-risk-highBg">
                  <div className="flex items-center gap-1.5">
                    <p className="label-micro">Expected revenue loss</p>
                    <InfoTip text="Sum of (churn probability × lifetime revenue) across flagged customers. The probability-weighted view." />
                  </div>
                  <p className="tabular mt-2 text-display text-risk-high">
                    {fmt.currencyCompact(data.expected_revenue_loss)}
                  </p>
                  <p className="mt-1 text-caption text-navy-500">
                    of {fmt.currencyCompact(data.revenue_at_risk)} total revenue at risk
                  </p>
                </div>
                <div className="card">
                  <p className="label-micro">Customers flagged</p>
                  <p className="tabular mt-2 text-display text-navy-900">{fmt.number(data.total)}</p>
                  <p className="mt-1 text-caption text-navy-500">
                    of {fmt.number(data.scored_customers)} scored ({fmt.ratioAsPercent(data.scored_customers ? data.total / data.scored_customers : 0)})
                  </p>
                </div>
                <div className="card">
                  <p className="label-micro">Avg revenue per flagged customer</p>
                  <p className="tabular mt-2 text-display text-navy-900">
                    {fmt.currency(data.total ? data.revenue_at_risk / data.total : 0)}
                  </p>
                  <p className="mt-1 text-caption text-navy-500">Lifetime value basis</p>
                </div>
              </div>

              <Panel
                title="At-risk customers"
                subtitle={`${fmt.number(data.items.length)} shown of ${fmt.number(data.total)} above the threshold`}
                action={
                  <div className="flex items-center gap-2">
                    <SlidersHorizontal size={14} className="text-navy-400" aria-hidden="true" />
                    <label htmlFor="threshold" className="label-micro">
                      Risk ≥ {fmt.ratioAsPercent(threshold, 0)}
                    </label>
                    <input
                      id="threshold"
                      type="range"
                      min={0}
                      max={0.95}
                      step={0.05}
                      value={threshold}
                      onChange={(event) => setThreshold(Number(event.target.value))}
                      className="h-1 w-32 cursor-pointer appearance-none rounded-full bg-navy-200 accent-teal-600"
                      aria-valuetext={`${Math.round(threshold * 100)} percent`}
                    />
                  </div>
                }
              >
                {data.items.length === 0 ? (
                  <EmptyState
                    compact
                    icon={Users}
                    title="No customers match this filter"
                    message={`No scored customer has a churn probability of ${fmt.ratioAsPercent(threshold, 0)} or higher. Lower the threshold to widen the net.`}
                  />
                ) : (
                  <AtRiskTable rows={data.items} />
                )}
              </Panel>
            </>
          )}
        </AsyncBoundary>

        {/* Separate boundary: a missing model artifact returns 503 here, and the
            at-risk list above must survive that. */}
        <AsyncBoundary
          state={modelQuery}
          skeleton={<SkeletonChart height={260} />}
          empty={<EmptyState icon={AlertTriangle} title="Model performance unavailable" />}
        >
          {(model) => <ModelPanel model={model} />}
        </AsyncBoundary>
      </div>
    </>
  );
}
