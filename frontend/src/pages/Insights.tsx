/**
 * Statistical Insights — hypothesis tests written for a reader, not a
 * statistician.
 *
 * The design principle: lead with the plain-language conclusion, then show the
 * chart that supports it, then the test statistics for anyone who wants to
 * check the work. A p-value with no interpretation is a number nobody acts on.
 *
 * Assumption checks are surfaced rather than hidden, because a significant
 * result from a violated assumption is worse than no result.
 */

import { AlertTriangle, CheckCircle2, FlaskConical, MinusCircle } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useInsights } from '@/hooks/useApi';
import { AsyncBoundary, EmptyState, SkeletonChart } from '@/components/ui/States';
import { Badge, InfoTip, PageHeader, Panel } from '@/components/ui/Primitives';
import * as fmt from '@/lib/format';
import type { GroupSummary, HypothesisTestResult } from '@/api/types';

const AXIS = { tick: { fill: 'var(--c-navy-400)', fontSize: 11 }, stroke: 'var(--c-navy-200)' } as const;

/** Group means with ±1 SE whiskers. Bars alone imply a precision the data
 *  doesn't have; the whiskers are what make "significantly higher" legible. */
function GroupChart({ groups, isCurrency }: { groups: GroupSummary[]; isCurrency: boolean }) {
  const rows = groups
    .filter((group) => typeof group.mean === 'number')
    .map((group) => {
      const n = typeof group.n === 'number' ? group.n : 0;
      const std = typeof group.std === 'number' ? group.std : 0;
      // Standard error, not standard deviation: the question is about the
      // precision of the mean, not the spread of the sample.
      const standardError = n > 1 ? std / Math.sqrt(n) : 0;
      return {
        name: group.name,
        mean: group.mean as number,
        error: standardError,
        n,
      };
    });

  if (rows.length === 0) {
    return <EmptyState compact title="No group summaries returned for this test" />;
  }

  const format = (value: number) => (isCurrency ? fmt.currency(value) : fmt.decimal(value, 2));

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, rows.length * 42)}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 24, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--c-grid)" horizontal={false} />
        <XAxis type="number" tick={AXIS.tick} stroke={AXIS.stroke} tickFormatter={format} />
        <YAxis type="category" dataKey="name" tick={AXIS.tick} stroke={AXIS.stroke} width={124} />
        <Tooltip
          cursor={{ fill: 'var(--c-navy-200)', fillOpacity: 0.25 }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const row = payload[0]?.payload as { name: string; mean: number; error: number; n: number };
            return (
              <div className="rounded-control border border-navy-200 bg-surface px-3 py-2 shadow-popover">
                <p className="text-caption font-semibold text-navy-900">{row.name}</p>
                <p className="tabular text-caption text-navy-600">mean {format(row.mean)}</p>
                <p className="tabular text-caption text-navy-500">
                  ±{format(row.error)} SE · n={fmt.number(row.n)}
                </p>
              </div>
            );
          }}
        />
        <Bar dataKey="mean" radius={[0, 3, 3, 0]} animationDuration={600}>
          {rows.map((row, index) => (
            <Cell key={row.name} fill="var(--c-teal-500)" fillOpacity={Math.max(0.45, 1 - index * 0.12)} />
          ))}
          <ErrorBar dataKey="error" width={4} strokeWidth={1.5} stroke="var(--c-navy-700)" direction="x" />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function TestCard({ test }: { test: HypothesisTestResult }) {
  // A test that couldn't compute is reported as such rather than rendered as a
  // non-significant result, which would be a different and false claim.
  const couldNotRun = test.p_value === null || test.statistic === null;

  const { Icon, tone, verdict } = couldNotRun
    ? { Icon: MinusCircle, tone: 'neutral' as const, verdict: 'Inconclusive' }
    : test.significant
      ? { Icon: CheckCircle2, tone: 'info' as const, verdict: 'Significant' }
      : { Icon: MinusCircle, tone: 'neutral' as const, verdict: 'Not significant' };

  // Monetary tests (AOV) format as currency; chi-square counts don't.
  const isCurrency = /aov|order value|revenue|monetary/i.test(test.question);

  const assumptionWarnings = Object.entries(test.assumptions).filter(([, value]) => {
    if (typeof value === 'boolean') return value === false;
    if (typeof value === 'string') return /violat|fail|not met|warn/i.test(value);
    return false;
  });

  return (
    <Panel>
      <div className="flex items-start gap-3">
        <div className={`mt-0.5 rounded-full p-1.5 ${test.significant && !couldNotRun ? 'bg-teal-50' : 'bg-navy-100'}`}>
          <Icon size={16} className={test.significant && !couldNotRun ? 'text-teal-600' : 'text-navy-400'} aria-hidden="true" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-heading-lg text-navy-900">{test.question}</h3>
            <Badge tone={tone}>{verdict}</Badge>
            <Badge>{test.test_type}</Badge>
          </div>

          {/* The headline: the backend writes this sentence so the wording stays
              consistent with the numbers that produced it. */}
          <p className="mt-2 text-body text-navy-700">{test.interpretation}</p>

          <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-1.5 text-caption">
            <div className="flex items-center gap-1.5">
              <dt className="text-navy-400">statistic</dt>
              <dd className="tabular font-medium text-navy-900">{fmt.decimal(test.statistic)}</dd>
            </div>
            <div className="flex items-center gap-1.5">
              <dt className="text-navy-400">p-value</dt>
              <dd className={`tabular font-semibold ${test.significant ? 'text-teal-700' : 'text-navy-700'}`}>
                {fmt.pValue(test.p_value)}
              </dd>
            </div>
            {test.degrees_of_freedom !== null && (
              <div className="flex items-center gap-1.5">
                <dt className="text-navy-400">df</dt>
                <dd className="tabular font-medium text-navy-900">{fmt.decimal(test.degrees_of_freedom, 0)}</dd>
              </div>
            )}
            {test.effect_size !== null && (
              <div className="flex items-center gap-1.5">
                <dt className="flex items-center gap-1 text-navy-400">
                  {test.effect_size_name ?? 'effect size'}
                  <InfoTip text="Effect size measures how large the difference is. Statistical significance only says a difference exists — with a large sample, a trivial gap can still be significant." />
                </dt>
                <dd className="tabular font-medium text-navy-900">
                  {fmt.decimal(test.effect_size, 3)}
                  {test.effect_size_label && (
                    <span className="ml-1 font-normal text-navy-400">({test.effect_size_label})</span>
                  )}
                </dd>
              </div>
            )}
            <div className="flex items-center gap-1.5">
              <dt className="text-navy-400">α</dt>
              <dd className="tabular font-medium text-navy-700">{test.alpha}</dd>
            </div>
          </dl>

          <p className="mt-2 text-caption italic text-navy-400">H₀: {test.null_hypothesis}</p>

          {assumptionWarnings.length > 0 && (
            <div className="mt-3 flex items-start gap-2 rounded-control border border-risk-mediumBorder bg-risk-mediumBg px-3 py-2">
              <AlertTriangle size={13} className="mt-0.5 shrink-0 text-risk-medium" aria-hidden="true" />
              <p className="text-caption text-navy-700">
                <span className="font-semibold">Assumption caveat: </span>
                {assumptionWarnings.map(([key]) => fmt.humanise(key)).join('; ')}. Interpret with
                care — the test's validity depends on these holding.
              </p>
            </div>
          )}
        </div>
      </div>

      {test.groups.length > 0 && (
        <div className="mt-4 border-t border-navy-100 pt-4">
          <GroupChart groups={test.groups} isCurrency={isCurrency} />
          <p className="mt-2 text-caption text-navy-400">
            Bars show group means; whiskers are ±1 standard error.
          </p>
        </div>
      )}
    </Panel>
  );
}

export default function Insights() {
  const query = useInsights();

  return (
    <>
      <PageHeader
        title="Statistical Insights"
        description="Hypothesis tests run against the warehouse, reported in plain language with the supporting numbers."
      />

      <AsyncBoundary
        state={query}
        skeleton={
          <div className="space-y-6">
            <SkeletonChart height={200} />
            <SkeletonChart height={200} />
          </div>
        }
        isEmpty={(data) => data.tests.length === 0 && data.unavailable.length === 0}
        empty={
          <EmptyState
            icon={FlaskConical}
            title="No tests could be run"
            message="Statistical tests need loaded sales data and computed RFM segments. Run the seed pipeline first."
          />
        }
      >
        {(data) => (
          <div className="space-y-6">
            <p className="text-caption text-navy-500">
              Tests evaluated at α = {data.alpha}. {fmt.number(data.tests.length)} test
              {data.tests.length === 1 ? '' : 's'} completed
              {data.unavailable.length > 0 && `, ${fmt.number(data.unavailable.length)} could not run`}.
            </p>

            {data.tests.map((test, index) => (
              <TestCard key={`${test.test_type}-${index}`} test={test} />
            ))}

            {/* Skipped tests are listed with their reason. Silently omitting them
                would leave the reader assuming the analysis was exhaustive. */}
            {data.unavailable.length > 0 && (
              <Panel title="Tests not run" subtitle="Reported rather than silently skipped">
                <ul className="space-y-2">
                  {data.unavailable.map((entry, index) => (
                    <li key={index} className="flex items-start gap-2 text-caption">
                      <MinusCircle size={13} className="mt-0.5 shrink-0 text-navy-300" aria-hidden="true" />
                      <span className="text-navy-600">
                        {Object.entries(entry).map(([key, value]) => (
                          <span key={key}>
                            <span className="font-medium text-navy-900">{fmt.humanise(key)}: </span>
                            {value}{' '}
                          </span>
                        ))}
                      </span>
                    </li>
                  ))}
                </ul>
              </Panel>
            )}
          </div>
        )}
      </AsyncBoundary>
    </>
  );
}
