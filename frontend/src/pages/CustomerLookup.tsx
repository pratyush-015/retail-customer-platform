/**
 * Customer Lookup — the entry point to the drill-down, plus a what-if churn
 * calculator.
 *
 * The calculator is where `POST /api/predict/churn` earns its place: it scores
 * hypothetical RFM values rather than a stored customer, which is what makes
 * the model interactive instead of a static report. It also exercises the 422
 * path, since the form deliberately allows submitting values the backend will
 * reject.
 */

import { Calculator, Search, UserSearch } from 'lucide-react';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePredictChurn } from '@/hooks/useApi';
import { useToast } from '@/components/ui/Toast';
import { ErrorState } from '@/components/ui/States';
import { Badge, InfoTip, PageHeader, Panel, riskTone } from '@/components/ui/Primitives';
import * as fmt from '@/lib/format';

/** Client-side guard mirroring the Pydantic constraints. Not a replacement for
 *  them — the server still validates — but it turns an avoidable round trip
 *  into instant feedback. */
function validate(fields: { recency: string; frequency: string; monetary: string }): string | null {
  const recency = Number(fields.recency);
  const frequency = Number(fields.frequency);
  const monetary = Number(fields.monetary);

  if (!Number.isFinite(recency) || recency < 0) return 'Recency must be zero or more days.';
  if (!Number.isInteger(frequency) || frequency < 1) return 'Frequency must be at least one order.';
  if (!Number.isFinite(monetary) || monetary <= 0) return 'Monetary value must be greater than zero.';
  return null;
}

function ChurnCalculator() {
  const [fields, setFields] = useState({ recency: '120', frequency: '3', monetary: '450' });
  const [localError, setLocalError] = useState<string | null>(null);
  const predict = usePredictChurn();
  const { toast } = useToast();

  const update = (key: keyof typeof fields) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setFields((current) => ({ ...current, [key]: event.target.value }));
    setLocalError(null);
  };

  const onSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const message = validate(fields);
    if (message) {
      setLocalError(message);
      return;
    }

    predict.mutate(
      {
        recency: Number(fields.recency),
        frequency: Number(fields.frequency),
        monetary: Number(fields.monetary),
      },
      {
        onSuccess: (result) => toast(`Scored: ${fmt.ratioAsPercent(result.churn_probability)} churn probability`, 'success'),
        onError: () => toast('Prediction failed — see the panel for details', 'error'),
      },
    );
  };

  const INPUTS = [
    { key: 'recency' as const, label: 'Recency (days)', hint: 'Days since the last purchase.', min: 0, step: 1 },
    { key: 'frequency' as const, label: 'Frequency (orders)', hint: 'Total distinct orders placed.', min: 1, step: 1 },
    { key: 'monetary' as const, label: 'Monetary (£)', hint: 'Lifetime revenue from this customer.', min: 0, step: 10 },
  ];

  return (
    <Panel
      title="Churn what-if calculator"
      subtitle="Score a hypothetical customer against the trained model"
    >
      <form onSubmit={onSubmit} className="space-y-4">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {INPUTS.map((input) => (
            <div key={input.key}>
              <label htmlFor={input.key} className="label-micro flex items-center gap-1.5">
                {input.label}
                <InfoTip text={input.hint} />
              </label>
              <input
                id={input.key}
                type="number"
                min={input.min}
                step={input.step}
                value={fields[input.key]}
                onChange={update(input.key)}
                className="mt-1.5 w-full rounded-control border border-navy-200 bg-surface px-3 py-2 text-body text-navy-900 transition-colors focus:border-teal-500 focus:outline-none focus:ring-2 focus:ring-teal-500/20"
              />
            </div>
          ))}
        </div>

        {localError && (
          <p role="alert" className="text-caption font-medium text-risk-high">
            {localError}
          </p>
        )}

        <button
          type="submit"
          disabled={predict.isPending}
          className="inline-flex items-center gap-1.5 rounded-control bg-navy-900 px-4 py-2 text-body font-medium text-white transition-colors hover:bg-navy-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <Calculator size={15} aria-hidden="true" />
          {predict.isPending ? 'Scoring…' : 'Predict churn risk'}
        </button>
      </form>

      {/* The mutation's own error state: a 503 here means the model artifact is
          missing, which the ErrorState copy already explains. */}
      {predict.isError && (
        <div className="mt-4">
          <ErrorState compact error={predict.error} onRetry={() => predict.reset()} />
        </div>
      )}

      {predict.data && !predict.isError && (
        <div className="mt-4 animate-fade-in-up rounded-card border border-navy-100 bg-surface-sunken p-4">
          <div className="flex flex-wrap items-baseline gap-3">
            <span className="tabular text-display text-navy-900">
              {fmt.ratioAsPercent(predict.data.churn_probability)}
            </span>
            <Badge tone={riskTone(predict.data.risk_band)}>{predict.data.risk_band} risk</Badge>
            <span className="ml-auto text-micro text-navy-400">model v{predict.data.model_version}</span>
          </div>
          <p className="mt-2 text-body text-navy-600">{predict.data.interpretation}</p>
          {predict.data.derived_features.length > 0 && (
            <p className="mt-2 text-caption text-navy-400">
              Derived from the values supplied: {predict.data.derived_features.map(fmt.humanise).join(', ')}.
            </p>
          )}
        </div>
      )}
    </Panel>
  );
}

export default function CustomerLookup() {
  const [customerId, setCustomerId] = useState('');
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const onSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = customerId.trim();

    // Caught here rather than at the API: a non-numeric id can't identify a
    // customer, so there's nothing to ask the server about.
    if (!/^\d+$/.test(trimmed)) {
      setError('Customer IDs are numeric — for example 12748.');
      return;
    }
    setError(null);
    navigate(`/customers/${trimmed}`);
  };

  return (
    <>
      <PageHeader
        title="Customer Lookup"
        description="Open an individual customer profile, or score a hypothetical one against the churn model."
      />

      <div className="space-y-6">
        <Panel title="Find a customer" subtitle="Enter a numeric customer ID from the dataset">
          <form onSubmit={onSubmit} className="flex flex-wrap items-start gap-2">
            <div className="min-w-[200px] flex-1">
              <div className="relative">
                <Search
                  size={15}
                  className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-navy-300"
                  aria-hidden="true"
                />
                <input
                  type="text"
                  inputMode="numeric"
                  value={customerId}
                  onChange={(event) => {
                    setCustomerId(event.target.value);
                    setError(null);
                  }}
                  placeholder="e.g. 12748"
                  aria-label="Customer ID"
                  aria-invalid={Boolean(error)}
                  className="w-full rounded-control border border-navy-200 bg-surface py-2 pl-9 pr-3 text-body text-navy-900 transition-colors focus:border-teal-500 focus:outline-none focus:ring-2 focus:ring-teal-500/20"
                />
              </div>
              {error && (
                <p role="alert" className="mt-1.5 text-caption font-medium text-risk-high">
                  {error}
                </p>
              )}
            </div>
            <button
              type="submit"
              className="inline-flex items-center gap-1.5 rounded-control bg-navy-900 px-4 py-2 text-body font-medium text-white transition-colors hover:bg-navy-700"
            >
              <UserSearch size={15} aria-hidden="true" />
              Open profile
            </button>
          </form>
          <p className="mt-3 text-caption text-navy-400">
            Customers can also be reached by clicking any row on the Churn Risk page.
          </p>
        </Panel>

        <ChurnCalculator />
      </div>
    </>
  );
}
