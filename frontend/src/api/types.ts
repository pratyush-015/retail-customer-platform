/**
 * TypeScript mirrors of the backend's Pydantic response models.
 *
 * Derived from `docs/api_contract.txt`, which is generated from the live
 * OpenAPI document (`python scripts/dump_api_contract.py`). Regenerate that
 * file and reconcile here after changing any response model.
 *
 * `| null` is load-bearing throughout. The warehouse leaves RFM and churn
 * columns NULL until the analytics pipeline runs, so "not yet computed" is a
 * real state the UI must render differently from zero.
 */

/** Shape every failure arrives in, produced by the backend's global handler. */
export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    status: number;
    details?: unknown;
  };
}

// ---------------------------------------------------------------- health

export interface ComponentHealth {
  name: string;
  status: 'ok' | 'degraded' | 'down' | string;
  detail?: string | null;
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'down' | string;
  version: string;
  environment: string;
  components: ComponentHealth[];
}

// ------------------------------------------------------------------ kpis

export interface MoMChange {
  current_period: string;
  previous_period: string | null;
  revenue_pct: number | null;
  orders_pct: number | null;
  customers_pct: number | null;
  aov_pct: number | null;
}

export interface KPISummary {
  total_revenue: number;
  total_orders: number;
  total_customers: number;
  avg_order_value: number;
  total_units: number;
  return_rate_pct: number;
  period_start: string | null;
  period_end: string | null;
  mom: MoMChange;
}

// ----------------------------------------------------------------- sales

export type Granularity = 'month' | 'week';

export interface TrendPoint {
  period: string;
  period_start: string;
  revenue: number;
  orders: number;
  customers: number;
  units: number;
}

export interface SalesTrend {
  granularity: Granularity | string;
  points: TrendPoint[];
  total_revenue: number;
}

export interface CountryRevenue {
  country_id: number;
  country: string;
  region: string | null;
  revenue: number;
  orders: number;
  customers: number;
  avg_order_value: number;
  revenue_share_pct: number;
}

export interface SalesByCountry {
  items: CountryRevenue[];
  total_revenue: number;
  country_count: number;
}

// ------------------------------------------------------------- customers

/** Mirrors `SEGMENT_HEALTH` in `services/customer.py`. Colour from this, never
 *  from the segment name — the backend owns that mapping. */
export type SegmentHealth = 'strong' | 'stable' | 'watch' | 'critical' | string;

export interface SegmentStats {
  segment: string;
  health: SegmentHealth;
  customer_count: number;
  customer_pct: number;
  revenue: number;
  revenue_pct: number;
  orders: number;
  avg_revenue_per_customer: number;
  avg_recency_days: number;
  avg_frequency: number;
  avg_monetary: number;
  avg_churn_risk: number | null;
}

export interface SegmentDistribution {
  segments: SegmentStats[];
  total_customers: number;
  /** Revenue attributable to identified customers; guest invoices carry no
   *  customer id, so this is legitimately below `/kpis/summary` revenue. */
  attributed_revenue: number;
  /** False when the analytics pipeline has not run yet. */
  computed: boolean;
}

export interface RFMDetail {
  recency_days: number | null;
  frequency: number | null;
  monetary: number | null;
  r_score: number | null;
  f_score: number | null;
  m_score: number | null;
  rfm_score: string | null;
  segment: string | null;
  health: SegmentHealth | null;
}

export interface OrderHistoryEntry {
  invoice_no: string;
  order_date: string;
  revenue: number;
  items: number;
  distinct_products: number;
  is_cancelled: boolean;
}

export interface TopProduct {
  stock_code: string;
  description: string | null;
  category: string | null;
  revenue: number;
  quantity: number;
}

export type RiskBand = 'high' | 'medium' | 'low';

export interface CustomerProfile {
  customer_id: number;
  country: string | null;
  first_purchase_date: string | null;
  last_purchase_date: string | null;
  tenure_days: number | null;
  rfm: RFMDetail;
  rfm_computed: boolean;
  total_orders: number | null;
  total_revenue: number;
  total_items: number | null;
  avg_order_value: number | null;
  distinct_products: number | null;
  return_rate: number | null;
  avg_days_between_orders: number | null;
  revenue_percentile: number | null;
  churn_risk_score: number | null;
  churn_risk_band: RiskBand | null;
  is_churned: boolean | null;
  order_history: OrderHistoryEntry[];
  top_products: TopProduct[];
}

// ----------------------------------------------------------------- churn

export interface AtRiskCustomer {
  customer_id: number;
  country: string | null;
  segment: string | null;
  recency_days: number | null;
  frequency: number | null;
  total_revenue: number;
  avg_order_value: number | null;
  last_purchase_date: string | null;
  churn_risk_score: number;
  risk_band: RiskBand;
  /** churn_risk_score × total_revenue — the ranking key for this list. */
  expected_revenue_loss: number;
}

export interface AtRiskResponse {
  items: AtRiskCustomer[];
  total: number;
  limit: number;
  offset: number;
  threshold: number;
  /** Worst case: full revenue of every customer above the threshold. */
  revenue_at_risk: number;
  /** Probability-weighted, and the more defensible headline number. */
  expected_revenue_loss: number;
  scored_customers: number;
}

export interface ConfusionMatrix {
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
  true_positives: number;
}

export interface FeatureImportance {
  feature: string;
  coefficient: number;
  abs_coefficient: number;
  odds_ratio: number;
  /** 'increases' | 'decreases' — precomputed so the UI doesn't infer meaning
   *  from the coefficient's sign. */
  direction: string;
  description?: string | null;
}

export interface ROCPoint {
  fpr: number;
  tpr: number;
}

/** Precision/recall as the decision threshold moves. Lets the churn page
 *  justify its default threshold rather than presenting 0.5 as a given. */
export interface ThresholdRow {
  threshold: number;
  customers_flagged: number;
  pct_flagged: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface ModelPerformance {
  model_version: string;
  model_type: string;
  trained_at: string;
  churn_definition: string;
  features: string[];
  roc_auc: number;
  pr_auc: number | null;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  brier_score: number | null;
  /** Majority-class accuracy. Shown beside accuracy so a 0.87 that merely
   *  matches the base rate cannot masquerade as a good model. */
  baseline_accuracy: number | null;
  confusion_matrix: ConfusionMatrix;
  feature_importance: FeatureImportance[];
  roc_curve: ROCPoint[];
  threshold_analysis: ThresholdRow[];
  cross_validation?: Record<string, unknown> | null;
  dataset?: Record<string, unknown> | null;
}

/** Only the three RFM values are required; the backend derives the remaining
 *  four behavioural features and reports which in `derived_features`. */
export interface ChurnPredictionRequest {
  recency: number;
  frequency: number;
  monetary: number;
  avg_order_value?: number | null;
  tenure_days?: number | null;
  avg_days_between_orders?: number | null;
  distinct_products?: number | null;
  total_items?: number | null;
  return_rate?: number | null;
}

export interface ChurnPredictionResponse {
  churn_probability: number;
  risk_band: RiskBand;
  model_version: string;
  features_used: Record<string, number>;
  derived_features: string[];
  interpretation: string;
}

// ------------------------------------------------------------------ stats

export type TestType = 'ttest' | 'anova' | 'chi2';

export interface GroupSummary {
  name: string;
  n?: number | null;
  mean?: number | null;
  std?: number | null;
  median?: number | null;
  [key: string]: unknown;
}

export interface HypothesisTestResult {
  test_type: string;
  /** The business question in plain language — the page headline. */
  question: string;
  null_hypothesis: string;
  /** Null when the test could not be computed. */
  statistic: number | null;
  p_value: number | null;
  significant: boolean;
  alpha: number;
  interpretation: string;
  effect_size: number | null;
  effect_size_name: string | null;
  effect_size_label: string | null;
  degrees_of_freedom: number | null;
  groups: GroupSummary[];
  /** Assumption checks (normality, equal variance, expected cell counts) so
   *  caveats travel with the result instead of being dropped. */
  assumptions: Record<string, unknown>;
  details: Record<string, unknown>;
}

export interface InsightsResponse {
  tests: HypothesisTestResult[];
  /** Tests that could not run (e.g. too few observations in a group) are
   *  reported rather than silently dropped. `list[dict[str, str]]` server-side. */
  unavailable: Record<string, string>[];
  alpha: number;
}
