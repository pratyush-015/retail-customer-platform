/**
 * Routing table and global providers.
 *
 * Pages are lazy-loaded so the Executive Overview — the landing page and the
 * one a reviewer sees first — isn't delayed by parsing the churn and stats
 * bundles it doesn't need yet.
 */

import { Suspense, lazy } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { Layout } from '@/components/Layout';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { ToastProvider } from '@/components/ui/Toast';
import { EmptyState, SkeletonCard, SkeletonChart } from '@/components/ui/States';
import { FileQuestion } from 'lucide-react';
import { Link } from 'react-router-dom';

const Overview = lazy(() => import('@/pages/Overview'));
const Segments = lazy(() => import('@/pages/Segments'));
const Churn = lazy(() => import('@/pages/Churn'));
const CustomerDetail = lazy(() => import('@/pages/CustomerDetail'));
const CustomerLookup = lazy(() => import('@/pages/CustomerLookup'));
const Insights = lazy(() => import('@/pages/Insights'));

/** Skeleton shown while a route chunk downloads. Mirrors the common page shape
 *  (KPI row + chart) so the transition doesn't flash a different layout. */
function RouteFallback() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }).map((_, index) => (
          <SkeletonCard key={index} />
        ))}
      </div>
      <SkeletonChart />
    </div>
  );
}

function NotFound() {
  return (
    <EmptyState
      icon={FileQuestion}
      title="Page not found"
      message="That route does not exist in this dashboard."
      action={
        <Link
          to="/"
          className="mt-1 rounded-control bg-navy-900 px-3 py-1.5 text-caption font-medium text-white transition-colors hover:bg-navy-700"
        >
          Back to overview
        </Link>
      }
    />
  );
}

export function App() {
  return (
    <ErrorBoundary>
      <ToastProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route
                path="/"
                element={
                  <Suspense fallback={<RouteFallback />}>
                    <Overview />
                  </Suspense>
                }
              />
              <Route
                path="/segments"
                element={
                  <Suspense fallback={<RouteFallback />}>
                    <Segments />
                  </Suspense>
                }
              />
              <Route
                path="/churn"
                element={
                  <Suspense fallback={<RouteFallback />}>
                    <Churn />
                  </Suspense>
                }
              />
              <Route
                path="/customers"
                element={
                  <Suspense fallback={<RouteFallback />}>
                    <CustomerLookup />
                  </Suspense>
                }
              />
              <Route
                path="/customers/:customerId"
                element={
                  <Suspense fallback={<RouteFallback />}>
                    <CustomerDetail />
                  </Suspense>
                }
              />
              <Route
                path="/insights"
                element={
                  <Suspense fallback={<RouteFallback />}>
                    <Insights />
                  </Suspense>
                }
              />
              {/* Legacy/typo path kept as a redirect so bookmarked links survive. */}
              <Route path="/customer/:customerId" element={<Navigate to="/customers" replace />} />
              <Route path="*" element={<NotFound />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ToastProvider>
    </ErrorBoundary>
  );
}
