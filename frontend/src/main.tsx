import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './App';
import { ApiError } from './api/client';
import './index.css';

/**
 * Retry policy is deliberate rather than default.
 *
 * React Query retries 3× for everything out of the box, which is wrong here:
 * a 404 for a nonexistent customer will never become a 200, and retrying it
 * three times just delays the error state by a couple of seconds. Only
 * genuinely transient failures are retried.
 */
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (failureCount, error) => {
        if (error instanceof ApiError && !error.isRetryable) return false;
        return failureCount < 2;
      },
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 5000),
      // The warehouse is refreshed by a batch pipeline, so refetching whenever
      // the window regains focus would spend queries to redraw identical charts.
      refetchOnWindowFocus: false,
      gcTime: 10 * 60 * 1000,
    },
  },
});

const rootElement = document.getElementById('root');
if (!rootElement) throw new Error('Root element #root not found in index.html');

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
);
