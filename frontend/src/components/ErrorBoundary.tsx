/**
 * Class error boundary — the last line of defence.
 *
 * `AsyncBoundary` handles *expected* failures (a rejected fetch). This catches
 * *unexpected* ones: a render-time TypeError from an unanticipated data shape,
 * for example. Without it, React 18 unmounts the whole tree on a throw and the
 * reviewer gets a white screen with nothing but a console trace.
 *
 * Must be a class: there is still no hook equivalent of componentDidCatch.
 */

import { AlertOctagon, RotateCcw } from 'lucide-react';
import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  /** Remounting the subtree when this changes clears a stuck error on
   *  navigation, so a broken page doesn't poison the rest of the session. */
  resetKey?: string;
  fallback?: (error: Error, reset: () => void) => ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidUpdate(prevProps: Props) {
    if (prevProps.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // No error-reporting service is wired up in this project, so the console is
    // the honest destination. A real deployment would forward this to Sentry.
    console.error('Unhandled render error:', error, info.componentStack);
  }

  private reset = () => this.setState({ error: null });

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    if (this.props.fallback) return this.props.fallback(error, this.reset);

    return (
      <div
        role="alert"
        className="m-6 flex flex-col items-center gap-3 rounded-card border border-risk-highBorder bg-risk-highBg p-10 text-center"
      >
        <AlertOctagon className="text-risk-high" size={28} aria-hidden="true" />
        <div>
          <p className="text-heading-lg text-navy-900">This page failed to render</p>
          <p className="mx-auto mt-1 max-w-md text-caption text-navy-600">
            An unexpected error occurred. The rest of the dashboard is still usable — try
            reloading this section, or navigate to another page.
          </p>
        </div>
        <pre className="max-w-lg overflow-x-auto rounded-control bg-navy-900/5 px-3 py-2 text-left font-mono text-micro text-navy-700">
          {error.message}
        </pre>
        <button
          type="button"
          onClick={this.reset}
          className="inline-flex items-center gap-1.5 rounded-control border border-navy-200 bg-surface px-3 py-1.5 text-caption font-medium text-navy-700 transition-colors hover:bg-navy-50"
        >
          <RotateCcw size={13} aria-hidden="true" />
          Try again
        </button>
      </div>
    );
  }
}
