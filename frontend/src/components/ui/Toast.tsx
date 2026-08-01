/**
 * Minimal toast system: a context provider plus a `useToast()` hook.
 *
 * Hand-rolled rather than pulled from a library because the requirement is
 * modest (transient confirmations for copy/export/predict) and a dependency
 * would bring its own styling to override.
 */

import { CheckCircle2, Info, X, XCircle } from 'lucide-react';
import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';

type ToastTone = 'success' | 'error' | 'info';

interface Toast {
  id: number;
  tone: ToastTone;
  message: string;
}

interface ToastContextValue {
  toast: (message: string, tone?: ToastTone) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

const TONE_STYLES: Record<ToastTone, { icon: typeof Info; className: string }> = {
  success: { icon: CheckCircle2, className: 'border-risk-lowBorder bg-risk-lowBg text-risk-low' },
  error: { icon: XCircle, className: 'border-risk-highBorder bg-risk-highBg text-risk-high' },
  info: { icon: Info, className: 'border-navy-200 bg-surface text-navy-700' },
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  // Monotonic id: two toasts fired in the same millisecond would collide on
  // Date.now() and React would reuse a key.
  const nextId = useRef(0);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((item) => item.id !== id));
  }, []);

  const toast = useCallback(
    (message: string, tone: ToastTone = 'info') => {
      const id = nextId.current++;
      setToasts((current) => [...current, { id, tone, message }]);
      setTimeout(() => dismiss(id), 4000);
    },
    [dismiss],
  );

  const value = useMemo(() => ({ toast }), [toast]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      {/* aria-live so screen readers announce toasts without stealing focus. */}
      <div
        aria-live="polite"
        aria-atomic="false"
        className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-full max-w-xs flex-col gap-2"
      >
        {toasts.map((item) => {
          const { icon: Icon, className } = TONE_STYLES[item.tone];
          return (
            <div
              key={item.id}
              role="status"
              className={`pointer-events-auto flex animate-slide-in-right items-start gap-2 rounded-card border px-3 py-2.5 shadow-popover ${className}`}
            >
              <Icon size={15} className="mt-0.5 shrink-0" aria-hidden="true" />
              <p className="flex-1 text-caption font-medium text-navy-800">{item.message}</p>
              <button
                type="button"
                onClick={() => dismiss(item.id)}
                aria-label="Dismiss notification"
                className="shrink-0 text-navy-400 transition-colors hover:text-navy-700"
              >
                <X size={13} aria-hidden="true" />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const context = useContext(ToastContext);
  // Throwing beats returning a no-op: a silently swallowed toast is a bug that
  // only shows up when someone notices the confirmation never appears.
  if (!context) throw new Error('useToast must be used within a <ToastProvider>.');
  return context;
}
