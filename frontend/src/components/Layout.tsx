/**
 * App shell: sidebar navigation, header, and the routed content outlet.
 *
 * Responsive strategy — the sidebar is a fixed rail from `lg` up, and collapses
 * to an off-canvas drawer below that. Tablet (768px) is the stated minimum, so
 * the breakpoint is chosen to keep the rail off screens where it would eat the
 * chart area rather than help.
 */

import {
  Activity,
  BarChart3,
  FlaskConical,
  LayoutDashboard,
  Menu,
  ShieldAlert,
  Users,
  X,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useHealth } from '@/hooks/useApi';
import { ErrorBoundary } from './ErrorBoundary';

const NAV_ITEMS = [
  { to: '/', label: 'Executive Overview', icon: LayoutDashboard, end: true },
  { to: '/segments', label: 'Customer Segments', icon: Users, end: false },
  { to: '/churn', label: 'Churn Risk', icon: ShieldAlert, end: false },
  { to: '/customers', label: 'Customer Lookup', icon: BarChart3, end: false },
  { to: '/insights', label: 'Statistical Insights', icon: FlaskConical, end: false },
];

/**
 * Live backend status in the sidebar footer.
 *
 * Worth the pixels: the most common local-setup failure is the API not running,
 * and this turns "why is everything broken?" into a visible red dot.
 */
function HealthIndicator() {
  const { data, isError, isPending } = useHealth();

  const { colour, label } = isPending
    ? { colour: 'bg-navy-300', label: 'Checking…' }
    : isError
      ? { colour: 'bg-risk-high', label: 'API unreachable' }
      : data?.status === 'ok'
        ? { colour: 'bg-risk-low', label: 'All systems normal' }
        : { colour: 'bg-risk-medium', label: `Degraded — ${data?.status ?? 'unknown'}` };

  return (
    <div className="flex items-center gap-2 px-3 py-2">
      <span className="relative flex h-2 w-2" aria-hidden="true">
        {!isPending && !isError && data?.status === 'ok' && (
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-risk-low opacity-60" />
        )}
        <span className={`relative inline-flex h-2 w-2 rounded-full ${colour}`} />
      </span>
      <span className="text-micro font-medium text-navy-300">{label}</span>
    </div>
  );
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <>
      <div className="flex items-center gap-2.5 px-4 py-5">
        <div className="rounded-control bg-teal-500 p-1.5">
          <Activity size={17} className="text-white" aria-hidden="true" />
        </div>
        <div>
          <p className="text-heading leading-tight text-white">Retail Insights</p>
          <p className="text-micro text-navy-400">Online Retail II</p>
        </div>
      </div>

      <nav className="flex-1 space-y-0.5 px-2" aria-label="Main navigation">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            onClick={onNavigate}
            className={({ isActive }) =>
              [
                'flex items-center gap-2.5 rounded-control px-3 py-2 text-body font-medium transition-colors duration-150',
                isActive
                  ? 'bg-navy-700 text-white'
                  : 'text-navy-300 hover:bg-navy-800 hover:text-white',
              ].join(' ')
            }
          >
            {({ isActive }) => (
              <>
                <Icon
                  size={16}
                  className={isActive ? 'text-teal-300' : ''}
                  aria-hidden="true"
                />
                {label}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-navy-800 pb-2 pt-2">
        <HealthIndicator />
        <a
          href="/docs"
          target="_blank"
          rel="noreferrer"
          className="block px-3 py-1 text-micro text-navy-400 transition-colors hover:text-teal-300"
        >
          API documentation ↗
        </a>
      </div>
    </>
  );
}

export function Layout() {
  const [isDrawerOpen, setDrawerOpen] = useState(false);
  const location = useLocation();

  // Close the drawer on navigation, otherwise it covers the page just reached.
  useEffect(() => setDrawerOpen(false), [location.pathname]);

  // Escape closes the drawer — expected of any overlay.
  useEffect(() => {
    if (!isDrawerOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setDrawerOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isDrawerOpen]);

  return (
    <div className="min-h-screen bg-surface-sunken">
      {/* Fixed rail, lg and up. */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-60 flex-col bg-navy-900 lg:flex">
        <SidebarContent />
      </aside>

      {/* Off-canvas drawer below lg. */}
      {isDrawerOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div
            className="absolute inset-0 bg-navy-950/50"
            onClick={() => setDrawerOpen(false)}
            aria-hidden="true"
          />
          <aside className="absolute inset-y-0 left-0 flex w-60 animate-slide-in-right flex-col bg-navy-900">
            <button
              type="button"
              onClick={() => setDrawerOpen(false)}
              aria-label="Close navigation"
              className="absolute right-2 top-4 text-navy-400 hover:text-white"
            >
              <X size={18} aria-hidden="true" />
            </button>
            <SidebarContent onNavigate={() => setDrawerOpen(false)} />
          </aside>
        </div>
      )}

      <div className="lg:pl-60">
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-navy-100 bg-surface/85 px-4 py-3 backdrop-blur-sm lg:hidden">
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            aria-label="Open navigation"
            aria-expanded={isDrawerOpen}
            className="rounded-control p-1.5 text-navy-600 transition-colors hover:bg-navy-50"
          >
            <Menu size={19} aria-hidden="true" />
          </button>
          <span className="text-heading text-navy-900">Retail Insights</span>
        </header>

        <main className="mx-auto max-w-[1600px] px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          {/* Keyed on pathname so a crashed page recovers when navigating away. */}
          <ErrorBoundary resetKey={location.pathname}>
            <Outlet />
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
