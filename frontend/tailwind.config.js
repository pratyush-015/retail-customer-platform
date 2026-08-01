/**
 * The single source of truth for the design system.
 *
 * Every colour, spacing step and type size used in the app is declared here so
 * pages compose from one vocabulary instead of accumulating one-off hex values.
 *
 * Palette rationale:
 * - `navy` is the brand/structural colour (sidebar, headings). Deep enough that
 *   white text clears WCAG AA on 700+.
 * - `teal` is the single accent, used for primary data series and active states.
 *   One accent keeps charts readable; a second competing hue would make it
 *   ambiguous which series matters.
 * - `risk` maps to the model's own bands (high/medium/low at 0.7/0.4), so red
 *   always means "acting on this is urgent" and never merely "negative number".
 * - `positive`/`negative` are for deltas (MoM growth), deliberately distinct from
 *   the risk ramp: a revenue drop is not the same kind of red as a churn risk.
 *
 * Every colour is defined as a full 50–900 ramp rather than a single value, so
 * dark mode can later invert by swapping the step (e.g. `navy-900` surface with
 * `navy-100` text) without redefining the palette.
 */

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        navy: {
          50: '#f0f4f8',
          100: '#dae4ee',
          200: '#b6c9dd',
          300: '#8aa7c6',
          400: '#5f83ab',
          500: '#436690',
          600: '#345173',
          700: '#2a405c',
          800: '#1e2f44',
          900: '#14202f',
          950: '#0c1420',
        },
        teal: {
          50: '#eefbfa',
          100: '#d3f4f2',
          200: '#ace9e7',
          300: '#74d8d6',
          400: '#3cbfbe',
          500: '#22a3a3',
          600: '#1a8284',
          700: '#19686a',
          800: '#195456',
          900: '#194648',
        },
        risk: {
          high: '#dc2626',
          highBg: '#fef2f2',
          highBorder: '#fecaca',
          medium: '#d97706',
          mediumBg: '#fffbeb',
          mediumBorder: '#fde68a',
          low: '#059669',
          lowBg: '#ecfdf5',
          lowBorder: '#a7f3d0',
        },
        positive: '#059669',
        negative: '#e11d48',
        surface: {
          DEFAULT: '#ffffff',
          sunken: '#f7f9fb',
          raised: '#ffffff',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      fontSize: {
        // A restrained ramp: display for KPI values, then a normal text scale.
        'display-lg': ['2.25rem', { lineHeight: '2.5rem', letterSpacing: '-0.02em', fontWeight: '650' }],
        display: ['1.875rem', { lineHeight: '2.25rem', letterSpacing: '-0.02em', fontWeight: '650' }],
        'heading-lg': ['1.25rem', { lineHeight: '1.75rem', letterSpacing: '-0.01em', fontWeight: '600' }],
        heading: ['1rem', { lineHeight: '1.5rem', letterSpacing: '-0.005em', fontWeight: '600' }],
        body: ['0.875rem', { lineHeight: '1.25rem' }],
        caption: ['0.8125rem', { lineHeight: '1.125rem' }],
        micro: ['0.6875rem', { lineHeight: '1rem', letterSpacing: '0.04em' }],
      },
      spacing: {
        // 4px base grid; named steps stop pages drifting to arbitrary values.
        gutter: '1.5rem',
        card: '1.25rem',
        section: '2rem',
      },
      borderRadius: {
        card: '0.75rem',
        control: '0.5rem',
      },
      boxShadow: {
        card: '0 1px 2px 0 rgb(20 32 47 / 0.04), 0 1px 3px 0 rgb(20 32 47 / 0.06)',
        'card-hover': '0 4px 6px -1px rgb(20 32 47 / 0.08), 0 2px 4px -2px rgb(20 32 47 / 0.06)',
        popover: '0 10px 15px -3px rgb(20 32 47 / 0.1), 0 4px 6px -4px rgb(20 32 47 / 0.1)',
      },
      keyframes: {
        // Skeletons shimmer rather than pulse: a moving highlight reads as
        // "loading", whereas a fading block can look like a disabled control.
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
        'fade-in-up': {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in-right': {
          '0%': { opacity: '0', transform: 'translateX(1rem)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
      },
      animation: {
        shimmer: 'shimmer 1.6s infinite',
        'fade-in-up': 'fade-in-up 0.24s ease-out',
        'slide-in-right': 'slide-in-right 0.2s ease-out',
      },
    },
  },
  plugins: [],
};
