/**
 * ESLint config (eslintrc format, matching the ESLint 8 pin in package.json).
 *
 * Kept close to the recommended presets. The deviations below are the ones that
 * matter for a TypeScript React codebase where `tsc --noEmit` already runs in
 * CI, so rules that merely duplicate the compiler are turned off rather than
 * reported twice.
 */

module.exports = {
  root: true,
  env: { browser: true, es2022: true },
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    ecmaFeatures: { jsx: true },
  },
  plugins: ['@typescript-eslint', 'react-hooks', 'react-refresh'],
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react-hooks/recommended',
  ],
  settings: { react: { version: 'detect' } },
  rules: {
    // The dependency-array rule is the one that catches real staleness bugs;
    // a warning would be ignored, so it's an error.
    'react-hooks/exhaustive-deps': 'error',

    // Allow deliberately unused args when they document a signature, but only
    // when prefixed with an underscore.
    '@typescript-eslint/no-unused-vars': [
      'error',
      { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
    ],

    // `any` is banned; the API layer is fully typed, so reaching for it means
    // a contract type is missing.
    '@typescript-eslint/no-explicit-any': 'error',

    // Vite's fast refresh requires component files to export only components.
    // The page modules use default exports for lazy loading, which is allowed.
    'react-refresh/only-export-components': [
      'warn',
      { allowConstantExport: true },
    ],

    // console.error is how the ErrorBoundary reports unhandled render errors,
    // so warn/error are permitted and stray console.log is not.
    'no-console': ['error', { allow: ['warn', 'error'] }],
  },
  overrides: [
    {
      // These two modules intentionally export helpers alongside components:
      // `Toast.tsx` pairs the `useToast` hook with its provider, and
      // `Primitives.tsx` pairs the tone-mapping helpers with the badges that
      // consume them. Splitting them would scatter a cohesive unit across
      // files to satisfy a rule whose only benefit is finer-grained hot
      // reload. The tradeoff is a full module reload on edit, which is
      // acceptable for two small files.
      files: ['src/components/ui/Toast.tsx', 'src/components/ui/Primitives.tsx'],
      rules: { 'react-refresh/only-export-components': 'off' },
    },
  ],
};
