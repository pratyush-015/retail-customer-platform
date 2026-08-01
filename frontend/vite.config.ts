import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import path from 'node:path';

/**
 * The dev server proxies `/api` to the backend so the browser sees one origin.
 * That keeps CORS out of local development entirely and lets `VITE_API_URL`
 * stay unset locally while pointing at the deployed backend on Vercel.
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_PROXY_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        // Recharts + d3 dominate the bundle; splitting them keeps the initial
        // parse small so the KPI cards paint before the charts are ready.
        manualChunks: {
          react: ['react', 'react-dom', 'react-router-dom'],
          charts: ['recharts'],
        },
      },
    },
  },
});
