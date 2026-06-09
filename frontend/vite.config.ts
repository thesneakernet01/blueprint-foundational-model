import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// The FastAPI backend runs alongside this server in the SAME CML container,
// privately on 127.0.0.1:$BACKEND_PORT. Separate CML applications can't reach
// each other over localhost, so the UI and API ship as one application: Vite
// owns the public port (CDSW_APP_PORT) and proxies /api/* inward to the
// backend. The SPA therefore uses same-origin relative URLs — no VITE_API_BASE.
const BACKEND_PORT = process.env.BACKEND_PORT || '7100';
const PUBLIC_PORT = Number(process.env.CDSW_APP_PORT || process.env.PORT || 8100);

const proxy = {
  '/api': {
    target: `http://127.0.0.1:${BACKEND_PORT}`,
    changeOrigin: true,
  },
} as const;

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  // `npm run dev` — local development; proxy to a backend on $BACKEND_PORT.
  server: {
    host: '0.0.0.0',
    strictPort: false,
    proxy,
  },
  // `npm run preview` — how CML serves the built bundle. Bind the public port
  // on loopback (CML convention) and accept the reverse proxy's Host header.
  preview: {
    host: '127.0.0.1',
    port: PUBLIC_PORT,
    strictPort: true,
    proxy,
    allowedHosts: true,
  },
});
