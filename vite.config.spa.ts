import { resolve } from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Self-booting SPA build of the renderer, served by Inline Core on one port (mirrors ComfyUI's
 * frontend package). Unlike vite.config.ui.ts (a *library* consumed by the Node web server), this
 * emits a real index.html + hashed assets that boot themselves against a same-origin backend.
 *
 *   npm run build:spa   -> dist-web/  (the payload of the inline_studio_frontend PyPI package)
 *   npm run dev:web     -> Vite dev server on :5173 with HMR, proxying the backend routes to Core
 *                          (:8848). Edit the UI and see it live without rebuilding or republishing.
 */
const CORE = process.env.INLINE_CORE_URL ?? 'http://127.0.0.1:8848'
const CORE_WS = CORE.replace(/^http/, 'ws')

export default defineConfig({
  root: resolve('src/renderer/web'),
  resolve: {
    alias: {
      '@': resolve('src/renderer'),
      '@shared': resolve('src/shared'),
    },
  },
  plugins: [react()],
  build: {
    outDir: resolve('dist-web'),
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    // Dev loop: the app runs from Vite (HMR) but all backend calls proxy to the running Core, so the
    // one-port production shape is faithfully mimicked without a rebuild.
    proxy: {
      '/rpc': CORE,
      '/upload': CORE,
      '/media': CORE,
      '/download': CORE,
      '/v1': CORE,
      '/studio': CORE,
      '/events': { target: CORE_WS, ws: true },
    },
  },
})
