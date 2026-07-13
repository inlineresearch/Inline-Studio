import { resolve } from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Standalone library build of the renderer as @inlineresearch/ui (entry mountStudioApp). Separate
 * from electron-vite, which builds the desktop app; run with `vite build --config vite.config.ui.ts`.
 * React is bundled in (not externalized) so the headless web server can serve one self-contained
 * index.js + style.css. emptyOutDir is off to preserve the committed dist/index.d.ts contract.
 */
export default defineConfig({
  resolve: {
    alias: {
      '@': resolve('src/renderer'),
      '@shared': resolve('src/shared'),
    },
  },
  // Library builds do not auto-replace process.env.NODE_ENV like app builds do; React and other
  // deps read it and would throw "process is not defined" in the browser. Pin it to production.
  define: {
    'process.env.NODE_ENV': JSON.stringify('production'),
  },
  plugins: [react()],
  build: {
    outDir: 'src/renderer/dist',
    emptyOutDir: false,
    cssCodeSplit: false,
    lib: {
      entry: resolve('src/renderer/index.ts'),
      formats: ['es'],
      fileName: () => 'index.js',
    },
    rollupOptions: {
      output: { assetFileNames: 'style.css' },
    },
  },
})
