import { resolve } from 'node:path'
import { defineConfig } from 'vite'

/**
 * SSR build of the headless Storyline web server to a single CommonJS bundle (better-sqlite3 is a
 * native CJS addon). Node dependencies (better-sqlite3, ws, archiver, ffmpeg, etc.) stay external
 * and load from node_modules at runtime; our TS is bundled. Run with `vite build --config
 * vite.config.server.ts`.
 */
export default defineConfig({
  resolve: {
    alias: {
      '@shared': resolve('src/shared'),
      '@main': resolve('electron/main'),
    },
  },
  build: {
    ssr: 'server/index.ts',
    outDir: 'out/server',
    emptyOutDir: true,
    target: 'node20',
    minify: false,
    rollupOptions: {
      output: { entryFileNames: 'index.cjs', format: 'cjs' },
    },
  },
})
