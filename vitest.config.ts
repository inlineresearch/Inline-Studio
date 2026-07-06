import { resolve } from 'node:path'
import { defineConfig } from 'vitest/config'

/**
 * Vitest resolves the same path aliases as the app builds (see electron.vite.config.ts), so
 * main-process modules that import `@shared/*` can be unit-tested directly.
 */
export default defineConfig({
  resolve: {
    alias: {
      '@shared': resolve('src/shared'),
      '@main': resolve('electron/main'),
      '@': resolve('src/renderer'),
    },
  },
})
