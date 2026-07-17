import js from '@eslint/js'
import tseslint from 'typescript-eslint'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'

export default tseslint.config(
  // `core/` is the Python engine (incl. its .venv) — never lint it with the TS/JS linter.
  { ignores: ['out', 'dist', 'dist-web', 'node_modules', 'core', '*.config.js', '*.config.ts'] },

  // Base TypeScript rules for the whole repo
  {
    files: ['**/*.{ts,tsx}'],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: { ecmaVersion: 2023, sourceType: 'module' },
    rules: {
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    },
  },

  // Renderer (React) rules + the LAYERING RULE:
  // the renderer must never import Node / Electron-main modules directly.
  // All outside-world access goes through the typed IPC bridge (window.storyline).
  {
    files: ['src/renderer/**/*.{ts,tsx}'],
    plugins: { 'react-hooks': reactHooks, 'react-refresh': reactRefresh },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
      'no-restricted-imports': [
        'error',
        {
          paths: [
            { name: 'electron', message: 'Renderer must use the IPC bridge, not electron directly.' },
            { name: 'fs', message: 'Filesystem access belongs in the main process (via IPC).' },
            { name: 'node:fs', message: 'Filesystem access belongs in the main process (via IPC).' },
            { name: 'path', message: 'Use the main process for path logic (via IPC).' },
            { name: 'node:path', message: 'Use the main process for path logic (via IPC).' },
            { name: 'better-sqlite3', message: 'DB access belongs in the main process (via IPC).' },
            { name: 'ws', message: 'Sockets belong in the main process (via IPC).' },
            { name: 'fluent-ffmpeg', message: 'ffmpeg belongs in the main process (via IPC).' },
          ],
          patterns: [
            { group: ['@main/*', '../../electron/*'], message: 'Renderer must not import main-process code; use @shared types + IPC.' },
          ],
        },
      ],
    },
  },
)
