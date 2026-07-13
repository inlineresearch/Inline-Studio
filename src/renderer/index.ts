/**
 * Public entry for the renderer as a library (@inlineresearch/ui). The desktop app boots from
 * main.tsx instead; this barrel is what a browser shell imports to mount the same UI with an
 * injected backend client. Matches the committed dist/index.d.ts contract.
 */
export type { InlineStudioApi } from '@shared/ipc'
export { App } from './App'
export { studio, setStudioClient } from './lib/studio'
export { resolveMedia, setMediaResolver } from './lib/media'
export { mountStudioApp } from './lib/mount'
export { createWebClient } from './lib/webClient'
