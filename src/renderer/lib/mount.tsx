/**
 * Library entry for embedding the Studio UI in any shell. `mountStudioApp` injects the backend
 * client (an HTTP/WebSocket client in the browser, or nothing under Electron where it falls back to
 * window.inlineStudio) and renders the app. Importing the stylesheet here bundles it into the
 * library build so consumers get one JS + one CSS artifact.
 */
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import type { InlineStudioApi } from '@shared/ipc'
import { App } from '../App'
import { setStudioClient } from './studio'
import '../index.css'

export function mountStudioApp(root: HTMLElement, opts?: { client?: InlineStudioApi }): void {
  if (opts?.client) setStudioClient(opts.client)
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
}
