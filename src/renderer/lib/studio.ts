/**
 * The backend seam. The renderer reaches the backend only through `studio()`, never
 * `window.inlineStudio` directly, so the same UI runs under Electron (the preload bridge,
 * the default) or in the browser (an injected HTTP/WebSocket client). `mountStudioApp`
 * calls `setStudioClient` before rendering; under Electron nothing injects and it falls
 * back to the preload bridge.
 */
import type { InlineStudioApi } from '@shared/ipc'

let client: InlineStudioApi | null = null

/** Inject the backend client. Call before the app renders (mountStudioApp does this). */
export function setStudioClient(next: InlineStudioApi): void {
  client = next
}

/** The active backend client. Stores and views call this instead of window.inlineStudio. */
export function studio(): InlineStudioApi {
  return client ?? window.inlineStudio
}
