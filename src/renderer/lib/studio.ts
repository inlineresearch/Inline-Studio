/**
 * The backend seam. The renderer reaches the backend only through `studio()` - an injected
 * HTTP/WebSocket client that talks to Inline Core on the same origin. `mountStudioApp` calls
 * `setStudioClient` before rendering.
 */
import type { InlineStudioApi } from '@shared/ipc'

let client: InlineStudioApi | null = null

/** Inject the backend client. Call before the app renders (mountStudioApp does this). */
export function setStudioClient(next: InlineStudioApi): void {
  client = next
}

/** The active backend client. Stores and views call this to reach Inline Core. */
export function studio(): InlineStudioApi {
  if (!client) {
    throw new Error('No backend client injected. Call setStudioClient() before rendering.')
  }
  return client
}
