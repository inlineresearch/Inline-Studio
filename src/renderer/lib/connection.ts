/**
 * Whether the `/events` WebSocket to Core is up.
 *
 * This is the only continuous signal the browser has that Core is still there. An RPC call only
 * fails once you make one, so without the socket a dead or restarting server looks identical to an
 * idle one until the next click. `webClient` already reconnects every second; this just reports
 * which side of that loop we are on.
 */
type Listener = (connected: boolean) => void

let connected = false
const listeners = new Set<Listener>()

/** Called by the web client's socket handlers. */
export function setCoreConnected(next: boolean): void {
  if (next === connected) return
  connected = next
  for (const cb of [...listeners]) cb(connected)
}

export function isCoreConnected(): boolean {
  return connected
}

export function subscribeCoreConnection(cb: Listener): () => void {
  listeners.add(cb)
  return () => {
    listeners.delete(cb)
  }
}
