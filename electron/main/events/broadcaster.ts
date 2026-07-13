/**
 * Server-to-client events (library changed, generation progress, timeline progress, updates)
 * go through this seam instead of touching Electron windows directly. Electron fans out to
 * every BrowserWindow; the web server fans out to every open WebSocket. Defaults to a no-op so
 * a broadcast before a shell is wired is safe rather than a crash.
 */
export interface Broadcaster {
  send(channel: string, payload?: unknown): void
}

let broadcaster: Broadcaster = { send() {} }

/** Bind the shell broadcaster. Call before any handler can emit. */
export function setBroadcaster(next: Broadcaster): void {
  broadcaster = next
}

/** Push an event to every connected client. */
export function broadcast(channel: string, payload?: unknown): void {
  broadcaster.send(channel, payload)
}
