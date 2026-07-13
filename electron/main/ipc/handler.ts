/**
 * Registers each IPC handler against the active transport and wraps its result in the
 * typed `Result<T>` envelope, so a thrown error never crosses the bridge raw (see CLAUDE.md
 * "Async & errors"). The transport is pluggable: Electron binds `ipcMain` (electronTransport),
 * the headless web server binds an HTTP/WebSocket router. This file stays electron-free so the
 * same handler modules run under both shells.
 */
import { ok, err, type Result } from '@shared/result'

/** A registered handler: takes the call args, returns the Result envelope. */
export type Invoker = (...args: unknown[]) => Promise<Result<unknown>>

/** How handlers reach a shell. Set once at boot via setTransport before registering. */
export interface Transport {
  register(channel: string, invoke: Invoker): void
}

let transport: Transport | null = null

/** Bind the shell transport. Call before registerIpcHandlers(). */
export function setTransport(next: Transport): void {
  transport = next
}

export function handle<TArgs extends unknown[], TResult>(
  channel: string,
  fn: (...args: TArgs) => TResult | Promise<TResult>,
): void {
  if (!transport) {
    throw new Error(
      `Cannot register '${channel}': no IPC transport set. Call setTransport() first.`,
    )
  }
  transport.register(channel, async (...args: unknown[]): Promise<Result<TResult>> => {
    try {
      const value = await fn(...(args as TArgs))
      return ok(value)
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      console.error(`[ipc:${channel}]`, message)
      return err(message)
    }
  })
}
