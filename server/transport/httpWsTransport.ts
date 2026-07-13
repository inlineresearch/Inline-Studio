/**
 * The HTTP transport: handlers register here (via setTransport), and POST /rpc dispatches a
 * {channel, args} body to the matching invoker, returning its Result envelope as JSON. Same
 * envelope the Electron bridge returns, so the web client is a drop-in for window.inlineStudio.
 */
import type { Invoker, Transport } from '@main/ipc/handler'
import type { Result } from '@shared/result'

export class HttpWsTransport implements Transport {
  private readonly invokers = new Map<string, Invoker>()

  register(channel: string, invoke: Invoker): void {
    this.invokers.set(channel, invoke)
  }

  async dispatch(channel: string, args: unknown[]): Promise<Result<unknown>> {
    const invoke = this.invokers.get(channel)
    if (!invoke) return { ok: false, error: `Unknown channel: ${channel}` }
    return invoke(...args)
  }
}
