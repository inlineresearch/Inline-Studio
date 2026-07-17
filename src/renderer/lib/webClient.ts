/**
 * The browser backend client. Implements InlineStudioApi by posting {channel, args} to the web
 * server's POST /rpc (returning the same Result envelope the Electron preload returns) and by
 * subscribing to server events over a WebSocket. Built generically from IpcChannels so it never
 * drifts from the channel list. Injected via mountStudioApp({ client }); Electron uses the preload
 * bridge instead. getPathForFile has no meaning in a browser (no absolute paths) and returns ''.
 */
import { IpcChannels, type InlineStudioApi } from '@shared/ipc'
import { pickFilesViaInput, uploadFiles } from './importFiles'
import { resolveMedia } from './media'

type Args = unknown[]
type Unsub = () => void
type EventCb = (payload: unknown) => void

export function createWebClient(baseUrl: string): InlineStudioApi {
  const httpBase = baseUrl.replace(/\/$/, '')

  const rpc = async (channel: string, args: Args): Promise<unknown> => {
    try {
      const res = await fetch(`${httpBase}/rpc`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ channel, args }),
      })
      if (!res.ok) return { ok: false, error: `Request failed (HTTP ${res.status}).` }
      return await res.json()
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : String(e) }
    }
  }

  const listeners = new Map<string, Set<EventCb>>()
  const subscribe = (channel: string, cb: EventCb): Unsub => {
    const set = listeners.get(channel) ?? new Set<EventCb>()
    set.add(cb)
    listeners.set(channel, set)
    return () => {
      listeners.get(channel)?.delete(cb)
    }
  }
  connectEvents(httpBase.replace(/^http/, 'ws') + '/events', listeners)

  // Every non-event namespace: each method posts its channel to /rpc, returning the Result envelope.
  const api: Record<string, unknown> = {}
  for (const [namespace, channels] of Object.entries(IpcChannels)) {
    if (namespace === 'events') continue
    const methods: Record<string, (...a: Args) => Promise<unknown>> = {}
    for (const [name, channel] of Object.entries(channels as Record<string, string>)) {
      methods[name] = (...args: Args) => rpc(channel, args)
    }
    api[namespace] = methods
  }

  // events.on<Cap>(cb) => unsubscribe, one per events channel (timelineProgress belongs to timeline).
  const events: Record<string, (cb: EventCb) => Unsub> = {}
  for (const [name, channel] of Object.entries(IpcChannels.events as Record<string, string>)) {
    if (name === 'timelineProgress') continue
    events[`on${name.charAt(0).toUpperCase()}${name.slice(1)}`] = (cb) => subscribe(channel, cb)
  }
  api.events = events

  const timeline = api.timeline as Record<string, unknown>
  timeline.onProgress = (cb: EventCb): Unsub => subscribe(IpcChannels.events.timelineProgress, cb)
  api.getPathForFile = (): string => ''

  // Browser-native replacements for the desktop capability calls, so these do not hit /rpc (where
  // the web shell rejects them). Each returns the same Result envelope the renderer expects.
  const assets = api.assets as Record<string, unknown>
  assets.importDialog = async (folderId: string | null) => {
    const files = await pickFilesViaInput()
    return { ok: true, value: await uploadFiles(files, folderId) }
  }

  const shell = api.shell as Record<string, unknown>
  shell.openExternal = (url: string) => {
    window.open(url, '_blank', 'noopener,noreferrer')
    return Promise.resolve({ ok: true, value: undefined })
  }

  const clipboard = api.clipboard as Record<string, unknown>
  clipboard.writeText = (text: string) => tryVoid(() => navigator.clipboard.writeText(text))

  const media = api.media as Record<string, unknown>
  media.save = async (src: string, suggestedName: string) => {
    try {
      await downloadUrl(mediaHttpUrl(src), suggestedName)
      return { ok: true, value: true }
    } catch (e) {
      return { ok: false, error: errText(e) }
    }
  }
  media.copyImage = (src: string) =>
    tryVoid(async () => {
      const blob = await (await fetch(mediaHttpUrl(src))).blob()
      await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })])
    })

  return api as unknown as InlineStudioApi
}

/** Map a media reference (project-relative path, or an already-resolved URL) to a fetchable URL. */
function mediaHttpUrl(src: string): string {
  return /^(https?:|blob:|data:)/.test(src) || src.startsWith('/') ? src : resolveMedia(src)
}

/** Trigger a browser download of a URL under a suggested filename. */
async function downloadUrl(url: string, name: string): Promise<void> {
  const blob = await (await fetch(url)).blob()
  const objectUrl = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = objectUrl
  a.download = name || 'download'
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(objectUrl)
}

/** Run a void browser action, returning the Result envelope the renderer expects. */
async function tryVoid(
  fn: () => Promise<unknown>,
): Promise<{ ok: true; value: undefined } | { ok: false; error: string }> {
  try {
    await fn()
    return { ok: true, value: undefined }
  } catch (e) {
    return { ok: false, error: errText(e) }
  }
}

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

/** Keep one WebSocket to the server open, dispatching each {channel, payload} frame to listeners. */
function connectEvents(url: string, listeners: Map<string, Set<EventCb>>): void {
  const open = (): void => {
    const ws = new WebSocket(url)
    ws.onmessage = (ev: MessageEvent<string>): void => {
      try {
        const { channel, payload } = JSON.parse(ev.data) as { channel: string; payload: unknown }
        const set = listeners.get(channel)
        if (set) for (const cb of set) cb(payload)
      } catch {
        // ignore malformed frames
      }
    }
    ws.onclose = (): void => {
      setTimeout(open, 1000)
    }
  }
  open()
}
