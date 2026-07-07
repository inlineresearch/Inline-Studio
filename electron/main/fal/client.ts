/**
 * The fal.ai transport — the single place that knows how to talk to fal (engine-isolation rule,
 * mirrors `comfy/client.ts`). Everything above it (the executor, node registry) is fal-agnostic.
 *
 * Implemented with the global `fetch` and fal's public queue REST API (no SDK dependency, matching
 * the rest of the main process). Local input files are passed as base64 `data:` URIs, which fal
 * accepts for file inputs — so we avoid the storage-upload handshake. If a model ever needs true
 * uploads (very large videos), swap `fileToInputUrl` to fal storage behind this same interface.
 *
 * A run is split into `submitRequest` (returns the request id + its status/response URLs right
 * away) and `pollResult` (streams progress, resolves the response). Persisting the handle between
 * those two steps is what lets a run survive an app restart (see generation/pending.ts).
 */
import { readFileSync } from 'node:fs'
import { extname } from 'node:path'
import { getApiKey } from './credentials'

const QUEUE_BASE = 'https://queue.fal.run'

/** 0..1 progress + an optional short status label, mapped from the fal queue state. */
export interface FalProgress {
  fraction: number
  status?: string
}

/** A submitted request: enough to poll it to completion later (even after a restart). */
export interface FalHandle {
  requestId: string
  statusUrl: string
  responseUrl: string
}

export interface FalSubmitResult {
  requestId: string
  response: unknown
}

interface QueueSubmitResponse {
  request_id: string
  status_url?: string
  response_url?: string
}

interface QueueLog {
  message?: string
  level?: string
  timestamp?: string
}

interface QueueStatusResponse {
  status?: 'IN_QUEUE' | 'IN_PROGRESS' | 'COMPLETED' | string
  queue_position?: number
  logs?: QueueLog[]
}

const MIME_BY_EXT: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.bmp': 'image/bmp',
  '.tiff': 'image/tiff',
  '.mp4': 'video/mp4',
  '.mov': 'video/quicktime',
  '.webm': 'video/webm',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.m4a': 'audio/mp4',
}

function requireKey(): string {
  const key = getApiKey()
  if (!key) throw new Error('Add a fal.ai API key in Settings to generate with the fal engine.')
  return key
}

function authHeaders(key: string): Record<string, string> {
  return { Authorization: `Key ${key}` }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

const clamp01 = (n: number): number => Math.max(0, Math.min(1, n))

/**
 * Turn a local file into a fal-usable input URL. Currently a base64 `data:` URI (fal resolves
 * these server-side). Kept async + behind this seam so it can become a real storage upload later.
 */
export async function fileToInputUrl(absPath: string): Promise<string> {
  const mime = MIME_BY_EXT[extname(absPath).toLowerCase()] ?? 'application/octet-stream'
  const buf = readFileSync(absPath)
  return `data:${mime};base64,${buf.toString('base64')}`
}

/**
 * Fine-grained progress from a model's IN_PROGRESS logs. Many fal diffusion/video models log their
 * step or percent (e.g. "12/30", "step 12 of 30", "45%"); we map that into the 0.1..0.95 band so the
 * bar tracks real work. Returns null when no log line carries progress (caller falls back to coarse).
 */
export function progressFromLogs(logs: QueueLog[] | undefined): FalProgress | null {
  if (!logs || logs.length === 0) return null
  for (let i = logs.length - 1; i >= 0; i--) {
    const msg = logs[i]?.message ?? ''
    const step = msg.match(/(\d+)\s*(?:\/|of)\s*(\d+)/i)
    if (step) {
      const cur = Number(step[1])
      const total = Number(step[2])
      if (total > 0 && cur >= 0 && cur <= total) {
        return { fraction: clamp01(0.1 + 0.85 * (cur / total)), status: `Step ${cur}/${total}` }
      }
    }
    const pct = msg.match(/(\d{1,3}(?:\.\d+)?)\s*%/)
    if (pct) {
      const p = Math.min(100, Number(pct[1]))
      return { fraction: clamp01(0.1 + 0.85 * (p / 100)), status: `${Math.round(p)}%` }
    }
  }
  return null
}

/**
 * Map a queue status payload to a 0..1 fraction + label. The label leads with the phase
 * (Queued / Generating / Done), then any step/percent detail — e.g. "Generating 40%".
 */
function statusToProgress(status: QueueStatusResponse): FalProgress {
  switch (status.status) {
    case 'IN_QUEUE': {
      const pos = status.queue_position
      return { fraction: 0.05, status: pos && pos > 0 ? `Queued (#${pos})` : 'Queued' }
    }
    case 'IN_PROGRESS': {
      const detail = progressFromLogs(status.logs)
      return detail
        ? { fraction: detail.fraction, status: `Generating ${detail.status}` }
        : { fraction: 0.5, status: 'Generating' }
    }
    case 'COMPLETED':
      return { fraction: 1, status: 'Done' }
    default:
      return { fraction: 0.1, status: status.status ?? 'Working' }
  }
}

/** Submit a request to the fal queue; returns immediately with the handle needed to poll it. */
export async function submitRequest(
  endpoint: string,
  input: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<FalHandle> {
  const key = requireKey()
  const submitRes = await fetch(`${QUEUE_BASE}/${endpoint}`, {
    method: 'POST',
    headers: { ...authHeaders(key), 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
    signal,
  })
  if (!submitRes.ok) {
    throw new Error(`fal request failed (${submitRes.status}): ${await safeText(submitRes)}`)
  }
  const submitted = (await submitRes.json()) as QueueSubmitResponse
  return resolveQueueUrls(endpoint, submitted)
}

/**
 * Resolve a submitted request's status + result URLs. fal's queue endpoints live under the BASE
 * app id (owner/app) — never a model sub-path like `/image-to-image` — so:
 *  - prefer the URLs fal returns;
 *  - else derive the result URL from the status URL (drop the trailing `/status`);
 *  - else fall back to the base app path (first two id segments), NOT the full endpoint.
 * Rebuilding from the full endpoint is what 404'd sub-path models (e.g. gpt-image-2/image-to-image).
 */
export function resolveQueueUrls(endpoint: string, submitted: QueueSubmitResponse): FalHandle {
  const requestId = submitted.request_id
  const base = endpoint.split('/').slice(0, 2).join('/')
  const statusUrl = submitted.status_url ?? `${QUEUE_BASE}/${base}/requests/${requestId}/status`
  const responseUrl =
    submitted.response_url ??
    submitted.status_url?.replace(/\/status(\?.*)?$/, '') ??
    `${QUEUE_BASE}/${base}/requests/${requestId}`
  return { requestId, statusUrl, responseUrl }
}

function withLogs(statusUrl: string): string {
  return statusUrl.includes('?') ? `${statusUrl}&logs=1` : `${statusUrl}?logs=1`
}

/**
 * Poll a submitted request to completion and return its raw response. `onProgress` receives
 * step/queue updates; `signal` aborts the run (cancel). Safe to call on a request submitted in a
 * previous app session (resume) — it just re-polls the same status/response URLs.
 */
export async function pollResult(
  handle: FalHandle,
  onProgress?: (p: FalProgress) => void,
  signal?: AbortSignal,
): Promise<unknown> {
  const key = requireKey()
  const statusUrl = withLogs(handle.statusUrl)

  const startedAt = Date.now()
  const TIMEOUT_MS = 10 * 60 * 1000
  for (;;) {
    if (signal?.aborted) throw new Error('Generation cancelled.')
    if (Date.now() - startedAt > TIMEOUT_MS) throw new Error('fal generation timed out.')
    await delay(1500)
    const statusRes = await fetch(statusUrl, { headers: authHeaders(key), signal })
    if (!statusRes.ok) {
      // Transient status errors shouldn't kill a long video job; keep polling a bit.
      if (statusRes.status >= 500) continue
      throw new Error(`fal status failed (${statusRes.status}): ${await safeText(statusRes)}`)
    }
    const status = (await statusRes.json()) as QueueStatusResponse
    onProgress?.(statusToProgress(status))
    if (status.status === 'COMPLETED') break
  }

  const resultRes = await fetch(handle.responseUrl, { headers: authHeaders(key), signal })
  if (!resultRes.ok) {
    throw new Error(`fal result failed (${resultRes.status}): ${await safeText(resultRes)}`)
  }
  return (await resultRes.json()) as unknown
}

/** Ask fal to cancel a queued/running request (best-effort; also abort the local poll). */
export async function cancelRequest(endpoint: string, requestId: string): Promise<void> {
  const key = requireKey()
  await fetch(`${QUEUE_BASE}/${endpoint}/requests/${requestId}/cancel`, {
    method: 'PUT',
    headers: authHeaders(key),
  })
}

/**
 * Submit + poll in one call, for a fresh run. `onSubmitted` fires as soon as the request id is
 * known (before polling), so the caller can persist the handle for restart-resume.
 */
export async function submit(
  endpoint: string,
  input: Record<string, unknown>,
  opts?: {
    onProgress?: (p: FalProgress) => void
    onSubmitted?: (handle: FalHandle) => void
    signal?: AbortSignal
  },
): Promise<FalSubmitResult> {
  const handle = await submitRequest(endpoint, input, opts?.signal)
  opts?.onSubmitted?.(handle)
  opts?.onProgress?.({ fraction: 0.05, status: 'Queued' })
  const response = await pollResult(handle, opts?.onProgress, opts?.signal)
  return { requestId: handle.requestId, response }
}

async function safeText(res: Response): Promise<string> {
  try {
    const text = await res.text()
    return text.slice(0, 500)
  } catch {
    return res.statusText
  }
}
