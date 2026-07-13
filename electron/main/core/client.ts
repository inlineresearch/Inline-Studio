/**
 * The Inline Core (/v1) transport. All Inline Core knowledge lives here (engine-isolation rule;
 * mirrors comfy/client.ts and fal/client.ts). The executor and graph builder stay transport-agnostic.
 */
import type { CoreModels } from '@shared/coreNodes'
import type { CoreStatus } from '@shared/types'
import { getSettings } from '../settings/store'

function baseUrl(): string {
  return getSettings().coreUrl.replace(/\/+$/, '')
}

export interface CoreTake {
  id: string
  nodeId: string
  kind: 'image' | 'video' | 'audio'
  uri: string
  hash: string
  params: Record<string, unknown>
}

export interface CoreNodeState {
  state: string
  fraction: number
  status?: string
}

export interface CoreRunState {
  runId: string
  status: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  target: string
  fraction: number
  nodes: Record<string, CoreNodeState>
  takes: CoreTake[]
  error?: { message: string; nodeId?: string } | null
}

/** Is the configured Inline Core reachable? */
export async function pingCore(): Promise<CoreStatus> {
  const url = baseUrl()
  try {
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), 5000)
    const res = await fetch(`${url}/v1/health`, { signal: ctrl.signal })
    clearTimeout(timer)
    return { running: res.ok, url }
  } catch {
    return { running: false, url }
  }
}

/** The node descriptors served at GET /v1/models (the palette source). */
export async function fetchModels(): Promise<CoreModels> {
  const res = await fetch(`${baseUrl()}/v1/models`)
  if (!res.ok) throw new Error(`Could not fetch Inline Core models (${res.status}).`)
  return (await res.json()) as CoreModels
}

/** Submit a graph; returns the run id. Surfaces Inline Core's validation error (422) message. */
export async function submitRun(
  graph: unknown,
  target: string,
  signal?: AbortSignal,
): Promise<string> {
  const res = await fetch(`${baseUrl()}/v1/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ graph, target }),
    signal,
  })
  if (!res.ok) {
    const detail = (await res.json().catch(() => null)) as { error?: { message?: string } } | null
    throw new Error(detail?.error?.message ?? `Inline Core rejected the run (${res.status}).`)
  }
  return ((await res.json()) as { runId: string }).runId
}

/** The durable, pollable state of a run. */
export async function getRun(runId: string, signal?: AbortSignal): Promise<CoreRunState> {
  const res = await fetch(`${baseUrl()}/v1/runs/${runId}`, { signal })
  if (!res.ok) throw new Error(`Could not read the Inline Core run (${res.status}).`)
  return (await res.json()) as CoreRunState
}

/** Ask Inline Core to cancel a run (best-effort). */
export async function cancelRun(runId: string): Promise<void> {
  await fetch(`${baseUrl()}/v1/runs/${runId}`, { method: 'DELETE' }).catch(() => {})
}

/** Download a take's bytes. */
export async function downloadTake(take: CoreTake): Promise<Buffer> {
  const res = await fetch(`${baseUrl()}/v1/takes/${take.id}/bytes`)
  if (!res.ok) throw new Error(`Could not download the take (${res.status}).`)
  return Buffer.from(await res.arrayBuffer())
}
