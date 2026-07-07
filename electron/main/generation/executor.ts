/**
 * The DAG generation executor. Clicking a fal node's play button runs that frame AND its upstream
 * chain (topologically): each node resolves its inputs (upstream hero takes / assets), calls fal,
 * and saves the outputs as takes. Progress + completion stream back to the renderer via a GenEmitter.
 *
 * The DAG lives in `frame_inputs.source_frame_id` (the flow link) — the same edges the canvas
 * creates via `addSourceInput`. This engine is fal-specific; ComfyUI keeps its own UI-driven flow.
 */
import { randomUUID } from 'node:crypto'
import { join, extname } from 'node:path'
import { writeFileSync, mkdirSync } from 'node:fs'
import type { Take, AssetKind } from '@shared/types'
import { getNodeDef } from '@shared/nodes/registry'
import {
  emptyResolvedInputs,
  type FalOutputRef,
  type NodeDef,
  type PortKind,
  type ResolvedInputs,
} from '@shared/nodes/types'
import { getOpenProjectFolder } from '../db'
import { getFalFrame, frameInputAssetPaths, addTake, setHero } from '../frames/store'
import { promptTextForFrame } from '../moodboard/store'
import { fileToInputUrl, submit, pollResult, cancelRequest, type FalHandle } from '../fal/client'
import { recordPending, deletePendingByFrame, listPending, type PendingRun } from './pending'

/** Main → renderer callbacks the IPC layer wires to broadcast() events. */
export interface GenEmitter {
  progress(frameId: string, fraction: number, status?: string): void
  nodeDone(frameId: string, take: Take): void
  done(targetFrameId: string): void
  error(targetFrameId: string, error: string, frameId?: string): void
}

/** Direct upstream frame ids feeding `frameId` (its flow-link source frames). */
export type EdgesFn = (frameId: string) => string[]

// --- pure graph helpers (unit-tested; no DB/network) ---

/** All frames reachable upstream of `target` (inclusive). Visited-set guards cycles. */
export function upstreamClosure(target: string, edges: EdgesFn): string[] {
  const seen = new Set<string>()
  const stack = [target]
  while (stack.length > 0) {
    const id = stack.pop() as string
    if (seen.has(id)) continue
    seen.add(id)
    for (const up of edges(id)) if (!seen.has(up)) stack.push(up)
  }
  return [...seen]
}

/**
 * Kahn topological sort over the subgraph induced by `ids` (edge: source → dependent), returning
 * run order (dependencies first). Throws on a residual cycle.
 */
export function topoSort(ids: string[], edges: EdgesFn): string[] {
  const set = new Set(ids)
  const indegree = new Map<string, number>()
  const dependents = new Map<string, string[]>()
  for (const id of ids) {
    indegree.set(id, 0)
    dependents.set(id, [])
  }
  for (const id of ids) {
    for (const up of edges(id)) {
      if (!set.has(up)) continue
      indegree.set(id, (indegree.get(id) ?? 0) + 1)
      dependents.get(up)?.push(id)
    }
  }
  const queue = ids.filter((id) => (indegree.get(id) ?? 0) === 0)
  const order: string[] = []
  while (queue.length > 0) {
    const id = queue.shift() as string
    order.push(id)
    for (const dep of dependents.get(id) ?? []) {
      const next = (indegree.get(dep) ?? 0) - 1
      indegree.set(dep, next)
      if (next === 0) queue.push(dep)
    }
  }
  if (order.length !== ids.length) throw new Error('This graph has a cycle.')
  return order
}

// --- media kind helpers ---

const VIDEO_EXTS = new Set(['.mp4', '.mov', '.webm', '.mkv', '.avi', '.m4v'])
const AUDIO_EXTS = new Set(['.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a'])

function kindFromExt(ext: string): AssetKind {
  const e = ext.toLowerCase()
  if (VIDEO_EXTS.has(e)) return 'video'
  if (AUDIO_EXTS.has(e)) return 'audio'
  return 'image'
}

/** How many resolved inputs satisfy a given port kind (for the required-input check). */
function resolvedCountForKind(resolved: ResolvedInputs, kind: PortKind): number {
  switch (kind) {
    case 'image':
    case 'image[]':
      return resolved.images.length
    case 'video':
      return resolved.videos.length
    case 'audio':
      return resolved.audios.length
    case 'text':
      return resolved.texts.length
    default:
      return 0
  }
}

// --- run state (per-frame active runs; cancel aborts + tells fal) ---

interface ActiveRun {
  abort: AbortController
  endpoint: string
  /** Empty until the request has been submitted (needed for fal-side cancel). */
  requestId: string
}
const active = new Map<string, ActiveRun>()
/** Frames whose current run the user cancelled — suppresses the error banner for that stop. */
const cancelledFrames = new Set<string>()

/** Cancel a specific frame's run, or all in-flight runs when no frame id is given. */
export async function cancelGeneration(frameId?: string): Promise<void> {
  const entries: [string, ActiveRun][] = frameId
    ? active.has(frameId)
      ? [[frameId, active.get(frameId) as ActiveRun]]
      : []
    : [...active.entries()]
  for (const [fId, run] of entries) {
    cancelledFrames.add(fId)
    active.delete(fId)
    run.abort.abort()
    deletePendingByFrame(fId)
    if (run.requestId) void cancelRequest(run.endpoint, run.requestId).catch(() => {})
  }
}

/**
 * Resolve a frame's inputs to fal input URLs, grouped by media kind. Reuses the existing
 * `frameInputAssetPaths` (asset files + upstream hero takes). Masks/text routing is a later
 * enhancement — for now every image input goes to `resolved.images`.
 */
async function resolveInputs(frameId: string, folder: string): Promise<ResolvedInputs> {
  const resolved = emptyResolvedInputs()
  for (const { filePath } of frameInputAssetPaths(frameId)) {
    const abs = join(folder, filePath)
    const url = await fileToInputUrl(abs)
    const kind = kindFromExt(extname(filePath))
    if (kind === 'video') resolved.videos.push(url)
    else if (kind === 'audio') resolved.audios.push(url)
    else resolved.images.push(url)
  }
  return resolved
}

/** Download one fal output into the project's takes/ folder and record it as a take. */
async function downloadToTake(
  frameId: string,
  ref: FalOutputRef,
  params: Record<string, unknown>,
  requestId: string,
  folder: string,
): Promise<Take> {
  const res = await fetch(ref.url)
  if (!res.ok) throw new Error(`Couldn't download the generated output (${res.status}).`)
  const buf = Buffer.from(await res.arrayBuffer())
  const rel = `takes/${randomUUID()}${ref.ext}`
  mkdirSync(join(folder, 'takes'), { recursive: true })
  writeFileSync(join(folder, rel), buf)
  return addTake({ frameId, filePath: rel, kind: ref.kind, comfyPromptId: requestId, params })
}

/**
 * Run a single fal frame (the one whose play button was clicked). Its image inputs resolve to
 * whatever their upstream nodes have already produced (their hero take); the prompt resolves from
 * a connected Prompt node. Nothing upstream is (re)run — one click generates exactly one node.
 */
export async function runGraph(targetFrameId: string, emit: GenEmitter): Promise<void> {
  cancelledFrames.delete(targetFrameId)
  const abort = new AbortController()
  try {
    const folder = getOpenProjectFolder()
    if (!folder) throw new Error('No project is open.')

    const { provider, modelId, params } = getFalFrame(targetFrameId)
    if (provider !== 'fal' || !modelId) throw new Error('This is not a generation node.')
    const def = getNodeDef(modelId)
    if (!def) throw new Error(`Unknown generation model: ${modelId}`)

    if (cancelledFrames.has(targetFrameId)) throw new Error('Generation cancelled.')
    emit.progress(targetFrameId, 0.02, 'Preparing inputs')

    const resolved = await resolveInputs(targetFrameId, folder)
    for (const port of def.inputs) {
      if (port.required && resolvedCountForKind(resolved, port.kind) === 0) {
        throw new Error(`${def.title} needs an input wired to "${port.label}".`)
      }
    }

    // The prompt is fed by a connected Prompt node (not a stored param).
    const promptText = promptTextForFrame(targetFrameId)
    if (!promptText) throw new Error('Connect a Prompt node with some text to generate.')
    const runParams = { ...params, prompt: promptText }

    const endpoint = def.resolveEndpoint(resolved)
    const body = def.buildRequest(runParams, resolved)
    active.set(targetFrameId, { abort, endpoint, requestId: '' })

    const { requestId, response } = await submit(endpoint, body, {
      signal: abort.signal,
      onProgress: (p) => emit.progress(targetFrameId, p.fraction, p.status),
      // As soon as fal accepts the request, persist it so a restart can resume/finish it.
      onSubmitted: (handle) => {
        const run = active.get(targetFrameId)
        if (run) run.requestId = handle.requestId
        recordPending({
          frameId: targetFrameId,
          modelId,
          endpoint,
          requestId: handle.requestId,
          statusUrl: handle.statusUrl,
          responseUrl: handle.responseUrl,
          params: runParams,
        })
      },
    })

    await finishRun(targetFrameId, def, response, runParams, requestId, folder, emit)
  } catch (e) {
    handleRunError(targetFrameId, e, emit)
  } finally {
    active.delete(targetFrameId)
  }
}

/** Download a completed response's outputs into takes, set the hero, and clear the pending row. */
async function finishRun(
  frameId: string,
  def: NodeDef,
  response: unknown,
  params: Record<string, unknown>,
  requestId: string,
  folder: string,
  emit: GenEmitter,
): Promise<void> {
  const refs = def.parseOutputs(response)
  if (refs.length === 0) throw new Error(`${def.title} returned no output.`)
  let firstTake: Take | null = null
  for (const ref of refs) {
    const take = await downloadToTake(frameId, ref, params, requestId, folder)
    if (!firstTake) firstTake = take
  }
  // addTake sets the last-added take as hero; prefer the first output deterministically.
  if (firstTake) {
    setHero(frameId, firstTake.id)
    emit.nodeDone(frameId, firstTake)
  }
  deletePendingByFrame(frameId)
  emit.done(frameId)
}

/** Clear the pending row and surface the failure — unless it was a user-initiated cancel. */
function handleRunError(frameId: string, e: unknown, emit: GenEmitter): void {
  deletePendingByFrame(frameId)
  if (cancelledFrames.has(frameId)) {
    // User pressed Stop: the renderer resets the node optimistically, so stay quiet.
    cancelledFrames.delete(frameId)
    return
  }
  emit.error(frameId, e instanceof Error ? e.message : String(e), frameId)
}

/**
 * On project open, finish any generations that were in flight when the app last closed: re-poll
 * each persisted fal request and download its take (or resume its progress). One run per frame.
 */
export async function resumePendingGenerations(emit: GenEmitter): Promise<void> {
  const folder = getOpenProjectFolder()
  if (!folder) return
  for (const run of listPending()) {
    const def = getNodeDef(run.modelId)
    if (!def) {
      deletePendingByFrame(run.frameId)
      continue
    }
    void resumeOne(run, def, folder, emit)
  }
}

async function resumeOne(
  run: PendingRun,
  def: NodeDef,
  folder: string,
  emit: GenEmitter,
): Promise<void> {
  const abort = new AbortController()
  active.set(run.frameId, { abort, endpoint: run.endpoint, requestId: run.requestId })
  cancelledFrames.delete(run.frameId)
  try {
    emit.progress(run.frameId, 0.05, 'Resuming…')
    const handle: FalHandle = {
      requestId: run.requestId,
      statusUrl: run.statusUrl,
      responseUrl: run.responseUrl,
    }
    const response = await pollResult(
      handle,
      (p) => emit.progress(run.frameId, p.fraction, p.status),
      abort.signal,
    )
    await finishRun(run.frameId, def, response, run.params, run.requestId, folder, emit)
  } catch (e) {
    handleRunError(run.frameId, e, emit)
  } finally {
    active.delete(run.frameId)
  }
}
