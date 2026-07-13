/**
 * The Inline Core generation path. Mirrors the fal executor: build the graph, submit it, poll the
 * run to completion, download each take, and stream progress/done via the shared GenEmitter. Inline
 * Core owns durability and batching; here we just drive one frame's run.
 */
import { randomUUID } from 'node:crypto'
import { join, extname } from 'node:path'
import { writeFileSync, mkdirSync } from 'node:fs'
import type { Take } from '@shared/types'
import type { GenEmitter } from './executor'
import { getOpenProjectFolder } from '../db'
import { addTake, setHero } from '../frames/store'
import { buildFrameGraph } from '../core/graph'
import { buildWorkflowGraph } from '../core/workflow'
import { setCoreNodeOutput } from '../moodboard/store'
import { submitRun, getRun, cancelRun, downloadTake, type CoreTake } from '../core/client'

const delay = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))

interface ActiveCoreRun {
  abort: AbortController
  runId: string
}
const active = new Map<string, ActiveCoreRun>()
const cancelled = new Set<string>()

const EXT_BY_KIND: Record<CoreTake['kind'], string> = {
  image: '.png',
  video: '.mp4',
  audio: '.mp3',
}

export async function runCoreGraph(frameId: string, emit: GenEmitter): Promise<void> {
  cancelled.delete(frameId)
  const abort = new AbortController()
  active.set(frameId, { abort, runId: '' })
  try {
    const folder = getOpenProjectFolder()
    if (!folder) throw new Error('No project is open.')
    emit.progress(frameId, 0.02, 'Submitting')
    const { graph, target } = buildFrameGraph(frameId)
    const runId = await submitRun(graph, target, abort.signal)
    const entry = active.get(frameId)
    if (entry) entry.runId = runId
    await pollRun(frameId, runId, folder, abort.signal, emit)
  } catch (e) {
    if (cancelled.has(frameId)) {
      cancelled.delete(frameId)
      return
    }
    emit.error(frameId, e instanceof Error ? e.message : String(e), frameId)
  } finally {
    active.delete(frameId)
  }
}

async function pollRun(
  frameId: string,
  runId: string,
  folder: string,
  signal: AbortSignal,
  emit: GenEmitter,
): Promise<void> {
  const seen = new Set<string>()
  let heroSet = false
  for (;;) {
    if (signal.aborted) throw new Error('Generation cancelled.')
    const state = await getRun(runId, signal)
    emit.progress(frameId, Math.max(0.05, state.fraction), state.nodes[frameId]?.status)
    for (const take of state.takes) {
      if (seen.has(take.id)) continue
      seen.add(take.id)
      const saved = await saveTake(frameId, runId, take, folder)
      if (!heroSet) {
        setHero(frameId, saved.id)
        heroSet = true
      }
      emit.nodeDone(frameId, saved)
    }
    if (state.status === 'done') {
      emit.done(frameId)
      return
    }
    if (state.status === 'cancelled') return
    if (state.status === 'error') throw new Error(state.error?.message ?? 'Generation failed.')
    await delay(600)
  }
}

async function saveTake(
  frameId: string,
  runId: string,
  take: CoreTake,
  folder: string,
): Promise<Take> {
  const bytes = await downloadTake(take)
  const ext = extname(take.uri) || EXT_BY_KIND[take.kind]
  const rel = `takes/${randomUUID()}${ext}`
  mkdirSync(join(folder, 'takes'), { recursive: true })
  writeFileSync(join(folder, rel), bytes)
  return addTake({
    frameId,
    filePath: rel,
    kind: take.kind,
    comfyPromptId: runId,
    params: take.params,
  })
}

/** Run a canvas workflow: serialize the closure of `itemId`, submit it, stream progress, and store
 * each media node's output on its item. Takes attach to `core` items (not Frames). */
export async function runCoreWorkflow(itemId: string, emit: GenEmitter): Promise<void> {
  cancelled.delete(itemId)
  const abort = new AbortController()
  active.set(itemId, { abort, runId: '' })
  try {
    const folder = getOpenProjectFolder()
    if (!folder) throw new Error('No project is open.')
    emit.progress(itemId, 0.02, 'Submitting')
    const { graph, target } = buildWorkflowGraph(itemId)
    const runId = await submitRun(graph, target, abort.signal)
    const entry = active.get(itemId)
    if (entry) entry.runId = runId
    await pollWorkflow(itemId, runId, folder, abort.signal, emit)
  } catch (e) {
    if (cancelled.has(itemId)) {
      cancelled.delete(itemId)
      return
    }
    emit.error(itemId, e instanceof Error ? e.message : String(e), itemId)
  } finally {
    active.delete(itemId)
  }
}

async function pollWorkflow(
  targetItemId: string,
  runId: string,
  folder: string,
  signal: AbortSignal,
  emit: GenEmitter,
): Promise<void> {
  const seen = new Set<string>()
  for (;;) {
    if (signal.aborted) throw new Error('Generation cancelled.')
    const state = await getRun(runId, signal)
    emit.progress(targetItemId, Math.max(0.05, state.fraction), state.nodes[targetItemId]?.status)
    for (const take of state.takes) {
      if (seen.has(take.id)) continue
      seen.add(take.id)
      const filePath = await saveWorkflowTake(take, folder)
      setCoreNodeOutput(take.nodeId, { takeId: take.id, filePath, kind: take.kind })
    }
    if (state.status === 'done') {
      emit.done(targetItemId)
      return
    }
    if (state.status === 'cancelled') return
    if (state.status === 'error') throw new Error(state.error?.message ?? 'Generation failed.')
    await delay(600)
  }
}

async function saveWorkflowTake(take: CoreTake, folder: string): Promise<string> {
  const bytes = await downloadTake(take)
  const ext = extname(take.uri) || EXT_BY_KIND[take.kind]
  const rel = `takes/${randomUUID()}${ext}`
  mkdirSync(join(folder, 'takes'), { recursive: true })
  writeFileSync(join(folder, rel), bytes)
  return rel
}

export async function cancelCoreGeneration(frameId?: string): Promise<void> {
  const entries: Array<[string, ActiveCoreRun]> = frameId
    ? active.has(frameId)
      ? [[frameId, active.get(frameId) as ActiveCoreRun]]
      : []
    : [...active.entries()]
  for (const [id, entry] of entries) {
    cancelled.add(id)
    active.delete(id)
    entry.abort.abort()
    if (entry.runId) void cancelRun(entry.runId)
  }
}
