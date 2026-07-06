/**
 * The ComfyUI bridge. All ComfyUI knowledge lives here (CLAUDE.md engine-isolation
 * rule). Slice B is an embed + bridge: we don't drive workflows via the API yet —
 * we upload a frame's input so it's available in Comfy, and pull the latest output
 * back as a take. Uses Comfy's HTTP API: /system_stats, /upload/image, /history, /view.
 */
import { join, extname } from 'node:path'
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs'
import { randomUUID } from 'node:crypto'
import type { Take, ComfyStatus, AssetKind, Frame, ComfyOutput, ComfyRun } from '@shared/types'
import { getSettings } from '../settings/store'
import { getOpenProjectFolder } from '../db'
import {
  addTake,
  getFrameById,
  linkWorkflow,
  setWorkflowReady,
  frameInputFileNames,
  frameInputAssetPaths,
} from '../frames/store'
import { getCurrentProject } from '../project/store'

function baseUrl(): string {
  return getSettings().comfyUrl.replace(/\/+$/, '')
}

const delay = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))

const VIDEO_EXTS = new Set(['.mp4', '.webm', '.mov', '.mkv', '.gif', '.avi', '.m4v'])
const AUDIO_EXTS = new Set(['.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a'])

function kindForExt(ext: string): AssetKind {
  const e = ext.toLowerCase()
  if (VIDEO_EXTS.has(e)) return 'video'
  if (AUDIO_EXTS.has(e)) return 'audio'
  return 'image'
}

/** Is the configured ComfyUI reachable? */
export async function ping(): Promise<ComfyStatus> {
  const url = baseUrl()
  try {
    const ctrl = new AbortController()
    // Generous: ComfyUI's server can stall on its event loop mid-render, and a too-short
    // timeout here reads as "not reachable" and would tear down the embedded page.
    const timer = setTimeout(() => ctrl.abort(), 6000)
    const res = await fetch(`${url}/system_stats`, { signal: ctrl.signal })
    clearTimeout(timer)
    return { running: res.ok, url }
  } catch {
    return { running: false, url }
  }
}

// ── Capability detection ──────────────────────────────────────────────────────
// Seeded workflows must only reference nodes and model files that actually exist in THIS
// ComfyUI. `/object_info` is the source of truth: the full node catalogue, with installed
// model filenames appearing as enum options on loader widgets (ckpt_name, lora_name, vae_name, ...).

/** A compact snapshot of what a ComfyUI install can do. */
export interface ComfyCapabilities {
  url: string
  fetchedAt: number
  /** Every available node type name. */
  nodeTypes: string[]
  /** Installed model files by category (checkpoints, loras, vae, controlnet, ...). */
  models: Record<string, string[]>
}

/** Input/output schema for a single node type (for wiring sockets/widgets correctly). */
export interface ComfyNodeSchema {
  name: string
  input: { required: Record<string, unknown>; optional: Record<string, unknown> }
  output: string[]
  outputNames: string[]
}

/** Loader widget name → model category. Tolerant: a category is simply absent if unused. */
const MODEL_WIDGET_CATEGORIES: Record<string, string> = {
  ckpt_name: 'checkpoints',
  lora_name: 'loras',
  vae_name: 'vae',
  control_net_name: 'controlnet',
  unet_name: 'unet',
  clip_name: 'clip',
  style_model_name: 'style_models',
  gligen_name: 'gligen',
  model_name: 'upscale_models',
}

interface RawObjectInfo {
  [nodeType: string]: {
    input?: { required?: Record<string, unknown>; optional?: Record<string, unknown> }
    output?: unknown[]
    output_name?: unknown
  }
}

/** Short-lived cache of the raw /object_info so lookups don't refetch the large payload. */
let rawObjectInfo: { url: string; data: RawObjectInfo } | null = null

async function getObjectInfo(): Promise<RawObjectInfo> {
  const url = baseUrl()
  const ctrl = new AbortController()
  // /object_info can be several MB and slow on a cold cloud box.
  const timer = setTimeout(() => ctrl.abort(), 15000)
  try {
    const res = await fetch(`${url}/object_info`, { signal: ctrl.signal })
    if (!res.ok) throw new Error(`ComfyUI /object_info returned ${res.status}. Is it running?`)
    const data = (await res.json()) as RawObjectInfo
    rawObjectInfo = { url, data }
    return data
  } finally {
    clearTimeout(timer)
  }
}

/** Fetch + parse the full capability snapshot from the connected ComfyUI. */
export async function fetchCapabilities(): Promise<ComfyCapabilities> {
  const url = baseUrl()
  const info = await getObjectInfo()
  const nodeTypes = Object.keys(info).sort()
  const models: Record<string, Set<string>> = {}
  for (const node of Object.values(info)) {
    const specs = { ...(node?.input?.required ?? {}), ...(node?.input?.optional ?? {}) }
    for (const [widget, spec] of Object.entries(specs)) {
      const category = MODEL_WIDGET_CATEGORIES[widget]
      if (!category) continue
      // Enum options are encoded as [ ["a.safetensors", ...], {config} ]; a scalar socket
      // type like "MODEL" is NOT an enum, so only treat an array-of-options as model files.
      const options = Array.isArray(spec) && Array.isArray(spec[0]) ? (spec[0] as unknown[]) : null
      if (!options) continue
      const set = (models[category] ??= new Set())
      for (const o of options) if (typeof o === 'string') set.add(o)
    }
  }
  const modelsOut: Record<string, string[]> = {}
  for (const [k, v] of Object.entries(models)) modelsOut[k] = [...v].sort()
  return { url, fetchedAt: Date.now(), nodeTypes, models: modelsOut }
}

/** Return the input/output schema for the named node types (skips unknown names). */
export async function lookupNodeSchemas(names: string[]): Promise<ComfyNodeSchema[]> {
  const info =
    rawObjectInfo && rawObjectInfo.url === baseUrl() ? rawObjectInfo.data : await getObjectInfo()
  const out: ComfyNodeSchema[] = []
  for (const name of names) {
    const node = info[name]
    if (!node) continue
    out.push({
      name,
      input: { required: node.input?.required ?? {}, optional: node.input?.optional ?? {} },
      output: Array.isArray(node.output) ? node.output.map((o) => String(o)) : [],
      outputNames: Array.isArray(node.output_name) ? (node.output_name as string[]) : [],
    })
  }
  return out
}

/** The set of available node type names (cached snapshot), for validation. */
export async function availableNodeTypes(): Promise<Set<string>> {
  const info =
    rawObjectInfo && rawObjectInfo.url === baseUrl() ? rawObjectInfo.data : await getObjectInfo()
  return new Set(Object.keys(info))
}

function sanitizeSegment(name: string): string {
  // Keep only characters that survive ComfyUI's userdata path + workflow-store lookup
  // unchanged. Apostrophes, parentheses, slashes, etc. get encoded differently and
  // break the "open the saved workflow" match → it opens an Unsaved copy and Save
  // then conflicts (409). Allow letters, digits, space, dash, underscore.
  return (
    name
      .replace(/[^A-Za-z0-9 _-]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim() || 'untitled'
  )
}

/**
 * A minimal, guaranteed-to-load LiteGraph workflow with a Note titled after the frame.
 * When the frame has an input, it also seeds a LoadImage node pre-set to that input
 * (later re-pointed at the uploaded filename), so the displayed input loads on open.
 */
function buildSeedWorkflow(frameName: string, inputFileNames: string[]): unknown {
  const inputsLine = inputFileNames.length > 0 ? `\nInputs:\n  ${inputFileNames.join('\n  ')}` : ''
  const noteText =
    `Inline Studio frame: ${frameName}` +
    inputsLine +
    `\n\nBuild this frame's workflow here, then Save (the link persists).`
  const nodes: unknown[] = [
    {
      id: 1,
      type: 'Note',
      pos: [80, 80],
      size: [380, 120],
      flags: {},
      order: 0,
      mode: 0,
      inputs: [],
      outputs: [],
      title: frameName,
      properties: {},
      widgets_values: [noteText],
      color: '#432',
      bgcolor: '#653',
    },
  ]
  if (inputFileNames.length > 0) {
    nodes.push({
      id: 2,
      type: 'LoadImage',
      pos: [80, 230],
      size: [320, 314],
      flags: {},
      order: 1,
      mode: 0,
      inputs: [],
      outputs: [
        { name: 'IMAGE', type: 'IMAGE', links: null },
        { name: 'MASK', type: 'MASK', links: null },
      ],
      title: 'Input',
      properties: { 'Node name for S&R': 'LoadImage' },
      widgets_values: [inputFileNames[0], 'image'],
    })
  }
  return {
    last_node_id: inputFileNames.length > 0 ? 2 : 1,
    last_link_id: 0,
    nodes,
    links: [],
    groups: [],
    config: {},
    extra: {},
    version: 0.4,
  }
}

// ── Durable workflow storage ────────────────────────────────────────────────
// Inline Studio owns the canonical copy of each frame's workflow at
// <project>/workflows/<frameId>.json, so switching ComfyUI installs (e.g. an
// ephemeral cloud box) never loses it. ComfyUI holds a working copy under
// /userdata/workflows/<name>.json; we push our copy to it and pull edits back.

function workflowNameFor(frame: Frame): string {
  const project = getCurrentProject()
  const projectSeg = sanitizeSegment(project?.name ?? 'Project')
  const frameSeg = sanitizeSegment(frame.name)
  return `Inline Studio/${projectSeg}/${frameSeg} ${frame.id.slice(0, 6)}`
}

function localWorkflowPath(frameId: string): string {
  const folder = getOpenProjectFolder()
  if (!folder) throw new Error('No project is open.')
  return join(folder, 'workflows', `${frameId}.json`)
}

function readLocalWorkflow(frameId: string): unknown | null {
  const path = localWorkflowPath(frameId)
  if (!existsSync(path)) return null
  try {
    return JSON.parse(readFileSync(path, 'utf-8'))
  } catch {
    return null
  }
}

function writeLocalWorkflow(frameId: string, json: unknown): void {
  const folder = getOpenProjectFolder()
  if (!folder) throw new Error('No project is open.')
  mkdirSync(join(folder, 'workflows'), { recursive: true })
  writeFileSync(localWorkflowPath(frameId), JSON.stringify(json), 'utf-8')
}

function userdataUrl(name: string): string {
  return `${baseUrl()}/userdata/${encodeURIComponent(`workflows/${name}.json`)}`
}

/** Fetch a workflow from ComfyUI's userdata (null if it doesn't exist there). */
async function getRemoteWorkflow(name: string): Promise<unknown | null> {
  const res = await fetch(userdataUrl(name))
  if (res.status === 404) return null
  if (!res.ok) throw new Error(`Could not read the workflow from ComfyUI (${res.status}).`)
  return res.json()
}

/**
 * Store a workflow into ComfyUI's userdata, overwriting any existing copy. Retries
 * transient failures (network drop, 5xx, or an unexpected 409) with a short backoff
 * and a per-attempt timeout — the connected ComfyUI may be a remote/cloud box that
 * briefly hiccups. Clear client errors (e.g. 400/404) fail fast.
 */
async function pushWorkflowToComfy(name: string, json: unknown): Promise<void> {
  const url = `${userdataUrl(name)}?overwrite=true`
  const body = JSON.stringify(json)
  const ATTEMPTS = 3
  let lastError = 'no response'

  for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
    let status: number | null = null
    try {
      const ctrl = new AbortController()
      const timer = setTimeout(() => ctrl.abort(), 10_000)
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        signal: ctrl.signal,
      })
      clearTimeout(timer)
      if (res.ok) return
      status = res.status
      lastError = `ComfyUI returned ${res.status}`
    } catch (e) {
      lastError = e instanceof Error ? e.message : String(e)
    }
    // A clear client error (bad request, not found, …) won't fix itself — stop. The
    // exception is 409: with overwrite=true it signals a stale lock, so retry it.
    if (status !== null && status < 500 && status !== 409) break
    if (attempt < ATTEMPTS) await delay(300 * attempt)
  }

  throw new Error(
    `Could not save the workflow to ComfyUI (${lastError}). Make sure it's running and recent enough to support the userdata API.`,
  )
}

/**
 * Set the image of each LoadImage node (in graph order) to the matching uploaded
 * input filename, so the frame's displayed input is actually used by the workflow.
 * Mutates `workflow` in place; returns true if anything changed. Tolerant of the two
 * widgets_values shapes ComfyUI has shipped (array vs object).
 */
function injectInputsIntoWorkflow(workflow: unknown, names: string[]): boolean {
  if (names.length === 0 || !workflow || typeof workflow !== 'object') return false
  const nodes = (workflow as { nodes?: unknown }).nodes
  if (!Array.isArray(nodes)) return false
  let next = 0
  let changed = false
  for (const node of nodes) {
    if (next >= names.length) break
    if (!node || typeof node !== 'object' || (node as { type?: unknown }).type !== 'LoadImage')
      continue
    const wv = (node as { widgets_values?: unknown }).widgets_values
    if (Array.isArray(wv)) {
      if (wv[0] !== names[next]) {
        wv[0] = names[next]
        changed = true
      }
      next++
    } else if (wv && typeof wv === 'object') {
      ;(wv as Record<string, unknown>).image = names[next]
      changed = true
      next++
    }
  }
  return changed
}

/** Node types the seed workflow ships with — they don't make a workflow "built". */
const SEED_NODE_TYPES = new Set(['Note', 'MarkdownNote', 'LoadImage', 'PreviewImage'])

/** The serialized graph's nodes array, or null if this isn't a workflow-shaped object. */
function workflowNodes(workflow: unknown): unknown[] | null {
  if (!workflow || typeof workflow !== 'object') return null
  const nodes = (workflow as { nodes?: unknown }).nodes
  return Array.isArray(nodes) ? nodes : null
}

/**
 * A graph worth persisting: workflow-shaped with at least one node. Guards against
 * clobbering a real saved copy with a blank/failed serialization during page load.
 */
function isMeaningfulWorkflow(workflow: unknown): boolean {
  const nodes = workflowNodes(workflow)
  return !!nodes && nodes.length > 0
}

/**
 * Has the user actually built this workflow, vs. it still being the seed? True when the
 * graph contains any node beyond the seed's boilerplate (a Note + optional LoadImage) —
 * i.e. something that does real generation work.
 */
function isBuiltWorkflow(workflow: unknown): boolean {
  const nodes = workflowNodes(workflow)
  if (!nodes) return false
  return nodes.some((n) => {
    const t = (n as { type?: unknown })?.type
    return typeof t === 'string' && !SEED_NODE_TYPES.has(t)
  })
}

/**
 * Link/open a frame's ComfyUI workflow, with Inline Studio as the durable source of
 * truth: if we have a local copy, push it to the connected ComfyUI (restores it
 * after an install switch); else adopt ComfyUI's copy if present; else seed a new
 * one. The frame's workflow name is persisted on first link.
 *
 * The frame's inputs (library assets and Preview/flow links → the source frame's
 * hero take) are uploaded to ComfyUI and wired into the workflow's LoadImage nodes,
 * so the input shown on the frame is the one ComfyUI loads.
 */
export async function linkFrameWorkflow(frameId: string): Promise<Frame> {
  const frame = getFrameById(frameId)
  const name = frame.comfyWorkflowName ?? workflowNameFor(frame)
  const linked = frame.comfyWorkflowName ? frame : linkWorkflow(frameId, name)

  // Resolve the workflow JSON: our durable copy, else ComfyUI's, else a fresh seed.
  let workflow = readLocalWorkflow(frameId)
  if (workflow == null) {
    workflow =
      (await getRemoteWorkflow(name)) ?? buildSeedWorkflow(frame.name, frameInputFileNames(frameId))
  }

  // Upload inputs so they're available in ComfyUI, then point LoadImage nodes at them.
  // Best-effort: if upload fails (comfy momentarily down), still push the workflow.
  let uploaded: string[] = []
  try {
    uploaded = await uploadFrameInputs(frameId)
  } catch {
    // ignore — the push below surfaces a real connectivity problem
  }
  injectInputsIntoWorkflow(workflow, uploaded)

  await pushWorkflowToComfy(name, workflow)
  writeLocalWorkflow(frameId, workflow)
  return linked
}

/** Pull the frame's workflow from ComfyUI into the project copy. Returns true if it changed. */
export async function pullWorkflowToProject(frameId: string): Promise<boolean> {
  const frame = getFrameById(frameId)
  if (!frame.comfyWorkflowName) return false
  const remote = await getRemoteWorkflow(frame.comfyWorkflowName)
  if (remote == null) return false
  if (isBuiltWorkflow(remote) && !frame.comfyWorkflowReady) setWorkflowReady(frameId, true)
  const prev = readLocalWorkflow(frameId)
  if (JSON.stringify(prev) === JSON.stringify(remote)) return false
  writeLocalWorkflow(frameId, remote)
  return true
}

/**
 * Capture the LIVE (possibly unsaved) graph the renderer serialized straight off the
 * ComfyUI canvas into the frame's durable copy, and mirror it to ComfyUI's saved file
 * so the named tab and our copy stay in sync. This is what makes "forgot to press Save"
 * non-destructive — Inline Studio owns the graph regardless of ComfyUI's own save action.
 * Ignores blank/non-workflow payloads. Returns the updated frame if anything changed,
 * else null.
 */
export async function saveLiveWorkflow(frameId: string, workflow: unknown): Promise<Frame | null> {
  const frame = getFrameById(frameId)
  if (!frame.comfyWorkflowName) return null
  if (!isMeaningfulWorkflow(workflow)) return null

  const built = isBuiltWorkflow(workflow)
  const becameReady = built && !frame.comfyWorkflowReady
  const prev = readLocalWorkflow(frameId)
  const unchanged = JSON.stringify(prev) === JSON.stringify(workflow)
  if (unchanged && !becameReady) return null

  if (!unchanged) {
    writeLocalWorkflow(frameId, workflow)
    // Mirror to ComfyUI's saved file so re-link / "Open workflow.json" reflect it too.
    // Best-effort: the durable copy above is what matters for not losing work.
    try {
      await pushWorkflowToComfy(frame.comfyWorkflowName, workflow)
    } catch {
      // ignore — comfy momentarily unreachable; the project copy is already saved
    }
  }
  if (becameReady) return setWorkflowReady(frameId, true)
  return getFrameById(frameId)
}

/** Push the project's copy of the frame's workflow to ComfyUI. */
export async function pushWorkflowFromProject(frameId: string): Promise<void> {
  const frame = getFrameById(frameId)
  if (!frame.comfyWorkflowName) return
  const local = readLocalWorkflow(frameId)
  if (local == null) return
  await pushWorkflowToComfy(frame.comfyWorkflowName, local)
}

/**
 * Upload a frame's input assets to ComfyUI via /upload/image so they're available in
 * the LoadImage picker — the cloud-safe alternative to sharing a local input folder.
 * Returns the filenames ComfyUI stored them under. No-op if the frame has no inputs.
 */
export async function uploadFrameInputs(frameId: string): Promise<string[]> {
  const folder = getOpenProjectFolder()
  if (!folder) throw new Error('No project is open.')
  const inputs = frameInputAssetPaths(frameId)
  const uploaded: string[] = []
  for (const { filePath, name } of inputs) {
    const bytes = readFileSync(join(folder, filePath))
    const form = new FormData()
    form.append('image', new Blob([new Uint8Array(bytes)]), name)
    form.append('overwrite', 'true')
    const res = await fetch(`${baseUrl()}/upload/image`, { method: 'POST', body: form })
    if (!res.ok) {
      throw new Error(`Could not upload "${name}" to ComfyUI (${res.status}). Is it reachable?`)
    }
    const json = (await res.json().catch(() => ({}))) as { name?: string }
    uploaded.push(json.name ?? name)
  }
  return uploaded
}

interface OutputFile {
  filename: string
  subfolder?: string
  type?: string
}

interface HistoryEntry {
  outputs: Record<string, Record<string, OutputFile[]>>
}

/**
 * All downloadable files across a history entry's node outputs, in node order,
 * deduped by (filename, subfolder, type). The same file commonly appears under
 * several nodes (e.g. a Preview node and a Save node both reference one temp file),
 * which would otherwise produce duplicate capture tiles / colliding React keys.
 */
function collectOutputs(outputs: HistoryEntry['outputs']): OutputFile[] {
  const files: OutputFile[] = []
  const seen = new Set<string>()
  for (const node of Object.values(outputs)) {
    for (const value of Object.values(node)) {
      if (!Array.isArray(value)) continue
      for (const item of value) {
        if (!item || typeof item.filename !== 'string') continue
        const key = `${item.filename}|${item.subfolder ?? ''}|${item.type ?? ''}`
        if (seen.has(key)) continue
        seen.add(key)
        files.push(item)
      }
    }
  }
  return files
}

function viewUrl(file: OutputFile): string {
  return (
    `${baseUrl()}/view?filename=${encodeURIComponent(file.filename)}` +
    `&subfolder=${encodeURIComponent(file.subfolder ?? '')}` +
    `&type=${encodeURIComponent(file.type ?? 'output')}`
  )
}

/** Download a ComfyUI output file into the project's takes/ and attach it as a take. */
async function saveOutputAsTake(
  frameId: string,
  file: OutputFile,
  promptId: string | null,
): Promise<Take> {
  const bin = await fetch(viewUrl(file))
  if (!bin.ok) throw new Error(`Could not download ComfyUI output (${bin.status}).`)
  const folder = getOpenProjectFolder()
  if (!folder) throw new Error('No project is open.')
  const ext = extname(file.filename) || '.png'
  const relPath = `takes/${randomUUID()}${ext}`
  writeFileSync(join(folder, relPath), Buffer.from(await bin.arrayBuffer()))
  return addTake({
    frameId,
    filePath: relPath,
    kind: kindForExt(ext),
    comfyPromptId: promptId,
    params: {},
  })
}

/**
 * Pull the most recent ComfyUI output and attach it to the frame as a take.
 * Heuristic: the last entry in /history is the latest run.
 */
export async function pullLatestToFrame(frameId: string): Promise<Take> {
  const res = await fetch(`${baseUrl()}/history`)
  if (!res.ok) throw new Error(`Could not read ComfyUI history (${res.status}). Is it running?`)
  const history = (await res.json()) as Record<string, HistoryEntry>
  const ids = Object.keys(history)
  if (ids.length === 0) {
    throw new Error('No ComfyUI output found yet — generate something in ComfyUI first.')
  }
  const promptId = ids[ids.length - 1]
  const file = collectOutputs(history[promptId].outputs)[0]
  if (!file) throw new Error('The latest ComfyUI run produced no downloadable output.')
  return saveOutputAsTake(frameId, file, promptId)
}

/** The most recent ComfyUI run and all its output files (for the capture strip). */
export async function latestRun(): Promise<ComfyRun | null> {
  const res = await fetch(`${baseUrl()}/history`)
  if (!res.ok) return null
  const history = (await res.json()) as Record<string, HistoryEntry>
  const ids = Object.keys(history)
  if (ids.length === 0) return null
  const promptId = ids[ids.length - 1]
  const outputs: ComfyOutput[] = collectOutputs(history[promptId].outputs).map((f) => ({
    filename: f.filename,
    subfolder: f.subfolder ?? '',
    type: f.type ?? 'output',
    kind: kindForExt(extname(f.filename)),
    url: viewUrl(f),
  }))
  return { promptId, outputs }
}

/** Download a specific ComfyUI output (from the capture strip) into the frame. */
export async function captureOutput(frameId: string, output: ComfyOutput): Promise<Take> {
  return saveOutputAsTake(
    frameId,
    { filename: output.filename, subfolder: output.subfolder, type: output.type },
    null,
  )
}
