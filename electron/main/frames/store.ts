/**
 * Frames: the timeline's atomic unit. Each frame has an Input (an imported asset)
 * and a history of generated Takes; its hero take is the Output. For now all frames
 * live in a single auto-created default sequence (sequences aren't exposed in the UI yet).
 */
import { randomUUID } from 'node:crypto'
import { join } from 'node:path'
import { existsSync, unlinkSync, mkdirSync, copyFileSync } from 'node:fs'
import type { Frame, Take, FrameInput, FrameKind, AssetKind } from '@shared/types'
import { getNodeDef, listNodeDefs } from '@shared/nodes/registry'
import { defaultParams } from '@shared/nodes/types'
import { takeWaveformPath } from '@shared/media'
import { getDb, getOpenProjectFolder } from '../db'
import { importViaDialog } from '../assets/store'
import { ffmpegAvailable, generatePeaks } from '../media/ffmpeg'

interface FrameRow {
  id: string
  sequence_id: string
  name: string
  kind: FrameKind
  position: number
  input_asset_id: string | null
  hero_take_id: string | null
  provider: string | null
  model_id: string | null
  params: string | null
  workflow_template_id: string | null
  comfy_workflow_name: string | null
  comfy_workflow_ready: number
  created_at: number
  updated_at: number
}

interface TakeRow {
  id: string
  frame_id: string
  file_path: string
  kind: AssetKind
  params: string
  comfy_prompt_id: string | null
  created_at: number
}

function parseParams(raw: string | null): Record<string, unknown> {
  if (!raw) return {}
  try {
    const parsed = JSON.parse(raw) as unknown
    return parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : {}
  } catch {
    return {}
  }
}

function rowToFrame(row: FrameRow): Frame {
  return {
    id: row.id,
    sequenceId: row.sequence_id,
    name: row.name,
    kind: row.kind,
    position: row.position,
    inputAssetId: row.input_asset_id,
    heroTakeId: row.hero_take_id,
    provider: row.provider === 'fal' ? 'fal' : row.provider === 'unset' ? 'unset' : 'comfy',
    modelId: row.model_id,
    params: parseParams(row.params),
    workflowTemplateId: row.workflow_template_id,
    comfyWorkflowName: row.comfy_workflow_name,
    comfyWorkflowReady: row.comfy_workflow_ready === 1,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  }
}

function rowToTake(row: TakeRow): Take {
  return {
    id: row.id,
    frameId: row.frame_id,
    filePath: row.file_path,
    kind: row.kind,
    params: JSON.parse(row.params) as Record<string, unknown>,
    comfyPromptId: row.comfy_prompt_id,
    createdAt: row.created_at,
  }
}

interface FrameInputRow {
  id: string
  frame_id: string
  asset_id: string | null
  source_frame_id: string | null
  position: number
}

function rowToFrameInput(row: FrameInputRow): FrameInput {
  return {
    id: row.id,
    frameId: row.frame_id,
    assetId: row.asset_id,
    sourceFrameId: row.source_frame_id,
    position: row.position,
  }
}

function projectId(): string {
  const row = getDb().prepare('SELECT id FROM project LIMIT 1').get() as { id: string } | undefined
  if (!row) throw new Error('No project is open.')
  return row.id
}

/** The single default sequence frames are created in; created on first use. */
function defaultSequenceId(): string {
  const db = getDb()
  const existing = db.prepare('SELECT id FROM sequences ORDER BY position LIMIT 1').get() as
    | { id: string }
    | undefined
  if (existing) return existing.id
  const id = randomUUID()
  db.prepare('INSERT INTO sequences (id, project_id, name, position) VALUES (?, ?, ?, ?)').run(
    id,
    projectId(),
    'Main',
    0,
  )
  return id
}

function getFrame(id: string): Frame {
  const row = getDb().prepare('SELECT * FROM frames WHERE id = ?').get(id) as FrameRow | undefined
  if (!row) throw new Error('Frame not found.')
  return rowToFrame(row)
}

export function listFrames(): Frame[] {
  const seqId = defaultSequenceId()
  const rows = getDb()
    .prepare('SELECT * FROM frames WHERE sequence_id = ? ORDER BY position')
    .all(seqId) as FrameRow[]
  return rows.map(rowToFrame)
}

function createFrame(asset: { id: string; kind: AssetKind }): Frame {
  const db = getDb()
  const seqId = defaultSequenceId()
  const count = (
    db.prepare('SELECT COUNT(*) AS n FROM frames WHERE sequence_id = ?').get(seqId) as { n: number }
  ).n
  const now = Date.now()
  // Bound params (no comfy_workflow_ready: it defaults to 0 in the schema). New frames start
  // `unset` — a chooser node the user resolves to ComfyUI or fal (see setProvider).
  const frame = {
    id: randomUUID(),
    sequenceId: seqId,
    name: String(count + 1),
    kind: asset.kind,
    position: count,
    inputAssetId: asset.id,
    heroTakeId: null,
    provider: 'unset',
    workflowTemplateId: null,
    comfyWorkflowName: null,
    createdAt: now,
    updatedAt: now,
  }
  db.prepare(
    `INSERT INTO frames
       (id, sequence_id, name, kind, position, input_asset_id, hero_take_id, provider, workflow_template_id, comfy_workflow_name, created_at, updated_at)
     VALUES (@id, @sequenceId, @name, @kind, @position, @inputAssetId, @heroTakeId, @provider, @workflowTemplateId, @comfyWorkflowName, @createdAt, @updatedAt)`,
  ).run(frame)
  // Every frame starts with exactly one input (the asset it was created from).
  db.prepare('INSERT INTO frame_inputs (id, frame_id, asset_id, position) VALUES (?, ?, ?, 0)').run(
    randomUUID(),
    frame.id,
    asset.id,
  )
  return { ...frame, provider: 'unset', modelId: null, params: {}, comfyWorkflowReady: false }
}

function assetById(id: string): { id: string; kind: AssetKind } {
  const row = getDb().prepare('SELECT id, kind FROM assets WHERE id = ?').get(id) as
    | { id: string; kind: AssetKind }
    | undefined
  if (!row) throw new Error('Asset not found.')
  return row
}

export function addFromAsset(assetId: string): Frame {
  return createFrame(assetById(assetId))
}

/** Create an empty frame (no inputs yet) — fed later by a dropped asset or a flow link. */
export function createEmptyFrame(): Frame {
  const db = getDb()
  const seqId = defaultSequenceId()
  const count = (
    db.prepare('SELECT COUNT(*) AS n FROM frames WHERE sequence_id = ?').get(seqId) as { n: number }
  ).n
  const now = Date.now()
  // Bound params (no comfy_workflow_ready: it defaults to 0 in the schema). Starts `unset` — a
  // chooser node the user resolves to ComfyUI or fal (see setProvider).
  const frame = {
    id: randomUUID(),
    sequenceId: seqId,
    name: String(count + 1),
    kind: 'image' as FrameKind,
    position: count,
    inputAssetId: null,
    heroTakeId: null,
    provider: 'unset',
    workflowTemplateId: null,
    comfyWorkflowName: null,
    createdAt: now,
    updatedAt: now,
  }
  db.prepare(
    `INSERT INTO frames
       (id, sequence_id, name, kind, position, input_asset_id, hero_take_id, provider, workflow_template_id, comfy_workflow_name, created_at, updated_at)
     VALUES (@id, @sequenceId, @name, @kind, @position, @inputAssetId, @heroTakeId, @provider, @workflowTemplateId, @comfyWorkflowName, @createdAt, @updatedAt)`,
  ).run(frame)
  return { ...frame, provider: 'unset', modelId: null, params: {}, comfyWorkflowReady: false }
}

/**
 * Create a fal generation frame for a registry model. Its `kind` comes from the node def's
 * output kind (image/video/audio), `params` are seeded from the def's field defaults, and it
 * starts with no inputs (fed later by a dropped asset or an upstream flow link).
 */
export function createFalFrame(modelId: string): Frame {
  const def = getNodeDef(modelId)
  if (!def) throw new Error(`Unknown generation model: ${modelId}`)
  const db = getDb()
  const seqId = defaultSequenceId()
  const count = (
    db.prepare('SELECT COUNT(*) AS n FROM frames WHERE sequence_id = ?').get(seqId) as { n: number }
  ).n
  const now = Date.now()
  const params = defaultParams(def)
  const frame = {
    id: randomUUID(),
    sequenceId: seqId,
    name: def.title,
    kind: def.outputKind as FrameKind,
    position: count,
    inputAssetId: null,
    heroTakeId: null,
    provider: 'fal',
    modelId,
    params: JSON.stringify(params),
    workflowTemplateId: null,
    comfyWorkflowName: null,
    createdAt: now,
    updatedAt: now,
  }
  db.prepare(
    `INSERT INTO frames
       (id, sequence_id, name, kind, position, input_asset_id, hero_take_id, provider, model_id, params, workflow_template_id, comfy_workflow_name, created_at, updated_at)
     VALUES (@id, @sequenceId, @name, @kind, @position, @inputAssetId, @heroTakeId, @provider, @modelId, @params, @workflowTemplateId, @comfyWorkflowName, @createdAt, @updatedAt)`,
  ).run(frame)
  return {
    ...frame,
    provider: 'fal',
    modelId,
    params,
    comfyWorkflowReady: false,
  }
}

/**
 * Switch a fal frame to a different registry model: resets its params to the new model's defaults
 * and updates its output kind (image/video/audio). Keeps the frame's inputs + take history.
 */
export function setModel(frameId: string, modelId: string): Frame {
  const def = getNodeDef(modelId)
  if (!def) throw new Error(`Unknown generation model: ${modelId}`)
  getFrame(frameId)
  getDb()
    .prepare('UPDATE frames SET model_id = ?, params = ?, kind = ?, updated_at = ? WHERE id = ?')
    .run(modelId, JSON.stringify(defaultParams(def)), def.outputKind, Date.now(), frameId)
  return getFrame(frameId)
}

/**
 * Resolve an `unset` chooser frame to a generation engine (or switch an existing frame).
 * `fal` seeds the given model's defaults (params + output kind), like createFalFrame but on an
 * existing frame; `comfy` clears any fal model and keeps the frame's kind. Keeps inputs + takes.
 */
export function setProvider(frameId: string, provider: 'comfy' | 'fal', modelId?: string): Frame {
  getFrame(frameId)
  const db = getDb()
  const now = Date.now()
  if (provider === 'fal') {
    const id = modelId ?? listNodeDefs()[0]?.id
    const def = id ? getNodeDef(id) : undefined
    if (!def) throw new Error('No generation model is available.')
    db.prepare(
      'UPDATE frames SET provider = ?, model_id = ?, params = ?, kind = ?, updated_at = ? WHERE id = ?',
    ).run('fal', def.id, JSON.stringify(defaultParams(def)), def.outputKind, now, frameId)
  } else {
    db.prepare(
      "UPDATE frames SET provider = 'comfy', model_id = NULL, updated_at = ? WHERE id = ?",
    ).run(now, frameId)
  }
  return getFrame(frameId)
}

/** Persist a fal frame's param values (from the GenNode widgets). Returns the updated frame. */
export function setFalParams(frameId: string, params: Record<string, unknown>): Frame {
  getFrame(frameId)
  getDb()
    .prepare('UPDATE frames SET params = ?, updated_at = ? WHERE id = ?')
    .run(JSON.stringify(params ?? {}), Date.now(), frameId)
  return getFrame(frameId)
}

/** The fal generation fields of a frame — used by the executor. */
export function getFalFrame(frameId: string): {
  provider: 'comfy' | 'fal' | 'core' | 'unset'
  modelId: string | null
  params: Record<string, unknown>
} {
  const frame = getFrame(frameId)
  return { provider: frame.provider, modelId: frame.modelId, params: frame.params }
}

/**
 * Link another frame's output as an input (the refine/flow connector). Resolves to
 * `sourceFrameId`'s hero take at generate time. Deduped; self-links are rejected.
 */
export function addSourceInput(frameId: string, sourceFrameId: string): FrameInput {
  getFrame(frameId)
  getFrame(sourceFrameId)
  if (frameId === sourceFrameId) throw new Error('A frame cannot use its own output as input.')
  const existing = frameInputRows(frameId)
  const dup = existing.find((r) => r.source_frame_id === sourceFrameId)
  if (dup) return rowToFrameInput(dup)
  const input: FrameInput = {
    id: randomUUID(),
    frameId,
    assetId: null,
    sourceFrameId,
    position: existing.length,
  }
  getDb()
    .prepare(
      'INSERT INTO frame_inputs (id, frame_id, asset_id, source_frame_id, position) VALUES (?, ?, NULL, ?, ?)',
    )
    .run(input.id, input.frameId, sourceFrameId, input.position)
  return input
}

const basename = (p: string): string => p.split('/').pop() ?? p

/**
 * Resolve a frame's output file to feed downstream (director timeline) or upload to ComfyUI:
 * its hero take, else its newest take, else — when it has never been generated — its first
 * input, following a flow link (`source_frame_id`) up the chain to a real asset/take. So a
 * frame fed only by another frame's output still resolves to a file even before that source has
 * generated (as long as SOME upstream asset/take exists). Returns the relative path + kind +
 * basename, or null when nothing resolves. `seen` guards against cyclic flow links.
 */
function resolveFrameFile(
  frameId: string,
  seen: Set<string> = new Set(),
): { filePath: string; kind: AssetKind; name: string } | null {
  if (seen.has(frameId)) return null
  seen.add(frameId)
  const db = getDb()
  const fr = db.prepare('SELECT hero_take_id FROM frames WHERE id = ?').get(frameId) as
    | { hero_take_id: string | null }
    | undefined
  let tk = fr?.hero_take_id
    ? (db.prepare('SELECT file_path, kind FROM takes WHERE id = ?').get(fr.hero_take_id) as
        | { file_path: string; kind: AssetKind }
        | undefined)
    : undefined
  if (!tk) {
    tk = db
      .prepare(
        'SELECT file_path, kind FROM takes WHERE frame_id = ? ORDER BY created_at DESC LIMIT 1',
      )
      .get(frameId) as { file_path: string; kind: AssetKind } | undefined
  }
  if (tk) return { filePath: tk.file_path, kind: tk.kind, name: basename(tk.file_path) }
  // Never generated: resolve the first input — an imported asset directly, or an upstream frame's
  // output followed up the flow chain (so a linked input still uploads before its source has run).
  for (const row of frameInputRows(frameId)) {
    if (row.asset_id) {
      const a = db.prepare('SELECT file_path, kind FROM assets WHERE id = ?').get(row.asset_id) as
        | { file_path: string; kind: AssetKind }
        | undefined
      if (a) return { filePath: a.file_path, kind: a.kind, name: basename(a.file_path) }
    } else if (row.source_frame_id) {
      const up = resolveFrameFile(row.source_frame_id, seen)
      if (up) return up
    }
  }
  return null
}

/**
 * The file path + basename of a frame's output to feed downstream — the hero take, the newest
 * take, or (when never generated) its first resolvable input (asset or upstream flow link).
 */
function heroTakeFile(frameId: string): { filePath: string; name: string } | null {
  const f = resolveFrameFile(frameId)
  return f ? { filePath: f.filePath, name: f.name } : null
}

/**
 * Resolve a frame's output (hero take, else newest take, else its first resolvable input,
 * following flow links) to a relative path + kind, for the director timeline.
 */
export function resolveFrameOutput(frameId: string): { filePath: string; kind: AssetKind } | null {
  const f = resolveFrameFile(frameId)
  return f ? { filePath: f.filePath, kind: f.kind } : null
}

export async function importAsFrames(): Promise<Frame[]> {
  const assets = await importViaDialog(null)
  return assets.map((a) => createFrame({ id: a.id, kind: a.kind }))
}

export function renameFrame(id: string, name: string): Frame {
  const trimmed = name.trim()
  if (!trimmed) throw new Error('Frame name is required.')
  getFrame(id)
  getDb()
    .prepare('UPDATE frames SET name = ?, updated_at = ? WHERE id = ?')
    .run(trimmed, Date.now(), id)
  return getFrame(id)
}

export function reorderFrames(orderedIds: string[]): void {
  const db = getDb()
  const now = Date.now()
  const stmt = db.prepare('UPDATE frames SET position = ?, updated_at = ? WHERE id = ?')
  const tx = db.transaction(() => {
    orderedIds.forEach((id, index) => stmt.run(index, now, id))
  })
  tx()
}

export function deleteFrame(id: string): void {
  const db = getDb()
  const tx = db.transaction(() => {
    db.prepare('DELETE FROM takes WHERE frame_id = ?').run(id)
    db.prepare('DELETE FROM frame_inputs WHERE frame_id = ?').run(id)
    // Drop any canvas node(s) for this frame so they don't orphan, plus their edges.
    const items = db
      .prepare("SELECT id FROM moodboard_items WHERE frame_id = ? AND type = 'frame'")
      .all(id) as Array<{ id: string }>
    for (const item of items) {
      db.prepare('DELETE FROM moodboard_connectors WHERE from_item_id = ? OR to_item_id = ?').run(
        item.id,
        item.id,
      )
    }
    db.prepare("DELETE FROM moodboard_items WHERE frame_id = ? AND type = 'frame'").run(id)
    db.prepare('DELETE FROM frames WHERE id = ?').run(id)
  })
  tx()
  // Remove the durable workflow copy (best-effort; outside the DB transaction).
  const folder = getOpenProjectFolder()
  if (folder) {
    const wf = join(folder, 'workflows', `${id}.json`)
    if (existsSync(wf)) unlinkSync(wf)
  }
}

/** All frame inputs across the project (renderer groups by frameId). */
export function listInputs(): FrameInput[] {
  const rows = getDb()
    .prepare('SELECT * FROM frame_inputs ORDER BY frame_id, position')
    .all() as FrameInputRow[]
  return rows.map(rowToFrameInput)
}

function frameInputRows(frameId: string): FrameInputRow[] {
  return getDb()
    .prepare('SELECT * FROM frame_inputs WHERE frame_id = ? ORDER BY position')
    .all(frameId) as FrameInputRow[]
}

export function addInput(frameId: string, assetId: string): FrameInput {
  getFrame(frameId)
  const existing = frameInputRows(frameId)
  const dup = existing.find((r) => r.asset_id === assetId)
  if (dup) return rowToFrameInput(dup)
  const input: FrameInput = {
    id: randomUUID(),
    frameId,
    assetId,
    sourceFrameId: null,
    position: existing.length,
  }
  getDb()
    .prepare('INSERT INTO frame_inputs (id, frame_id, asset_id, position) VALUES (?, ?, ?, ?)')
    .run(input.id, input.frameId, input.assetId, input.position)
  return input
}

/**
 * Append several library assets as inputs in a single transaction (skips already-present ones),
 * assigning sequential positions. One atomic call avoids the position race when the renderer used
 * to fire per-asset `addInput`s in parallel. Returns the rows actually inserted, in order.
 */
export function addInputs(frameId: string, assetIds: string[]): FrameInput[] {
  getFrame(frameId)
  const db = getDb()
  const added: FrameInput[] = []
  const ins = db.prepare(
    'INSERT INTO frame_inputs (id, frame_id, asset_id, position) VALUES (?, ?, ?, ?)',
  )
  db.transaction(() => {
    const existing = frameInputRows(frameId)
    const have = new Set(existing.map((r) => r.asset_id).filter((x): x is string => !!x))
    let pos = existing.length
    for (const assetId of assetIds) {
      if (have.has(assetId)) continue
      have.add(assetId)
      const input: FrameInput = {
        id: randomUUID(),
        frameId,
        assetId,
        sourceFrameId: null,
        position: pos++,
      }
      ins.run(input.id, frameId, assetId, input.position)
      added.push(input)
    }
  })()
  return added
}

/** Remove an input by its library asset id (kept for the Frame Inspector). Allows removing all. */
export function removeInput(frameId: string, assetId: string): void {
  getDb()
    .prepare('DELETE FROM frame_inputs WHERE frame_id = ? AND asset_id = ?')
    .run(frameId, assetId)
}

/** Remove one input by its row id — works for asset inputs AND flow (`source_frame_id`) links. */
export function removeInputById(frameId: string, inputId: string): void {
  getDb().prepare('DELETE FROM frame_inputs WHERE frame_id = ? AND id = ?').run(frameId, inputId)
}

export function reorderInputs(frameId: string, orderedAssetIds: string[]): void {
  const db = getDb()
  const stmt = db.prepare(
    'UPDATE frame_inputs SET position = ? WHERE frame_id = ? AND asset_id = ?',
  )
  const tx = db.transaction(() => {
    orderedAssetIds.forEach((assetId, index) => stmt.run(index, frameId, assetId))
  })
  tx()
}

/**
 * Resolve a frame's inputs to file paths + basenames, in order. Asset inputs map to
 * their library file; flow inputs (`source_frame_id`) map to the source frame's hero
 * take. Rows that don't resolve (e.g. a flow source with no hero take yet) are skipped.
 */
export function frameInputAssetPaths(frameId: string): Array<{ filePath: string; name: string }> {
  const out: Array<{ filePath: string; name: string }> = []
  for (const row of frameInputRows(frameId)) {
    if (row.asset_id) {
      const asset = getDb()
        .prepare('SELECT file_path FROM assets WHERE id = ?')
        .get(row.asset_id) as { file_path: string } | undefined
      if (asset)
        out.push({
          filePath: asset.file_path,
          name: asset.file_path.split('/').pop() ?? asset.file_path,
        })
    } else if (row.source_frame_id) {
      const hero = heroTakeFile(row.source_frame_id)
      if (hero) out.push(hero)
    }
  }
  return out
}

/** Input filenames (basenames) of a frame, in order — used to seed the workflow Note. */
export function frameInputFileNames(frameId: string): string[] {
  return frameInputAssetPaths(frameId).map((p) => p.name)
}

/** All takes across the project (renderer groups by frameId). */
export function listAllTakes(): Take[] {
  const rows = getDb()
    .prepare('SELECT * FROM takes ORDER BY frame_id, created_at DESC')
    .all() as TakeRow[]
  return rows.map(rowToTake)
}

export function deleteTake(takeId: string): void {
  const db = getDb()
  const take = db.prepare('SELECT * FROM takes WHERE id = ?').get(takeId) as TakeRow | undefined
  if (!take) return
  const folder = getOpenProjectFolder()
  const tx = db.transaction(() => {
    db.prepare('UPDATE frames SET hero_take_id = NULL WHERE hero_take_id = ?').run(takeId)
    db.prepare('DELETE FROM takes WHERE id = ?').run(takeId)
  })
  tx()
  // Best-effort: remove the generated file from disk.
  if (folder) {
    const abs = join(folder, take.file_path)
    if (existsSync(abs)) {
      try {
        unlinkSync(abs)
      } catch {
        // ignore
      }
    }
  }
}

export function setHero(id: string, takeId: string | null): Frame {
  getFrame(id)
  getDb()
    .prepare('UPDATE frames SET hero_take_id = ?, updated_at = ? WHERE id = ?')
    .run(takeId, Date.now(), id)
  return getFrame(id)
}

export function listTakes(frameId: string): Take[] {
  const rows = getDb()
    .prepare('SELECT * FROM takes WHERE frame_id = ? ORDER BY created_at DESC')
    .all(frameId) as TakeRow[]
  return rows.map(rowToTake)
}

/** The hero (Output) take of every frame that has one — one query for the timeline. */
export function heroTakes(): Take[] {
  const rows = getDb()
    .prepare('SELECT t.* FROM takes t JOIN frames s ON s.hero_take_id = t.id')
    .all() as TakeRow[]
  return rows.map(rowToTake)
}

/** Insert a generated take for a frame and make it the hero (Output). */
export function addTake(input: {
  frameId: string
  filePath: string
  kind: AssetKind
  comfyPromptId: string | null
  params: Record<string, unknown>
}): Take {
  getFrame(input.frameId) // ensure exists
  const id = randomUUID()
  const now = Date.now()
  getDb()
    .prepare(
      `INSERT INTO takes (id, frame_id, file_path, kind, params, comfy_prompt_id, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      id,
      input.frameId,
      input.filePath,
      input.kind,
      JSON.stringify(input.params),
      input.comfyPromptId,
      now,
    )
  setHero(input.frameId, id)
  // Background: render a waveform for audio takes (by convention, keyed on take id).
  if (input.kind === 'audio' && ffmpegAvailable()) {
    const folder = getOpenProjectFolder()
    if (folder) {
      void generatePeaks(join(folder, input.filePath), join(folder, takeWaveformPath(id))).catch(
        () => {
          /* ignore — the player still works without a waveform */
        },
      )
    }
  }
  return {
    id,
    frameId: input.frameId,
    filePath: input.filePath,
    kind: input.kind,
    params: input.params,
    comfyPromptId: input.comfyPromptId,
    createdAt: now,
  }
}

/** Read a single frame (throws if missing). */
export function getFrameById(id: string): Frame {
  return getFrame(id)
}

/** Record the ComfyUI workflow this frame is linked to. */
export function linkWorkflow(frameId: string, name: string): Frame {
  getFrame(frameId)
  getDb()
    .prepare('UPDATE frames SET comfy_workflow_name = ?, updated_at = ? WHERE id = ?')
    .run(name, Date.now(), frameId)
  return getFrame(frameId)
}

/** Detach the frame's ComfyUI workflow link (keeps the local copy for re-linking). */
export function unlinkWorkflow(frameId: string): Frame {
  getFrame(frameId)
  getDb()
    .prepare(
      'UPDATE frames SET comfy_workflow_name = NULL, comfy_workflow_ready = 0, updated_at = ? WHERE id = ?',
    )
    .run(Date.now(), frameId)
  return getFrame(frameId)
}

/** Mark whether a real (non-seed) workflow has been captured for this frame. */
export function setWorkflowReady(frameId: string, ready: boolean): Frame {
  getFrame(frameId)
  getDb()
    .prepare('UPDATE frames SET comfy_workflow_ready = ?, updated_at = ? WHERE id = ?')
    .run(ready ? 1 : 0, Date.now(), frameId)
  return getFrame(frameId)
}

/**
 * Duplicate a frame: copies its inputs and (if present) its stored workflow JSON to a
 * new frame. The clone starts unlinked (`comfy_workflow_name` null) so linking it
 * creates its own ComfyUI workflow seeded from the copied JSON.
 */
export function cloneFrame(frameId: string): Frame {
  const src = getFrame(frameId)
  const db = getDb()
  const seqId = defaultSequenceId()
  const count = (
    db.prepare('SELECT COUNT(*) AS n FROM frames WHERE sequence_id = ?').get(seqId) as { n: number }
  ).n
  const now = Date.now()
  // Bound params (no comfy_workflow_ready: it defaults to 0 in the schema). The clone
  // starts unlinked, so it isn't "ready" until it's linked and built itself.
  const clone = {
    id: randomUUID(),
    sequenceId: seqId,
    name: `${src.name} copy`,
    kind: src.kind,
    position: count,
    inputAssetId: src.inputAssetId,
    heroTakeId: null,
    provider: src.provider,
    modelId: src.modelId,
    params: JSON.stringify(src.params ?? {}),
    workflowTemplateId: src.workflowTemplateId,
    comfyWorkflowName: null,
    createdAt: now,
    updatedAt: now,
  }
  const tx = db.transaction(() => {
    db.prepare(
      `INSERT INTO frames
         (id, sequence_id, name, kind, position, input_asset_id, hero_take_id, provider, model_id, params, workflow_template_id, comfy_workflow_name, created_at, updated_at)
       VALUES (@id, @sequenceId, @name, @kind, @position, @inputAssetId, @heroTakeId, @provider, @modelId, @params, @workflowTemplateId, @comfyWorkflowName, @createdAt, @updatedAt)`,
    ).run(clone)
    const inputs = db
      .prepare('SELECT asset_id, source_frame_id, position FROM frame_inputs WHERE frame_id = ?')
      .all(frameId) as Array<{
      asset_id: string | null
      source_frame_id: string | null
      position: number
    }>
    const ins = db.prepare(
      'INSERT INTO frame_inputs (id, frame_id, asset_id, source_frame_id, position) VALUES (?, ?, ?, ?, ?)',
    )
    for (const i of inputs) {
      ins.run(randomUUID(), clone.id, i.asset_id, i.source_frame_id, i.position)
    }
  })
  tx()
  // Copy the durable workflow JSON, if any, so the clone can be linked independently.
  const folder = getOpenProjectFolder()
  if (folder) {
    const srcWf = join(folder, 'workflows', `${frameId}.json`)
    if (existsSync(srcWf)) {
      mkdirSync(join(folder, 'workflows'), { recursive: true })
      copyFileSync(srcWf, join(folder, 'workflows', `${clone.id}.json`))
    }
  }
  return getFrame(clone.id)
}
