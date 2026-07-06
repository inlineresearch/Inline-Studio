/**
 * Moodboard persistence: items (assets + text) and connectors for the open
 * project. The board shares the project's asset library — placing or importing
 * media here references/creates rows in the same `assets` table.
 */
import { randomUUID } from 'node:crypto'
import type {
  MoodboardItem,
  MoodboardItemType,
  MoodboardConnector,
  MoodboardSnapshot,
  MoodboardItemData,
  TextItemData,
} from '@shared/types'
import type { MoodboardItemPatch } from '@shared/ipc'
import { getDb } from '../db'
import { importViaDialog } from '../assets/store'
import {
  addFromAsset as createFrameFromAsset,
  createEmptyFrame,
  createFalFrame,
} from '../frames/store'

const DEFAULT_SIZE: Record<'image' | 'video' | 'audio', { w: number; h: number }> = {
  image: { w: 320, h: 180 },
  video: { w: 360, h: 203 },
  audio: { w: 320, h: 80 },
}

const DEFAULT_TEXT: TextItemData = {
  text: 'Text',
  fontSize: 18,
  bold: false,
  italic: false,
  underline: false,
  color: '#e4e4e7',
  align: 'left',
}

interface ItemRow {
  id: string
  project_id: string
  type: MoodboardItemType
  asset_id: string | null
  frame_id: string | null
  parent_id: string | null
  data: string | null
  x: number
  y: number
  width: number
  height: number
  rotation: number
  z_index: number
  created_at: number
  updated_at: number
}

interface ConnectorRow {
  id: string
  project_id: string
  from_item_id: string
  to_item_id: string
  label: string | null
  data: string | null
  created_at: number
}

function rowToItem(row: ItemRow): MoodboardItem {
  return {
    id: row.id,
    projectId: row.project_id,
    type: row.type,
    assetId: row.asset_id,
    frameId: row.frame_id,
    parentId: row.parent_id,
    data: row.data ? (JSON.parse(row.data) as MoodboardItemData) : {},
    x: row.x,
    y: row.y,
    width: row.width,
    height: row.height,
    rotation: row.rotation,
    zIndex: row.z_index,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  }
}

function rowToConnector(row: ConnectorRow): MoodboardConnector {
  return {
    id: row.id,
    projectId: row.project_id,
    fromItemId: row.from_item_id,
    toItemId: row.to_item_id,
    label: row.label,
    data: row.data ? (JSON.parse(row.data) as Record<string, unknown>) : {},
    createdAt: row.created_at,
  }
}

function projectId(): string {
  const row = getDb().prepare('SELECT id FROM project LIMIT 1').get() as { id: string } | undefined
  if (!row) throw new Error('No project is open.')
  return row.id
}

function getItem(id: string): MoodboardItem {
  const row = getDb().prepare('SELECT * FROM moodboard_items WHERE id = ?').get(id) as
    | ItemRow
    | undefined
  if (!row) throw new Error('Moodboard item not found.')
  return rowToItem(row)
}

/** Read a moodboard item by id (e.g. a director node's settings). */
export function getMoodboardItem(id: string): MoodboardItem {
  return getItem(id)
}

/** All connectors whose target is `itemId` (e.g. inputs wired into a director node). */
export function listConnectorsTo(itemId: string): MoodboardConnector[] {
  const rows = getDb()
    .prepare('SELECT * FROM moodboard_connectors WHERE to_item_id = ?')
    .all(itemId) as ConnectorRow[]
  return rows.map(rowToConnector)
}

/** Read all moodboard items (for resolving connector sources). */
export function listItems(): MoodboardItem[] {
  return (getDb().prepare('SELECT * FROM moodboard_items').all() as ItemRow[]).map(rowToItem)
}

function nextZIndex(): number {
  const row = getDb().prepare('SELECT MAX(z_index) AS z FROM moodboard_items').get() as {
    z: number | null
  }
  return (row.z ?? 0) + 1
}

function insertItem(item: MoodboardItem): MoodboardItem {
  getDb()
    .prepare(
      `INSERT INTO moodboard_items
         (id, project_id, type, asset_id, frame_id, parent_id, data, x, y, width, height, rotation, z_index, created_at, updated_at)
       VALUES (@id, @projectId, @type, @assetId, @frameId, @parentId, @data, @x, @y, @width, @height, @rotation, @zIndex, @createdAt, @updatedAt)`,
    )
    .run({ ...item, data: JSON.stringify(item.data) })
  return item
}

export function listBoard(): MoodboardSnapshot {
  const db = getDb()
  const items = (db.prepare('SELECT * FROM moodboard_items').all() as ItemRow[]).map(rowToItem)
  const connectors = (db.prepare('SELECT * FROM moodboard_connectors').all() as ConnectorRow[]).map(
    rowToConnector,
  )
  return { items, connectors }
}

/**
 * Replace the whole board with the given items + connectors (undo/redo restore).
 * Runs in one transaction: clear the project's items/connectors, then re-insert the
 * snapshot preserving ids. Items reference assets/frames by id, which persist
 * independently, so restoring the board re-links them.
 */
export function replaceBoard(items: MoodboardItem[], connectors: MoodboardConnector[]): void {
  const db = getDb()
  const pid = projectId()
  const insItem = db.prepare(
    `INSERT INTO moodboard_items
       (id, project_id, type, asset_id, frame_id, parent_id, data, x, y, width, height, rotation, z_index, created_at, updated_at)
     VALUES (@id, @projectId, @type, @assetId, @frameId, @parentId, @data, @x, @y, @width, @height, @rotation, @zIndex, @createdAt, @updatedAt)`,
  )
  const insConn = db.prepare(
    `INSERT INTO moodboard_connectors (id, project_id, from_item_id, to_item_id, label, data, created_at)
     VALUES (@id, @projectId, @fromItemId, @toItemId, @label, @data, @createdAt)`,
  )
  db.transaction(() => {
    db.prepare('DELETE FROM moodboard_connectors WHERE project_id = ?').run(pid)
    db.prepare('DELETE FROM moodboard_items WHERE project_id = ?').run(pid)
    for (const it of items)
      insItem.run({ ...it, projectId: pid, data: JSON.stringify(it.data ?? {}) })
    for (const c of connectors)
      insConn.run({ ...c, projectId: pid, data: JSON.stringify(c.data ?? {}) })
  })()
}

export function addAssetItem(assetId: string, x: number, y: number): MoodboardItem {
  const asset = getDb().prepare('SELECT id, kind FROM assets WHERE id = ?').get(assetId) as
    | { id: string; kind: 'image' | 'video' | 'audio' }
    | undefined
  if (!asset) throw new Error('Asset not found.')

  const size = DEFAULT_SIZE[asset.kind]
  const now = Date.now()
  return insertItem({
    id: randomUUID(),
    projectId: projectId(),
    type: 'asset',
    assetId,
    frameId: null,
    parentId: null,
    data: {},
    x,
    y,
    width: size.w,
    height: size.h,
    rotation: 0,
    zIndex: nextZIndex(),
    createdAt: now,
    updatedAt: now,
  })
}

export function addTextItem(x: number, y: number): MoodboardItem {
  const now = Date.now()
  return insertItem({
    id: randomUUID(),
    projectId: projectId(),
    type: 'text',
    assetId: null,
    frameId: null,
    parentId: null,
    data: { text: { ...DEFAULT_TEXT } },
    x,
    y,
    width: 200,
    height: 60,
    rotation: 0,
    zIndex: nextZIndex(),
    createdAt: now,
    updatedAt: now,
  })
}

/** Place an existing frame as a node on the canvas. */
export function addFrameItem(frameId: string, x: number, y: number): MoodboardItem {
  const now = Date.now()
  return insertItem({
    id: randomUUID(),
    projectId: projectId(),
    type: 'frame',
    assetId: null,
    frameId,
    parentId: null,
    data: {},
    x,
    y,
    width: 220,
    height: 200,
    rotation: 0,
    zIndex: nextZIndex(),
    createdAt: now,
    updatedAt: now,
  })
}

/** Create a frame from a library asset AND place a frame node on the canvas. */
export function addFrameFromAsset(assetId: string, x: number, y: number): MoodboardItem {
  const frame = createFrameFromAsset(assetId)
  return addFrameItem(frame.id, x, y)
}

/** Create an empty frame AND place a frame node on the canvas. */
export function addEmptyFrame(x: number, y: number): MoodboardItem {
  const frame = createEmptyFrame()
  return addFrameItem(frame.id, x, y)
}

/**
 * Create a fal generation frame for `modelId` AND place its node on the canvas. It's a regular
 * `type:'frame'` item (so it inherits the frame delete/flow-link machinery); the renderer renders
 * it as a GenNode because its backing frame is `provider:'fal'`. Sized taller for param widgets.
 */
export function addGenNode(modelId: string, x: number, y: number): MoodboardItem {
  const frame = createFalFrame(modelId)
  const now = Date.now()
  return insertItem({
    id: randomUUID(),
    projectId: projectId(),
    type: 'frame',
    assetId: null,
    frameId: frame.id,
    parentId: null,
    data: {},
    x,
    y,
    // Portrait by default — the output preview is the hero, params tuck underneath.
    width: 240,
    height: 380,
    rotation: 0,
    zIndex: nextZIndex(),
    createdAt: now,
    updatedAt: now,
  })
}

/** Add a resizable layer group container (frames can be dropped inside it). */
export function addLayer(x: number, y: number): MoodboardItem {
  const now = Date.now()
  return insertItem({
    id: randomUUID(),
    projectId: projectId(),
    type: 'layer',
    assetId: null,
    frameId: null,
    parentId: null,
    data: { name: 'Layer' },
    x,
    y,
    width: 420,
    height: 300,
    rotation: 0,
    // Layers sit behind everything else so frames render on top of them.
    zIndex: 0,
    createdAt: now,
    updatedAt: now,
  })
}

/** Add an empty Preview node (displays a connected frame's hero output). */
export function addPreview(x: number, y: number): MoodboardItem {
  const now = Date.now()
  return insertItem({
    id: randomUUID(),
    projectId: projectId(),
    type: 'preview',
    assetId: null,
    frameId: null,
    parentId: null,
    data: {},
    x,
    y,
    width: 280,
    height: 220,
    rotation: 0,
    zIndex: nextZIndex(),
    createdAt: now,
    updatedAt: now,
  })
}

/** Add a video-director node (a timeline-in-a-node: preview + video/audio tracks). */
export function addDirector(x: number, y: number): MoodboardItem {
  const now = Date.now()
  return insertItem({
    id: randomUUID(),
    projectId: projectId(),
    type: 'director',
    assetId: null,
    frameId: null,
    parentId: null,
    data: { name: 'Director', director: { width: 1920, height: 1080, fps: 30 } },
    x,
    y,
    // 2× a normal frame node (220×200) so the in-node timeline has room.
    width: 440,
    height: 400,
    rotation: 0,
    zIndex: nextZIndex(),
    createdAt: now,
    updatedAt: now,
  })
}

/** Add an "Edit Video/Audio" (trim) node: 1 input + 1 output, with an in/out window. */
export function addTrim(x: number, y: number): MoodboardItem {
  const now = Date.now()
  return insertItem({
    id: randomUUID(),
    projectId: projectId(),
    type: 'trim',
    assetId: null,
    frameId: null,
    parentId: null,
    data: { trim: { inPoint: 0, outPoint: 0 } },
    x,
    y,
    width: 360,
    height: 170,
    rotation: 0,
    zIndex: nextZIndex(),
    createdAt: now,
    updatedAt: now,
  })
}

/** Add a text-prompt node: a textarea whose output feeds a Generate node's prompt input. */
export function addPrompt(x: number, y: number): MoodboardItem {
  const now = Date.now()
  return insertItem({
    id: randomUUID(),
    projectId: projectId(),
    type: 'prompt',
    assetId: null,
    frameId: null,
    parentId: null,
    data: { promptText: '' },
    x,
    y,
    width: 240,
    height: 120,
    rotation: 0,
    zIndex: nextZIndex(),
    createdAt: now,
    updatedAt: now,
  })
}

/**
 * The prompt text feeding a Generate frame — the text of a Prompt node wired into the frame's
 * canvas node on the `prompt` handle. Null if nothing is connected. Used by the executor to inject
 * the prompt at run time (the Prompt node is the single source of truth, not a copied param).
 */
export function promptTextForFrame(frameId: string): string | null {
  const db = getDb()
  const nodes = db
    .prepare("SELECT id FROM moodboard_items WHERE frame_id = ? AND type = 'frame'")
    .all(frameId) as Array<{ id: string }>
  for (const node of nodes) {
    const conns = db
      .prepare('SELECT from_item_id, data FROM moodboard_connectors WHERE to_item_id = ?')
      .all(node.id) as Array<{ from_item_id: string; data: string | null }>
    for (const conn of conns) {
      const cdata = (conn.data ? JSON.parse(conn.data) : {}) as { targetHandle?: string }
      if (cdata.targetHandle !== 'prompt') continue
      const src = db
        .prepare("SELECT data FROM moodboard_items WHERE id = ? AND type = 'prompt'")
        .get(conn.from_item_id) as { data: string | null } | undefined
      if (!src) continue
      const sdata = (src.data ? JSON.parse(src.data) : {}) as MoodboardItemData
      if (typeof sdata.promptText === 'string' && sdata.promptText.trim().length > 0) {
        return sdata.promptText
      }
    }
  }
  return null
}

export function updateItem(id: string, patch: MoodboardItemPatch): MoodboardItem {
  getItem(id) // ensure exists
  const sets: string[] = []
  const params: Record<string, unknown> = { id, updatedAt: Date.now() }
  const map: Record<string, string> = {
    x: 'x',
    y: 'y',
    width: 'width',
    height: 'height',
    rotation: 'rotation',
    zIndex: 'z_index',
  }
  for (const [key, column] of Object.entries(map)) {
    const value = patch[key as keyof MoodboardItemPatch]
    if (typeof value === 'number') {
      sets.push(`${column} = @${key}`)
      params[key] = value
    }
  }
  if (patch.data !== undefined) {
    sets.push('data = @data')
    params.data = JSON.stringify(patch.data)
  }
  if (patch.parentId !== undefined) {
    sets.push('parent_id = @parentId')
    params.parentId = patch.parentId
  }
  sets.push('updated_at = @updatedAt')
  getDb()
    .prepare(`UPDATE moodboard_items SET ${sets.join(', ')} WHERE id = @id`)
    .run(params)
  return getItem(id)
}

export function deleteItem(id: string): void {
  const db = getDb()
  const tx = db.transaction(() => {
    db.prepare('DELETE FROM moodboard_connectors WHERE from_item_id = ? OR to_item_id = ?').run(
      id,
      id,
    )
    db.prepare('DELETE FROM moodboard_items WHERE id = ?').run(id)
  })
  tx()
}

export async function importAndPlace(x: number, y: number): Promise<MoodboardItem[]> {
  // Import into the shared library root, then place each on the board in a cascade.
  const assets = await importViaDialog(null)
  return assets.map((asset, i) => addAssetItem(asset.id, x + i * 28, y + i * 28))
}

export function createConnector(
  fromItemId: string,
  toItemId: string,
  sourceHandle: string | null = null,
  targetHandle: string | null = null,
): MoodboardConnector {
  getItem(fromItemId)
  getItem(toItemId)
  const connector: MoodboardConnector = {
    id: randomUUID(),
    projectId: projectId(),
    fromItemId,
    toItemId,
    label: null,
    // Remember which handles the edge attached to so it re-renders on the same
    // sides (frames have several handles: 'out' plus visual 'vl'/'vr'/'vb').
    data: { sourceHandle, targetHandle },
    createdAt: Date.now(),
  }
  getDb()
    .prepare(
      `INSERT INTO moodboard_connectors (id, project_id, from_item_id, to_item_id, label, data, created_at)
       VALUES (@id, @projectId, @fromItemId, @toItemId, @label, @data, @createdAt)`,
    )
    .run({ ...connector, data: JSON.stringify(connector.data) })
  return connector
}

export function deleteConnector(id: string): void {
  getDb().prepare('DELETE FROM moodboard_connectors WHERE id = ?').run(id)
}

/** Set the per-input audio volume (0..1) stored on a connector (director L1 inputs). */
export function setConnectorVolume(id: string, volume: number): void {
  const db = getDb()
  const row = db.prepare('SELECT data FROM moodboard_connectors WHERE id = ?').get(id) as
    | { data: string | null }
    | undefined
  const data = (row?.data ? JSON.parse(row.data) : {}) as Record<string, unknown>
  data.volume = Math.min(1, Math.max(0, volume))
  db.prepare('UPDATE moodboard_connectors SET data = ? WHERE id = ?').run(JSON.stringify(data), id)
}
