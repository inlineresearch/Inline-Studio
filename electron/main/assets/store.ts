/**
 * Asset library: import media into the open project and list it. Imported files
 * are copied into the project's `assets/` folder (by id) so the project stays a
 * self-contained, portable folder.
 */
import { join, extname, basename } from 'node:path'
import { copyFileSync, existsSync, unlinkSync } from 'node:fs'
import { randomUUID } from 'node:crypto'
import type { Asset, AssetKind } from '@shared/types'
import { IpcChannels } from '@shared/ipc'
import { broadcast } from '../events/broadcaster'
import { caps } from '../capabilities'
import { getDb, getOpenProjectFolder } from '../db'
import {
  ffmpegAvailable,
  generatePoster,
  generateImageThumb,
  generatePeaks,
  probeVideo,
  isWebPlayable,
  transcodeH264,
} from '../media/ffmpeg'

const KIND_BY_EXT: Record<string, AssetKind> = {
  '.png': 'image',
  '.jpg': 'image',
  '.jpeg': 'image',
  '.webp': 'image',
  '.gif': 'image',
  '.bmp': 'image',
  '.tiff': 'image',
  '.mp4': 'video',
  '.mov': 'video',
  '.webm': 'video',
  '.mkv': 'video',
  '.avi': 'video',
  '.m4v': 'video',
  '.mp3': 'audio',
  '.wav': 'audio',
  '.aac': 'audio',
  '.flac': 'audio',
  '.ogg': 'audio',
  '.m4a': 'audio',
}

function kindForFile(filePath: string): AssetKind | null {
  return KIND_BY_EXT[extname(filePath).toLowerCase()] ?? null
}

interface AssetRow {
  id: string
  project_id: string
  folder_id: string | null
  name: string
  file_path: string
  kind: AssetKind
  thumb_path: string | null
  preview_path: string | null
  created_at: number
}

function rowToAsset(row: AssetRow): Asset {
  return {
    id: row.id,
    projectId: row.project_id,
    folderId: row.folder_id,
    name: row.name,
    filePath: row.file_path,
    kind: row.kind,
    thumbPath: row.thumb_path,
    previewPath: row.preview_path,
    createdAt: row.created_at,
  }
}

/** Tell the client the library changed (a poster/transcode finished). */
function notifyLibraryChanged(): void {
  broadcast(IpcChannels.events.libraryChanged)
}

function setPreviewPath(id: string, rel: string): void {
  // The asset may have been deleted meanwhile; UPDATE no-ops if the row is gone.
  getDb().prepare('UPDATE assets SET preview_path = ? WHERE id = ?').run(rel, id)
  notifyLibraryChanged()
}

function setThumbPath(id: string, rel: string): void {
  getDb().prepare('UPDATE assets SET thumb_path = ? WHERE id = ?').run(rel, id)
  notifyLibraryChanged()
}

/**
 * Generate an audio waveform (peaks JSON) in the background and store it as the asset's
 * thumb_path (audio has no poster, so the column is reused). No-op if already present.
 */
async function ensureWaveform(id: string, srcAbs: string, folder: string): Promise<void> {
  try {
    const rel = `thumbs/${id}.peaks.json`
    if (await generatePeaks(srcAbs, join(folder, rel))) setThumbPath(id, rel)
  } catch {
    // ignore — the 🎵 placeholder covers display; retried on next project open
  }
}

/**
 * Generate a downscaled JPEG thumbnail for an image in the background and store it as
 * the asset's thumb_path, so nodes and the library grid stop decoding the full-res
 * original. No-op if it can't be produced (the original still renders meanwhile).
 */
async function ensureImageThumb(id: string, srcAbs: string, folder: string): Promise<void> {
  try {
    const rel = `thumbs/${id}.jpg`
    if (await generateImageThumb(srcAbs, join(folder, rel))) setThumbPath(id, rel)
  } catch {
    // ignore — the full-res original covers display; retried on next project open
  }
}

/**
 * Resolve a video's Chromium-playable source into `preview_path` (background):
 *  - already a playable codec  → point preview_path at the original;
 *  - otherwise                 → transcode to H.264 and point preview_path at that.
 * Either way the UI ends up with a `<video>` source it can actually decode. A video
 * whose codec can't be probed is left alone (poster covers it; retried next open).
 */
async function ensurePlayable(
  id: string,
  srcAbs: string,
  folder: string,
  originalRel: string,
): Promise<void> {
  try {
    const probe = await probeVideo(srcAbs)
    if (!probe) return
    if (isWebPlayable(probe.codec, probe.pixFmt)) {
      setPreviewPath(id, originalRel)
      return
    }
    const previewRel = `thumbs/${id}.preview.mp4`
    if (await transcodeH264(srcAbs, join(folder, previewRel))) setPreviewPath(id, previewRel)
  } catch {
    // ignore — the poster covers display; we retry on next project open
  }
}

function projectId(): string {
  const row = getDb().prepare('SELECT id FROM project LIMIT 1').get() as { id: string } | undefined
  if (!row) throw new Error('No project is open.')
  return row.id
}

/** Copy a single file into the project (under `folderId`) and insert its row. */
export async function importFile(absPath: string, folderId: string | null): Promise<Asset | null> {
  const kind = kindForFile(absPath)
  if (!kind) return null

  const folder = getOpenProjectFolder()
  if (!folder) throw new Error('No project is open.')

  const id = randomUUID()
  const ext = extname(absPath).toLowerCase()
  const relative = `assets/${id}${ext}`
  copyFileSync(absPath, join(folder, relative))

  // Videos get a first-frame poster now so they always render, regardless of codec.
  let thumbPath: string | null = null
  if (kind === 'video' && ffmpegAvailable()) {
    const thumbRel = `thumbs/${id}.jpg`
    if (await generatePoster(join(folder, relative), join(folder, thumbRel))) thumbPath = thumbRel
  }

  const asset: Asset = {
    id,
    projectId: projectId(),
    folderId,
    name: basename(absPath),
    filePath: relative,
    kind,
    thumbPath,
    previewPath: null,
    createdAt: Date.now(),
  }
  getDb()
    .prepare(
      `INSERT INTO assets (id, project_id, folder_id, name, file_path, kind, thumb_path, preview_path, created_at)
       VALUES (@id, @projectId, @folderId, @name, @filePath, @kind, @thumbPath, @previewPath, @createdAt)`,
    )
    .run(asset)

  // Background: resolve a playable source (the poster covers the meantime).
  if (kind === 'video' && ffmpegAvailable()) {
    void ensurePlayable(id, join(folder, relative), folder, relative)
  }
  // Background: downscale a thumbnail for images (full-res original covers the meantime).
  if (kind === 'image' && ffmpegAvailable()) {
    void ensureImageThumb(id, join(folder, relative), folder)
  }
  // Background: render a waveform for audio (the 🎵 placeholder covers the meantime).
  if (kind === 'audio' && ffmpegAvailable()) {
    void ensureWaveform(id, join(folder, relative), folder)
  }
  return asset
}

/**
 * Generate posters (and playable transcodes) for video assets that don't have one
 * yet — i.e. videos imported before this feature existed. Runs once per asset (keyed
 * on a missing thumb_path), sequentially in the background, refreshing the UI as each
 * finishes. Call after a project opens.
 */
export function backfillVideoAssets(): void {
  if (!ffmpegAvailable()) return
  const folder = getOpenProjectFolder()
  if (!folder) return
  const rows = getDb()
    .prepare(
      "SELECT id, file_path, thumb_path, preview_path FROM assets WHERE kind = 'video' AND (thumb_path IS NULL OR preview_path IS NULL)",
    )
    .all() as Array<{
    id: string
    file_path: string
    thumb_path: string | null
    preview_path: string | null
  }>
  if (rows.length === 0) return

  void (async () => {
    for (const r of rows) {
      if (getOpenProjectFolder() !== folder) return // project switched — stop
      const srcAbs = join(folder, r.file_path)
      if (!existsSync(srcAbs)) continue
      if (!r.thumb_path) {
        const thumbRel = `thumbs/${r.id}.jpg`
        if (await generatePoster(srcAbs, join(folder, thumbRel))) {
          getDb().prepare('UPDATE assets SET thumb_path = ? WHERE id = ?').run(thumbRel, r.id)
          notifyLibraryChanged()
        }
      }
      if (!r.preview_path) await ensurePlayable(r.id, srcAbs, folder, r.file_path)
    }
  })()
}

/**
 * Generate waveforms for audio assets imported before waveforms existed (keyed on a
 * missing thumb_path). Background + sequential, like the video backfill.
 */
export function backfillAudioAssets(): void {
  if (!ffmpegAvailable()) return
  const folder = getOpenProjectFolder()
  if (!folder) return
  const rows = getDb()
    .prepare("SELECT id, file_path FROM assets WHERE kind = 'audio' AND thumb_path IS NULL")
    .all() as Array<{ id: string; file_path: string }>
  if (rows.length === 0) return

  void (async () => {
    for (const r of rows) {
      if (getOpenProjectFolder() !== folder) return // project switched — stop
      const srcAbs = join(folder, r.file_path)
      if (!existsSync(srcAbs)) continue
      await ensureWaveform(r.id, srcAbs, folder)
    }
  })()
}

/**
 * Generate thumbnails for image assets imported before image thumbnails existed (keyed
 * on a missing thumb_path). Background + sequential, like the video/audio backfills.
 * Call after a project opens so older libraries speed up after the first open.
 */
export function backfillImageAssets(): void {
  if (!ffmpegAvailable()) return
  const folder = getOpenProjectFolder()
  if (!folder) return
  const rows = getDb()
    .prepare("SELECT id, file_path FROM assets WHERE kind = 'image' AND thumb_path IS NULL")
    .all() as Array<{ id: string; file_path: string }>
  if (rows.length === 0) return

  void (async () => {
    for (const r of rows) {
      if (getOpenProjectFolder() !== folder) return // project switched — stop
      const srcAbs = join(folder, r.file_path)
      if (!existsSync(srcAbs)) continue
      await ensureImageThumb(r.id, srcAbs, folder)
    }
  })()
}

export async function importViaDialog(folderId: string | null): Promise<Asset[]> {
  if (!getOpenProjectFolder()) throw new Error('Open a project first.')

  const filePaths = await caps().pickFiles({
    title: 'Import media',
    buttonLabel: 'Import',
    multiple: true,
    filters: [
      { name: 'Media', extensions: Object.keys(KIND_BY_EXT).map((e) => e.slice(1)) },
      { name: 'All Files', extensions: ['*'] },
    ],
  })
  if (filePaths.length === 0) return []

  const imported: Asset[] = []
  for (const filePath of filePaths) {
    const asset = await importFile(filePath, folderId)
    if (asset) imported.push(asset)
  }
  return imported
}

/** Import media by absolute path (e.g. files dropped from the OS); skips unsupported kinds. */
export async function importPaths(paths: string[], folderId: string | null): Promise<Asset[]> {
  if (!getOpenProjectFolder()) throw new Error('Open a project first.')
  const imported: Asset[] = []
  for (const p of paths) {
    if (typeof p !== 'string' || p.length === 0) continue
    const asset = await importFile(p, folderId)
    if (asset) imported.push(asset)
  }
  return imported
}

/** Resolve a library asset to a relative path + kind + name, for the director timeline. */
export function assetFile(
  assetId: string,
): { filePath: string; kind: AssetKind; name: string } | null {
  const row = getDb()
    .prepare('SELECT file_path, kind, name FROM assets WHERE id = ?')
    .get(assetId) as { file_path: string; kind: AssetKind; name: string } | undefined
  return row ? { filePath: row.file_path, kind: row.kind, name: row.name } : null
}

export function listAssets(): Asset[] {
  const rows = getDb().prepare('SELECT * FROM assets ORDER BY created_at DESC').all() as AssetRow[]
  return rows.map(rowToAsset)
}

/**
 * Delete a library asset: removes it from any moodboard items, deletes the row and
 * the file. Blocked if it's used as a frame input (a frame must keep ≥1 input) —
 * remove it from those frames first.
 */
export function deleteAsset(assetId: string): void {
  const db = getDb()
  const used = db
    .prepare('SELECT COUNT(*) AS n FROM frame_inputs WHERE asset_id = ?')
    .get(assetId) as { n: number }
  if (used.n > 0) {
    throw new Error(
      `This asset is used by ${used.n} frame${used.n === 1 ? '' : 's'} — remove it from those frames first.`,
    )
  }
  const row = db
    .prepare('SELECT file_path, thumb_path, preview_path FROM assets WHERE id = ?')
    .get(assetId) as
    | { file_path: string; thumb_path: string | null; preview_path: string | null }
    | undefined
  const folder = getOpenProjectFolder()
  const tx = db.transaction(() => {
    db.prepare('DELETE FROM moodboard_items WHERE asset_id = ?').run(assetId)
    db.prepare('DELETE FROM assets WHERE id = ?').run(assetId)
  })
  tx()
  if (row && folder) {
    for (const rel of [row.file_path, row.thumb_path, row.preview_path]) {
      if (!rel) continue
      const abs = join(folder, rel)
      if (existsSync(abs)) {
        try {
          unlinkSync(abs)
        } catch {
          // ignore
        }
      }
    }
  }
}
