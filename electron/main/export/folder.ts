/**
 * Export each frame's chosen Output (hero take), in frame order, into a folder the
 * user picks — numbered for finishing in an external NLE (Resolve/Premiere/CapCut).
 * Plain file copy; no ffmpeg. Frames without an Output yet are reported and skipped.
 */
import { join, extname } from 'node:path'
import { copyFileSync } from 'node:fs'
import type { ExportResult } from '@shared/types'
import { caps } from '../capabilities'
import { getDb, getOpenProjectFolder } from '../db'

interface ExportRow {
  name: string
  file_path: string | null
}

export async function exportFrames(): Promise<ExportResult | null> {
  const projectFolder = getOpenProjectFolder()
  if (!projectFolder) throw new Error('No project is open.')

  const dir = await caps().pickDirectory({
    title: 'Export frames to folder',
    buttonLabel: 'Export',
    createDirectory: true,
  })
  if (!dir) return null

  const rows = getDb()
    .prepare(
      `SELECT s.name AS name, t.file_path AS file_path
       FROM frames s
       LEFT JOIN takes t ON s.hero_take_id = t.id
       ORDER BY s.position`,
    )
    .all() as ExportRow[]

  let exported = 0
  const skipped: string[] = []
  for (const row of rows) {
    if (!row.file_path) {
      skipped.push(row.name)
      continue
    }
    exported++
    const ext = extname(row.file_path) || '.png'
    const num = String(exported).padStart(3, '0')
    const safeName = row.name.replace(/[^\w.-]+/g, '_')
    copyFileSync(join(projectFolder, row.file_path), join(dir, `${num}_${safeName}${ext}`))
  }

  return { dir, exported, skipped }
}
