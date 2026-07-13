/**
 * Export a whole project as a single portable .zip. A project is already a self-contained
 * `.inlinestudio` folder (project.db + assets/ + takes/ outputs + workflows/ + thumbs +
 * workflow-memory), so the export is just that folder zipped — everything needed to open
 * and run the project exactly on another machine (their own ComfyUI models/nodes aside).
 */
import { join, basename } from 'node:path'
import { createWriteStream, existsSync } from 'node:fs'
import archiver from 'archiver'
import type { ProjectExportResult } from '@shared/types'
import { caps } from '../capabilities'
import { checkpointProjectDb } from '../db'

/** SQLite sidecars that must NOT be shipped: -wal is folded into project.db by the
 *  checkpoint below, and -shm is shared memory that SQLite rebuilds on open. */
const DB_SIDECARS = ['project.db-wal', 'project.db-shm']

export async function exportProject(projectPath: string): Promise<ProjectExportResult | null> {
  if (!existsSync(join(projectPath, 'project.db'))) {
    throw new Error('Not a valid Inline Studio project (no project.db).')
  }
  const folderName = basename(projectPath) // e.g. MyFilm.inlinestudio

  const dest = await caps().pickSavePath({
    title: 'Export Project',
    defaultPath: `${folderName}.zip`,
    filters: [{ name: 'Zip archive', extensions: ['zip'] }],
  })
  if (!dest) return null

  // Fold the WAL into project.db so the exported file is complete on its own — otherwise
  // recent data lives in the sidecars, which we then leave out of the zip.
  checkpointProjectDb(projectPath)

  await zipFolder(projectPath, folderName, dest)
  return { path: dest }
}

/** Zip `srcDir` into `destZip`, nesting everything under `topName` so unzipping yields the folder. */
function zipFolder(srcDir: string, topName: string, destZip: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const output = createWriteStream(destZip)
    const archive = archiver('zip', { zlib: { level: 6 } })

    output.on('close', () => resolve())
    output.on('error', reject)
    archive.on('warning', (e: { code?: string }) => {
      // Missing-file warnings (e.g. a transient -shm) are non-fatal.
      if (e.code !== 'ENOENT') reject(e)
    })
    archive.on('error', reject)

    archive.pipe(output)
    // Include everything except the DB sidecars (folded into project.db above).
    archive.directory(srcDir, topName, (entry) => {
      const name = entry.name ?? ''
      return DB_SIDECARS.some((s) => name === s || name.endsWith(`/${s}`)) ? false : entry
    })
    void archive.finalize()
  })
}
