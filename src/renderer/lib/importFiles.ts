/**
 * Import media files chosen from an OS drag-drop or a file picker. Under Electron the File objects
 * carry a real filesystem path (getPathForFile -> importPaths). In the browser there is no path, so
 * the bytes are uploaded to the web server's /upload (same origin), which runs the same importFile
 * logic. Returns the created assets so callers (e.g. the moodboard) can place them.
 */
import type { Asset } from '@shared/types'
import { studio } from './studio'

export async function importFilesToLibrary(
  files: File[],
  folderId: string | null,
): Promise<Asset[]> {
  if (files.length === 0) return []
  const paths = files.map((f) => studio().getPathForFile(f))
  if (paths.every((p) => p.length > 0)) {
    const res = await studio().assets.importPaths(paths, folderId)
    return res.ok ? res.value : []
  }
  return uploadFiles(files, folderId)
}

/** Upload File bytes to the web server and return the created assets (one request per file). */
export async function uploadFiles(files: File[], folderId: string | null): Promise<Asset[]> {
  const created = await Promise.all(files.map((f) => uploadOne(f, folderId)))
  return created.filter((a): a is Asset => a !== null)
}

async function uploadOne(file: File, folderId: string | null): Promise<Asset | null> {
  const params = new URLSearchParams({ name: file.name })
  if (folderId) params.set('folderId', folderId)
  const res = await fetch(`/upload?${params.toString()}`, { method: 'POST', body: file })
  const body = (await res.json()) as { ok: boolean; value?: Asset | null }
  return body.ok ? (body.value ?? null) : null
}

/** Open the browser file picker; resolves with the chosen files (empty if cancelled). */
export function pickFilesViaInput(): Promise<File[]> {
  return new Promise((resolve) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = true
    input.onchange = () => resolve(input.files ? Array.from(input.files) : [])
    input.oncancel = () => resolve([])
    input.click()
  })
}
