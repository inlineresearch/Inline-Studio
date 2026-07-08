/** IPC for saving project media (takes/assets) out to a user-chosen location. */
import { dialog, clipboard, nativeImage } from 'electron'
import { copyFile } from 'node:fs/promises'
import { statSync } from 'node:fs'
import { basename, extname } from 'node:path'
import { IpcChannels } from '@shared/ipc'
import { handle } from './handler'
import { resolveMediaPath } from '../media/resolve'

/** Build the save dialog's default filename: the suggestion, with the source extension. */
function suggestedFileName(sourcePath: string, suggested: string): string {
  const ext = extname(sourcePath)
  let base = (typeof suggested === 'string' ? suggested : '').trim()
  if (!base) base = basename(sourcePath)
  // Strip characters that are awkward in filenames; keep it readable.
  base = base.replace(/[\\/:*?"<>|]+/g, ' ').trim() || 'media'
  // Ensure the real extension is present. Don't rely on `extname(base)` to detect it — the
  // suggested name often contains dots (e.g. a model version like "LTX 2.3"), which would fool
  // it into thinking the file already has an extension and leave the save with no format. Compare
  // the actual ending instead, so a video reliably keeps its .mp4 (etc.).
  if (ext && !base.toLowerCase().endsWith(ext.toLowerCase())) base += ext
  return base
}

export function registerMediaHandlers(): void {
  handle<[string, string], boolean>(IpcChannels.media.save, async (src, suggestedName) => {
    if (typeof src !== 'string' || src.length === 0) throw new Error('Invalid media reference.')

    const sourcePath = resolveMediaPath(src)
    if (!sourcePath) throw new Error('Media not found in the open project.')
    try {
      statSync(sourcePath)
    } catch {
      throw new Error('Media file no longer exists.')
    }

    const { canceled, filePath } = await dialog.showSaveDialog({
      title: 'Save media',
      defaultPath: suggestedFileName(sourcePath, suggestedName),
    })
    if (canceled || !filePath) return false

    await copyFile(sourcePath, filePath)
    return true
  })

  handle<[string], void>(IpcChannels.media.copyImage, async (src) => {
    if (typeof src !== 'string' || src.length === 0) throw new Error('Invalid media reference.')

    const sourcePath = resolveMediaPath(src)
    if (!sourcePath) throw new Error('Image not found in the open project.')

    const image = nativeImage.createFromPath(sourcePath)
    if (image.isEmpty()) throw new Error('That file could not be read as an image.')
    clipboard.writeImage(image)
  })
}
