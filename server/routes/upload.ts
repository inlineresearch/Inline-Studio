/**
 * Browser asset import: POST /upload?name=<filename>&folderId=<id> with the raw file bytes as the
 * body (one request per file). The bytes are written to a temp file so the existing importFile logic
 * (copy into the project, thumbnail/transcode) runs unchanged, then the temp file is removed. On
 * success it broadcasts libraryChanged so open tabs refresh, and returns the created asset.
 */
import { writeFile, mkdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, basename } from 'node:path'
import { randomUUID } from 'node:crypto'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { IpcChannels } from '@shared/ipc'
import { importFile } from '@main/assets/store'
import { broadcast } from '@main/events/broadcaster'

export function handleUpload(
  req: IncomingMessage,
  res: ServerResponse,
  query: URLSearchParams,
): void {
  // basename() strips any path components from the client-supplied name (traversal guard). importFile
  // names the asset from the file's basename, so the temp file must keep the original name.
  const name = basename(query.get('name') ?? 'upload') || 'upload'
  const folderId = query.get('folderId') || null
  const chunks: Buffer[] = []
  req.on('data', (chunk: Buffer) => chunks.push(chunk))
  req.on('end', () => {
    void (async () => {
      const dir = join(tmpdir(), `sl-upload-${randomUUID()}`)
      try {
        await mkdir(dir, { recursive: true })
        const tmp = join(dir, name)
        await writeFile(tmp, Buffer.concat(chunks))
        const asset = await importFile(tmp, folderId)
        if (asset) broadcast(IpcChannels.events.libraryChanged)
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ ok: true, value: asset }))
      } catch (e) {
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ ok: false, error: e instanceof Error ? e.message : String(e) }))
      } finally {
        await rm(dir, { recursive: true, force: true })
      }
    })()
  })
}
