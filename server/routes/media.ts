/**
 * Serves the open project's local media over HTTP: GET /media/<project-relative-path>. Ported from
 * the Electron inlinestudio-media:// protocol handler, keeping the `..` traversal guard, MIME map,
 * immutable caching, and Range/206 support so <video>/<audio> seek and stream in the browser.
 */
import { createReadStream, statSync } from 'node:fs'
import { join, normalize, sep, extname } from 'node:path'
import type { ServerResponse } from 'node:http'
import { getOpenProjectFolder } from '@main/db'

const MIME_BY_EXT: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.bmp': 'image/bmp',
  '.mp4': 'video/mp4',
  '.m4v': 'video/mp4',
  '.mov': 'video/quicktime',
  '.webm': 'video/webm',
  '.mkv': 'video/x-matroska',
  '.avi': 'video/x-msvideo',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.ogg': 'audio/ogg',
  '.m4a': 'audio/mp4',
  '.flac': 'audio/flac',
  '.json': 'application/json',
}

const IMMUTABLE_CACHE = 'public, max-age=31536000, immutable'

export function serveMedia(
  encodedRelative: string,
  rangeHeader: string | undefined,
  res: ServerResponse,
): void {
  const projectFolder = getOpenProjectFolder()
  if (!projectFolder) return sendText(res, 404, 'No project open')

  const relative = decodeURIComponent(encodedRelative).replace(/^\/+/, '')
  const target = normalize(join(projectFolder, relative))
  const root = normalize(projectFolder)
  if (target !== root && !target.startsWith(root + sep)) return sendText(res, 403, 'Forbidden')

  let size: number
  try {
    size = statSync(target).size
  } catch {
    return sendText(res, 404, 'Not found')
  }

  const type = MIME_BY_EXT[extname(target).toLowerCase()] ?? 'application/octet-stream'
  const match = rangeHeader ? /bytes=(\d*)-(\d*)/.exec(rangeHeader) : null
  if (match) {
    let start = match[1] ? parseInt(match[1], 10) : 0
    let end = match[2] ? parseInt(match[2], 10) : size - 1
    if (Number.isNaN(start) || start < 0) start = 0
    if (Number.isNaN(end) || end >= size) end = size - 1
    if (start > end) return sendText(res, 416, 'Range Not Satisfiable')
    res.writeHead(206, {
      'Content-Type': type,
      'Content-Range': `bytes ${start}-${end}/${size}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': String(end - start + 1),
      'Cache-Control': IMMUTABLE_CACHE,
    })
    createReadStream(target, { start, end }).pipe(res)
    return
  }

  res.writeHead(200, {
    'Content-Type': type,
    'Accept-Ranges': 'bytes',
    'Content-Length': String(size),
    'Cache-Control': IMMUTABLE_CACHE,
  })
  createReadStream(target).pipe(res)
}

function sendText(res: ServerResponse, status: number, message: string): void {
  res.writeHead(status, { 'Content-Type': 'text/plain' })
  res.end(message)
}
