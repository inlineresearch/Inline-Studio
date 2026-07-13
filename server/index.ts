/**
 * The headless Storyline web server. Reuses the Electron main handlers behind the transport,
 * capabilities, and broadcaster seams, serves the @inlineresearch/ui bundle and project media, and
 * exposes the full InlineStudioApi over POST /rpc + a WebSocket event stream. inline-core is a
 * separate engine reached over /v1 (unchanged). Run: node out/server/index.cjs.
 */
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { mkdirSync } from 'node:fs'
import { WebSocketServer } from 'ws'
import { setTransport } from '@main/ipc/handler'
import { setBroadcaster } from '@main/events/broadcaster'
import { setCapabilities } from '@main/capabilities'
import { registerIpcHandlers } from '@main/ipc'
import { closeProjectDb } from '@main/db'
import { loadConfig } from './config'
import { HttpWsTransport } from './transport/httpWsTransport'
import { WsBroadcaster } from './transport/wsBroadcaster'
import { webCapabilities } from './capabilities/web'
import { serveMedia } from './routes/media'
import { serveUiAsset, indexHtml } from './routes/static'
import { handleUpload } from './routes/upload'
import { startCore } from './core/supervisor'

function main(): void {
  const config = loadConfig()
  mkdirSync(config.dataDir, { recursive: true })
  mkdirSync(config.workspaceDir, { recursive: true })

  const transport = new HttpWsTransport()
  const broadcaster = new WsBroadcaster()
  setTransport(transport)
  setBroadcaster(broadcaster)
  setCapabilities(webCapabilities(config.dataDir, config.workspaceDir))
  registerIpcHandlers()

  // Ensure inline-core is available (connect if running, else spawn when managed). Non-blocking:
  // the web UI comes up immediately; only generation waits on core.
  const corePromise = startCore(config).catch(() => null)

  const html = indexHtml()
  const server = createServer((req, res) => {
    const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`)
    const path = url.pathname

    if (req.method === 'POST' && path === '/rpc') return handleRpc(req, res, transport)
    if (req.method === 'POST' && path === '/upload') return handleUpload(req, res, url.searchParams)
    if (path.startsWith('/media/')) {
      return serveMedia(path.slice('/media/'.length), req.headers.range, res)
    }
    if (path.startsWith('/ui/')) return serveUiAsset(config.uiDir, path.slice('/ui/'.length), res)
    if (path === '/' || path === '/index.html') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
      res.end(html)
      return
    }
    res.writeHead(404, { 'Content-Type': 'text/plain' })
    res.end('Not found')
  })

  const wss = new WebSocketServer({ server, path: '/events' })
  wss.on('connection', (socket) => broadcaster.add(socket))

  server.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code === 'EADDRINUSE') {
      console.error(
        `Port ${config.port} is already in use. Stop the other server or set STORYLINE_PORT.`,
      )
    } else {
      console.error(`Server error: ${err.message}`)
    }
    process.exit(1)
  })

  server.listen(config.port, config.host, () => {
    console.log(`Storyline web server: http://${config.host}:${config.port}`)
    console.log(`Projects workspace:   ${config.workspaceDir}`)
    console.log(`Inline Core expected at: ${config.coreUrl}`)
  })

  let shuttingDown = false
  const shutdown = (): void => {
    if (shuttingDown) return
    shuttingDown = true
    // Kill the managed core before exiting: process.exit would otherwise race the async kill and
    // orphan the child (a plain SIGTERM to us does not propagate to it).
    void corePromise.then((child) => {
      child?.kill()
      closeProjectDb()
      server.close(() => process.exit(0))
      setTimeout(() => process.exit(0), 2000).unref()
    })
  }
  process.on('SIGINT', shutdown)
  process.on('SIGTERM', shutdown)
}

function handleRpc(req: IncomingMessage, res: ServerResponse, transport: HttpWsTransport): void {
  const chunks: Buffer[] = []
  req.on('data', (chunk: Buffer) => chunks.push(chunk))
  req.on('end', () => {
    void (async () => {
      try {
        const body = JSON.parse(Buffer.concat(chunks).toString('utf-8')) as {
          channel: string
          args?: unknown[]
        }
        const result = await transport.dispatch(body.channel, body.args ?? [])
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify(result))
      } catch (e) {
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ ok: false, error: e instanceof Error ? e.message : String(e) }))
      }
    })()
  })
}

main()
