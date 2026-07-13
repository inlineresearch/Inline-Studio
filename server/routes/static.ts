/**
 * Serves the built @inlineresearch/ui bundle under /ui/* and the page that mounts it. The page
 * injects a web InlineStudioApi client (POST /rpc + WebSocket) and points resolveMedia at /media/*,
 * then calls mountStudioApp. UI, RPC, and media are one origin, so no CORS is needed.
 */
import { createReadStream, existsSync } from 'node:fs'
import { join, extname, normalize, sep } from 'node:path'
import type { ServerResponse } from 'node:http'

const CONTENT_TYPE: Record<string, string> = {
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.map': 'application/json',
  '.svg': 'image/svg+xml',
}

/** Stream a file from the UI dist dir, guarding against path traversal outside it. */
export function serveUiAsset(uiDir: string, name: string, res: ServerResponse): void {
  const root = normalize(uiDir)
  const file = normalize(join(root, name))
  const inside = file === root || file.startsWith(root + sep)
  if (!inside || !existsSync(file)) {
    res.writeHead(404, { 'Content-Type': 'text/plain' })
    res.end('Not found')
    return
  }
  res.writeHead(200, {
    'Content-Type': CONTENT_TYPE[extname(file).toLowerCase()] ?? 'application/octet-stream',
    'Cache-Control': 'no-cache',
  })
  createReadStream(file).pipe(res)
}

export function indexHtml(): string {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Inline Studio</title>
  <link rel="stylesheet" href="/ui/style.css" />
  <style>html,body,#root{height:100%;margin:0;background:#16171b}</style>
</head>
<body>
  <div id="root"></div>
  <script type="module">
    import { mountStudioApp, createWebClient, setMediaResolver } from '/ui/index.js'
    const base = location.origin
    setMediaResolver((rel) => base + '/media/' + rel.split('/').map(encodeURIComponent).join('/'))
    mountStudioApp(document.getElementById('root'), { client: createWebClient(base) })
  </script>
</body>
</html>
`
}
