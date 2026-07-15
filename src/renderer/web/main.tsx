/**
 * Web SPA entry — the self-booting build that Inline Core serves on its own port (mirrors ComfyUI's
 * frontend package). Unlike the desktop boot (`main.tsx`, which uses the Electron preload), this
 * injects an HTTP/WebSocket backend client pointed at the same origin: every InlineStudioApi call
 * posts to `/rpc` and events stream from `/events` on whatever host served this page (Core in prod,
 * the Vite dev server proxied to Core in `npm run dev:web`).
 */
import { mountStudioApp, createWebClient, setMediaResolver } from '@/index'

const root = document.getElementById('root')
if (!root) throw new Error('Missing #root element')

const base = window.location.origin
// Project media loads over Core's GET /media/<project-relative-path> route (same origin).
setMediaResolver((rel) => `${base}/media/` + rel.split('/').map(encodeURIComponent).join('/'))
mountStudioApp(root, { client: createWebClient(base) })
