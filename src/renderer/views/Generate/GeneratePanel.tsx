import { useCallback, useEffect, useRef, useState } from 'react'
import type { ComfyStatus, ComfyOutput, ComfyRun } from '@shared/types'
import { useSettingsStore } from '../../store/settingsStore'
import { useUiStore } from '../../store/uiStore'
import { useFrameStore } from '../../store/frameStore'
import type { ComfyWebview } from '../../types/webview'
import { ConnectionGuide } from './ConnectionGuide'

/**
 * Code injected INTO the embedded ComfyUI page (via webview.executeJavaScript, which
 * bypasses cross-origin limits in Electron) to open a frame's saved workflow.
 *
 * Preferred: open the *saved* workflow file through ComfyUI's workflow store (exposed
 * on window.comfyAPI) so it becomes that named tab and Save overwrites the same file —
 * which keeps the frame↔workflow link intact. Falls back to loadGraphData (which opens
 * an Unsaved Workflow) only if the store isn't reachable. Resolves to a status string:
 * 'opened' (saved tab) | 'loaded' (unsaved fallback) | 'failed'.
 */
function openWorkflowScript(name: string): string {
  const n = JSON.stringify(name)
  return `(async () => {
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const waitFor = async (fn) => {
      const start = Date.now();
      while (Date.now() - start < 8000) {
        try { if (fn()) return true; } catch (e) {}
        await sleep(200);
      }
      return false;
    };
    await waitFor(() => window.app && window.app.graph);
    const base = ${n};
    const path = 'workflows/' + base + '.json';
    // Match by BASENAME (filename without folders or .json) so detection is robust
    // across ComfyUI versions that expose the tab as a path, a key, a filename, or a
    // bare string — the cause of duplicate tabs was a too-strict full-path compare.
    const baseOf = (w) => {
      if (!w) return '';
      let p = (typeof w === 'string') ? w : (w.path || w.key || w.filename || '');
      if (typeof p !== 'string') p = '';
      return p.replace(/^.*\\//, '').replace(/\\.json$/i, '');
    };
    const matches = (w) => baseOf(w) === base;
    // The list of currently OPEN tabs, however this version names it.
    const openTabs = (s) => {
      for (const c of [s.openWorkflows, s.openedWorkflows, s.openWorkflowPaths]) {
        const arr = (typeof c === 'function') ? c() : c;
        if (Array.isArray(arr)) return arr;
      }
      return [];
    };

    // 1) Open the SAVED workflow via the workflow store (so Save targets the same file).
    try {
      const reg = window.comfyAPI || {};
      let useWorkflowStore = reg.workflowStore && reg.workflowStore.useWorkflowStore;
      if (!useWorkflowStore) {
        for (const k in reg) {
          if (reg[k] && reg[k].useWorkflowStore) { useWorkflowStore = reg[k].useWorkflowStore; break; }
        }
      }
      if (useWorkflowStore) {
        const store = useWorkflowStore();
        // 1a) Already the active tab? Nothing to do.
        if (matches(store.activeWorkflow)) return 'active';
        // 1b) Already open in another tab? Switch to it — never open a duplicate.
        const already = openTabs(store).find(matches);
        if (already) {
          // Resolve to a workflow object (the list may hold bare path strings).
          let wf = (already && typeof already === 'object') ? already
            : (typeof store.getWorkflowByPath === 'function' ? store.getWorkflowByPath(path) : null);
          if (!wf) wf = already;
          if (typeof store.openWorkflow === 'function') await store.openWorkflow(wf);
          else store.activeWorkflow = wf;
          return 'switched';
        }
        // 1c) Not open yet. Make sure the store knows the file Inline Studio just pushed
        // (else getWorkflowByPath misses it and we'd open a throwaway Unsaved tab that
        // future opens can't match — the duplicate-tab loop), then open the SAVED file.
        for (const m of ['syncWorkflows','loadWorkflows','reloadWorkflows','refreshWorkflows']) {
          if (typeof store[m] === 'function') { try { await store[m](); break; } catch (e) {} }
        }
        let wf = (typeof store.getWorkflowByPath === 'function') ? store.getWorkflowByPath(path) : null;
        if (!wf && Array.isArray(store.workflows)) wf = store.workflows.find(matches);
        if (wf && typeof store.openWorkflow === 'function') { await store.openWorkflow(wf); return 'opened'; }
      }
    } catch (e) { console.error('[inlinestudio] store open failed', e); }

    // 2) Fallback: switch to a matching open tab via the workflow manager if there is
    // one; else load the graph (opens as an Unsaved Workflow).
    try {
      const mgr = window.app && window.app.workflowManager;
      const openList = mgr && (mgr.openWorkflows || mgr.workflows);
      if (Array.isArray(openList)) {
        const already = openList.find(matches);
        if (already) {
          if (typeof mgr.setWorkflow === 'function') { mgr.setWorkflow(already); return 'switched'; }
          if (already.load) { already.load(); return 'switched'; }
        }
      }
    } catch (e) { console.error('[inlinestudio] tab switch failed', e); }
    try {
      if (window.app && typeof window.app.loadGraphData === 'function') {
        const res = await fetch('/userdata/' + encodeURIComponent(path));
        if (res.ok) { window.app.loadGraphData(await res.json(), true, true, ${n}); return 'loaded'; }
      }
    } catch (e) { console.error('[inlinestudio] loadGraphData failed', e); }
    return 'failed';
  })();`
}

/**
 * Serialize the LIVE graph on the ComfyUI canvas AND report which workflow tab is
 * currently active, as a JSON string `{ name, graph }`:
 *  - `graph`: the serialized graph (same shape ComfyUI writes on Save), or null if the
 *    page can't serialize (older ComfyUI without window.app.graph.serialize).
 *  - `name`: the active workflow's path/name (e.g. "workflows/18 <id>.json"), or null if
 *    it can't be identified.
 * The host uses `name` to attribute the graph to the RIGHT frame — so closing a frame's
 * tab (which makes ComfyUI switch to a different tab) can never save the wrong graph.
 */
function serializeActiveWorkflowScript(): string {
  return `(() => {
    const out = { name: null, graph: null };
    try {
      const app = window.app;
      const g = app && app.graph;
      if (g && typeof g.serialize === 'function') out.graph = g.serialize();
      // Identify the active workflow via the workflow store (modern) or manager (older).
      let active = null;
      try {
        const reg = window.comfyAPI || {};
        let useWorkflowStore = reg.workflowStore && reg.workflowStore.useWorkflowStore;
        if (!useWorkflowStore) {
          for (const k in reg) { if (reg[k] && reg[k].useWorkflowStore) { useWorkflowStore = reg[k].useWorkflowStore; break; } }
        }
        if (useWorkflowStore) active = useWorkflowStore().activeWorkflow;
      } catch (e) {}
      if (!active) { try { active = app.workflowManager && app.workflowManager.activeWorkflow; } catch (e) {} }
      if (active) {
        const p = active.path || active.key || active.filename;
        if (typeof p === 'string') out.name = p;
      }
    } catch (e) {}
    return JSON.stringify(out);
  })();`
}

/**
 * Reduce an active workflow's path/name (e.g. "workflows/18 <id>.json" or "18 <id>") to
 * the bare workflow name Inline Studio stores as `comfyWorkflowName`, for tab→frame mapping.
 */
function activeWorkflowBaseName(path: string): string {
  return path.replace(/^.*\//, '').replace(/\.json$/i, '')
}

/** Marker the in-page save hook logs; the host listens for it on `console-message`. */
const WF_SAVED_MARKER = '[inlinestudio:wf-saved]'

/**
 * Injected once into the ComfyUI page. Two layers, both idempotent:
 * (A) Patch `api.fetchApi` to force `overwrite=true` on POSTs to userdata *workflow*
 *     files. ComfyUI otherwise POSTs a freshly-opened workflow with overwrite=false and
 *     the server 409s because Inline Studio already pushed that file on link — the "save
 *     fails until app restart" bug. Patching at the fetch layer catches the workflow
 *     service even when it captured the original `storeUserData` before us (it still
 *     resolves `this.fetchApi` at call time) and works whether the tab opened saved or
 *     "Unsaved".
 * (B) Patch `storeUserData` to also force overwrite for direct callers and to log a
 *     marker the host catches to pull the saved JSON back into Inline Studio's copy.
 */
function saveHookScript(): string {
  return `(() => {
    if (window.__inlineStudioSaveHook) return 'already';
    const findApi = () => {
      if (window.app && window.app.api && typeof window.app.api.storeUserData === 'function') return window.app.api;
      const reg = window.comfyAPI || {};
      if (reg.api && reg.api.api && typeof reg.api.api.storeUserData === 'function') return reg.api.api;
      for (const k in reg) { try { if (reg[k] && typeof reg[k].storeUserData === 'function') return reg[k]; } catch (e) {} }
      return null;
    };
    const api = findApi();
    if (!api) return 'no-api';

    // (A) Network-level guard: force overwrite=true on every POST to a userdata *workflow*
    // file. This is the robust fix — ComfyUI's workflow service calls \`this.fetchApi\`
    // dynamically, so it lands here even when the service captured the original
    // \`storeUserData\` before our patch below, and regardless of whether the workflow
    // opened as a saved tab or as an "Unsaved Workflow". Without it, a freshly-linked
    // frame (whose file Inline Studio already pushed) 409s on the first Save until an app
    // restart re-indexes ComfyUI's workflow store. The route is URL-encoded, so slashes
    // in the name appear as %2F and 'workflows' is still a substring.
    if (api.fetchApi && !window.__inlineStudioFetchHook) {
      const origFetch = api.fetchApi.bind(api);
      api.fetchApi = (route, options) => {
        try {
          const method = ((options && options.method) || 'GET').toUpperCase();
          if (method === 'POST' && typeof route === 'string'
              && route.indexOf('/userdata/') !== -1 && route.indexOf('workflows') !== -1) {
            if (/[?&]overwrite=false/i.test(route)) {
              route = route.replace(/([?&])overwrite=false/i, '$1overwrite=true');
            } else if (!/[?&]overwrite=/i.test(route)) {
              route += (route.indexOf('?') === -1 ? '?' : '&') + 'overwrite=true';
            }
          }
        } catch (e) {}
        return origFetch(route, options);
      };
      window.__inlineStudioFetchHook = true;
    }

    // (B) storeUserData patch: force overwrite for direct callers AND log a marker after a
    // workflow save so the host can pull the JSON back into Inline Studio's durable copy.
    const orig = api.storeUserData.bind(api);
    api.storeUserData = async (file, data, options) => {
      let opts = options;
      const isWorkflow = typeof file === 'string' && file.indexOf('workflows/') !== -1;
      // Force overwrite for our workflow files so re-saving never 409s. Only touch a
      // real options object so ComfyUI's defaults (stringify/throwOnError) survive.
      if (isWorkflow && options && typeof options === 'object') {
        opts = Object.assign({}, options, { overwrite: true });
      }
      const r = await orig(file, data, opts);
      try { if (isWorkflow) console.log('${WF_SAVED_MARKER} ' + file); } catch (e) {}
      return r;
    };
    window.__inlineStudioSaveHook = true;
    return 'hooked';
  })();`
}

/**
 * The Generate tab embeds ComfyUI in an iframe. It polls the backend; when it's not
 * reachable it shows guidance instead. The URL is editable (persisted to settings).
 * Per-frame "Send to ComfyUI" / "Pull result" actions live on the frame timeline.
 */
export function GeneratePanel(): React.JSX.Element {
  const { comfyUrl, load, setComfyUrl } = useSettingsStore()
  const linkedWorkflow = useUiStore((s) => s.linkedWorkflow)
  const setLinkedWorkflow = useUiStore((s) => s.setLinkedWorkflow)
  const activeFrameId = useUiStore((s) => s.activeFrameId)
  const mode = useUiStore((s) => s.mode)
  const activeFrame = useFrameStore((s) => s.frames.find((sh) => sh.id === activeFrameId))
  const captureOutput = useFrameStore((s) => s.captureOutput)
  const pullWorkflow = useFrameStore((s) => s.pullWorkflow)
  const saveLiveWorkflow = useFrameStore((s) => s.saveLiveWorkflow)
  const [status, setStatus] = useState<ComfyStatus | null>(null)
  const [draftUrl, setDraftUrl] = useState('')
  const webviewRef = useRef<ComfyWebview | null>(null)
  const [webviewReady, setWebviewReady] = useState(false)
  // Capture strip: the latest finished run + which of its outputs we've captured.
  const [run, setRun] = useState<ComfyRun | null>(null)
  const [captured, setCaptured] = useState<Set<string>>(new Set())
  const seenPromptId = useRef<string | null>(null)
  // The frame whose workflow tab is currently open in the webview. Drives autosave and
  // capture-on-switch/leave so the right frame receives the captured graph.
  const prevFrameRef = useRef<string | null>(null)
  // Consecutive failed status pings — used to debounce the "not reachable" state so a
  // single slow ping doesn't tear down the embedded page (see `check`).
  const failures = useRef(0)
  // Whether ComfyUI has ever been reachable this session. Once it has, we keep the
  // <webview> mounted and overlay the guide when it drops — never unmount it.
  const [everConnected, setEverConnected] = useState(false)

  const running = status?.running ?? false
  const url = status?.url ?? comfyUrl

  const onCapture = (output: ComfyOutput): void => {
    if (!activeFrameId) return
    void captureOutput(activeFrameId, output)
    setCaptured((s) => new Set(s).add(output.url))
  }

  // Hard-reload the embedded page (bypass cache — a soft reload often leaves ComfyUI's
  // SPA looking unchanged). Reset webviewReady so the save / open-workflow hooks re-inject
  // on the next dom-ready. Falls back to a plain reload, then a fresh navigation.
  const reloadWebview = (): void => {
    setWebviewReady(false)
    const wv = webviewRef.current
    if (!wv) return
    try {
      wv.reloadIgnoringCache()
    } catch {
      try {
        wv.reload()
      } catch {
        wv.src = wv.getURL?.() || url
      }
    }
  }

  // Capture the live graph of the CURRENTLY ACTIVE ComfyUI tab into the frame it
  // belongs to — the core "forgot to Save" fix, made tab-safe. We attribute the graph
  // by the active workflow's name (not the frame Inline Studio last opened), so closing a
  // tab — which switches ComfyUI's active tab — can never write one frame's graph into
  // another. `hintFrameId` is used only for the saved-file fallback, which reads that
  // frame's own file and is therefore always safe.
  const captureLiveWorkflow = useCallback(
    async (hintFrameId: string | null, forceSaveToHint = false): Promise<void> => {
      const wv = webviewRef.current
      if (!wv) return
      let parsed: { name?: unknown; graph?: unknown } | null = null
      try {
        const raw = await wv.executeJavaScript(serializeActiveWorkflowScript())
        parsed =
          typeof raw === 'string' ? (JSON.parse(raw) as { name?: unknown; graph?: unknown }) : null
      } catch {
        parsed = null
      }
      const graph = parsed?.graph
      if (graph && typeof graph === 'object') {
        // Map the active tab to its frame by workflow name so the graph is attributed correctly.
        const name = typeof parsed?.name === 'string' ? activeWorkflowBaseName(parsed.name) : null
        const frame = name
          ? useFrameStore.getState().frames.find((f) => f.comfyWorkflowName === name)
          : undefined
        if (frame) await saveLiveWorkflow(frame.id, graph)
        // When switching away from a KNOWN frame (link/open replacing its tab) and the active tab
        // couldn't be name-matched, save the graph to that frame anyway — a guaranteed autosave so
        // opening the newly-linked workflow can't silently drop the previous frame's unsaved edits.
        else if (forceSaveToHint && hintFrameId) await saveLiveWorkflow(hintFrameId, graph)
        return
      }
      // No live graph (older ComfyUI): fall back to pulling the hinted frame's own
      // saved file — always its own file, so this never crosses tabs.
      if (hintFrameId) void pullWorkflow(hintFrameId)
    },
    [saveLiveWorkflow, pullWorkflow],
  )

  const check = async (): Promise<void> => {
    let down = false
    let nextUrl = comfyUrl
    try {
      const res = await window.inlineStudio.comfy.status()
      if (res.ok) {
        nextUrl = res.value.url
        down = !res.value.running
      } else {
        down = true
      }
    } catch {
      down = true
    }
    if (!down) {
      failures.current = 0
      setStatus({ running: true, url: nextUrl })
      return
    }
    // A single missed/slow ping must NOT flip `running` to false: that would unmount
    // the embedded ComfyUI page and lose in-progress work. ComfyUI's single-threaded
    // server routinely stalls past the ping timeout while generating, so only declare
    // it down after several consecutive failures (~12s at the 4s poll interval).
    failures.current += 1
    if (failures.current >= 3) setStatus({ running: false, url: nextUrl })
  }

  useEffect(() => {
    void load()
  }, [load])

  useEffect(() => {
    setDraftUrl(comfyUrl)
  }, [comfyUrl])

  useEffect(() => {
    void check()
    const timer = setInterval(() => void check(), 4000)
    return () => clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [comfyUrl])

  // Poll /history while running; when a NEW run finishes, show its outputs to capture.
  useEffect(() => {
    if (!running) return
    let cancelled = false
    const poll = async (): Promise<void> => {
      try {
        const res = await window.inlineStudio.comfy.latestRun()
        if (cancelled || !res.ok || !res.value) return
        const latest = res.value
        if (seenPromptId.current === null) {
          seenPromptId.current = latest.promptId // baseline: ignore pre-existing runs
        } else if (latest.promptId !== seenPromptId.current) {
          seenPromptId.current = latest.promptId
          if (latest.outputs.length > 0) {
            setRun(latest)
            setCaptured(new Set())
          }
        }
      } catch {
        // ignore transient errors
      }
    }
    void poll()
    const timer = setInterval(() => void poll(), 2000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [running])

  // Track when the embedded ComfyUI page has loaded enough to drive.
  useEffect(() => {
    const wv = webviewRef.current
    if (!wv) {
      setWebviewReady(false)
      return
    }
    const onReady = (): void => setWebviewReady(true)
    wv.addEventListener('dom-ready', onReady)
    return () => wv.removeEventListener('dom-ready', onReady)
  }, [running])

  // When a frame is linked (or changes), drive the embedded ComfyUI to open it.
  // Before replacing the open tab, capture its live graph so a switch can't lose the
  // previous frame's unsaved edits. `activeFrameId` is the frame being opened.
  useEffect(() => {
    if (!webviewReady || !linkedWorkflow || !webviewRef.current) return
    const wv = webviewRef.current
    const leaving = prevFrameRef.current
    const opening = activeFrameId
    // One-frame: clear after attempting so a remount can't replay a stale workflow
    // and re-select the wrong frame's tab.
    void (async () => {
      // Guaranteed autosave of the frame we're leaving before its tab is replaced, so linking a
      // new workflow (which opens a different tab) never loses the previous frame's unsaved edits.
      if (leaving && leaving !== opening) await captureLiveWorkflow(leaving, true)
      await wv.executeJavaScript(openWorkflowScript(linkedWorkflow)).catch(() => {})
      prevFrameRef.current = opening
    })().finally(() => setLinkedWorkflow(null))
  }, [webviewReady, linkedWorkflow, activeFrameId, captureLiveWorkflow, setLinkedWorkflow])

  // Install the save hook and listen for its marker: when the user saves a workflow
  // inside ComfyUI, pull the JSON back into Inline Studio's durable copy. The marker's
  // path identifies which frame's workflow was saved (fallback: the active frame).
  useEffect(() => {
    const wv = webviewRef.current
    if (!webviewReady || !wv) return
    wv.executeJavaScript(saveHookScript()).catch(() => {})
    const onConsole = (e: Event): void => {
      const msg = (e as unknown as { message?: string }).message ?? ''
      if (!msg.startsWith(WF_SAVED_MARKER)) return
      const savedPath = msg.slice(WF_SAVED_MARKER.length).trim()
      const frames = useFrameStore.getState().frames
      const match = frames.find(
        (f) => f.comfyWorkflowName && savedPath.includes(f.comfyWorkflowName),
      )
      const id = match?.id ?? prevFrameRef.current
      if (id) void pullWorkflow(id)
    }
    wv.addEventListener('console-message', onConsole)
    return () => wv.removeEventListener('console-message', onConsole)
  }, [webviewReady, pullWorkflow])

  // Debounced autosave: snapshot the live graph of the open frame on an interval, so a
  // user who never presses Save in ComfyUI still can't lose their work mid-edit.
  useEffect(() => {
    if (mode !== 'generate' || !webviewReady) return
    const timer = setInterval(() => void captureLiveWorkflow(prevFrameRef.current), 5000)
    return () => clearInterval(timer)
  }, [mode, webviewReady, captureLiveWorkflow])

  // Capture the open frame's live graph when leaving the Generate tab (a last snapshot
  // before the user navigates away).
  useEffect(() => {
    if (mode !== 'generate') void captureLiveWorkflow(prevFrameRef.current)
  }, [mode, captureLiveWorkflow])

  // Once ComfyUI has been reachable, remember it so the <webview> stays mounted even
  // across a transient drop — we overlay the guide rather than destroying the page.
  useEffect(() => {
    if (running) setEverConnected(true)
  }, [running])

  return (
    <div className="flex h-full flex-col bg-panel">
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <span
          className={`h-2 w-2 rounded-full ${running ? 'bg-green-500' : 'bg-zinc-600'}`}
          title={running ? 'ComfyUI is running' : 'ComfyUI is not reachable'}
        />
        <span className="text-xs uppercase tracking-wide text-zinc-400">ComfyUI</span>
        <input
          value={draftUrl}
          onChange={(e) => setDraftUrl(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') void setComfyUrl(draftUrl)
          }}
          spellCheck={false}
          className="ml-2 w-72 rounded border border-border bg-surface px-2 py-1 text-xs text-zinc-200 outline-none focus:border-accent"
        />
        <button
          onClick={() => void setComfyUrl(draftUrl)}
          className="rounded border border-border px-2 py-1 text-xs text-zinc-300 hover:bg-surface"
        >
          Save
        </button>
        <button
          onClick={reloadWebview}
          disabled={!running}
          title="Reload the embedded ComfyUI page"
          className="flex items-center gap-1 rounded border border-border px-2 py-1 text-xs text-zinc-300 hover:bg-surface disabled:opacity-40"
        >
          <RefreshIcon />
          Refresh
        </button>
      </div>

      <div className="relative flex-1">
        {running || everConnected ? (
          // Mounted once connected and never unmounted on a transient drop — destroying
          // and recreating this element is a full page reload that loses in-progress work.
          <webview
            ref={webviewRef}
            src={url}
            partition="persist:comfyui"
            className="h-full w-full border-0 bg-white"
          />
        ) : (
          <ConnectionGuide />
        )}

        {everConnected && !running && (
          <div className="absolute inset-0 z-20 bg-panel">
            <ConnectionGuide />
          </div>
        )}

        {run && run.outputs.length > 0 && (
          <CaptureStrip
            run={run}
            captured={captured}
            targetFrameName={activeFrame?.name ?? null}
            onCapture={onCapture}
            onDismiss={() => setRun(null)}
          />
        )}
      </div>
    </div>
  )
}

function RefreshIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-3.5 w-3.5"
    >
      <path d="M21 12a9 9 0 1 1-2.64-6.36" />
      <path d="M21 3v6h-6" />
    </svg>
  )
}

function CaptureStrip({
  run,
  captured,
  targetFrameName,
  onCapture,
  onDismiss,
}: {
  run: ComfyRun
  captured: Set<string>
  targetFrameName: string | null
  onCapture: (output: ComfyOutput) => void
  onDismiss: () => void
}): React.JSX.Element {
  // Dedupe by url so a tile (and its React key) is never repeated, even if the run
  // reports the same file under multiple nodes.
  const outputs = Array.from(new Map(run.outputs.map((o) => [o.url, o])).values())
  return (
    <div className="absolute inset-x-0 bottom-0 z-10 border-t border-border bg-panel/95 p-2 backdrop-blur">
      <div className="mb-1.5 flex items-center justify-between">
        <span className="text-[11px] text-zinc-300">
          New outputs ·{' '}
          {targetFrameName ? (
            <>
              capture to <span className="font-medium text-white">Frame {targetFrameName}</span>
            </>
          ) : (
            <span className="text-amber-400">open a frame's workflow to capture</span>
          )}
        </span>
        <button
          onClick={onDismiss}
          className="rounded border border-border px-2 py-0.5 text-[11px] text-zinc-300 hover:bg-surface"
        >
          Dismiss
        </button>
      </div>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {outputs.map((output) => (
          <CaptureTile
            key={output.url}
            output={output}
            captured={captured.has(output.url)}
            disabled={!targetFrameName}
            onCapture={() => onCapture(output)}
          />
        ))}
      </div>
    </div>
  )
}

function CaptureTile({
  output,
  captured,
  disabled,
  onCapture,
}: {
  output: ComfyOutput
  captured: boolean
  disabled: boolean
  onCapture: () => void
}): React.JSX.Element {
  return (
    <div className="group relative h-24 w-24 shrink-0 overflow-hidden rounded border border-border bg-black/40">
      {output.kind === 'video' ? (
        <video src={output.url} muted preload="metadata" className="h-full w-full object-cover" />
      ) : (
        <img src={output.url} alt="" className="h-full w-full object-cover" />
      )}
      <button
        onClick={onCapture}
        disabled={disabled || captured}
        className="absolute inset-0 flex items-center justify-center bg-black/60 text-[10px] font-medium text-white opacity-0 transition-opacity group-hover:opacity-100 disabled:cursor-not-allowed"
      >
        {captured ? '✓ Captured' : 'Update frame output'}
      </button>
      {captured && (
        <span className="absolute right-1 top-1 rounded bg-accent px-1 text-[9px] text-panel">
          ✓
        </span>
      )}
    </div>
  )
}
