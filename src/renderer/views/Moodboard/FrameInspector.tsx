import { useEffect, useState } from 'react'
import { studio } from '@/lib/studio'
import { resolveMedia } from '@/lib/media'
import { takeWaveformPath } from '@shared/media'
import type { Asset } from '@shared/types'
import { useUiStore } from '../../store/uiStore'
import { useFrameStore } from '../../store/frameStore'
import { useAssetStore } from '../../store/assetStore'
import { useSettingsStore } from '../../store/settingsStore'
import { getAssetDragIds } from '../../lib/dnd'
import { requireComfyConnected } from '../../lib/requireComfyConnected'
import { Waveform } from '../../components/Waveform'

const INPUT_DND = 'application/x-inlinestudio-frame-input'

/**
 * Right-side drawer to edit a frame: rename, clone, manage inputs (add via drop /
 * remove / reorder), pick the hero output, open the workflow JSON, and link/unlink
 * the ComfyUI workflow. Driven by uiStore.inspectorFrameId.
 */
export function FrameInspector(): React.JSX.Element | null {
  const frameId = useUiStore((s) => s.inspectorFrameId)
  const close = useUiStore((s) => s.setInspectorFrame)
  const setMode = useUiStore((s) => s.setMode)
  const setLinkedWorkflow = useUiStore((s) => s.setLinkedWorkflow)
  const setActiveFrame = useUiStore((s) => s.setActiveFrame)

  const frame = useFrameStore((s) => s.frames.find((f) => f.id === frameId))
  const inputs = useFrameStore((s) => (frameId ? s.inputsByFrame[frameId] : undefined)) ?? []
  const takes = useFrameStore((s) => (frameId ? s.takesByFrame[frameId] : undefined)) ?? []
  const busy = useFrameStore((s) => s.busyId === frameId)
  const assets = useAssetStore((s) => s.assets)
  const comfyUrl = useSettingsStore((s) => s.comfyUrl)

  const rename = useFrameStore((s) => s.rename)
  const addInputs = useFrameStore((s) => s.addInputs)
  const removeInput = useFrameStore((s) => s.removeInput)
  const reorderInputs = useFrameStore((s) => s.reorderInputs)
  const setHero = useFrameStore((s) => s.setHero)
  const deleteTake = useFrameStore((s) => s.deleteTake)
  const clone = useFrameStore((s) => s.clone)
  const unlink = useFrameStore((s) => s.unlink)
  const linkFrame = useFrameStore((s) => s.linkFrame)
  const uploadInputs = useFrameStore((s) => s.uploadInputs)

  const [draftName, setDraftName] = useState('')
  useEffect(() => {
    setDraftName(frame?.name ?? '')
  }, [frame?.name])

  if (!frameId || !frame) return null

  const inputAssets = inputs
    .map((i) => assets.find((a) => a.id === i.assetId))
    .filter((a): a is Asset => !!a)
  const orderedIds = inputAssets.map((a) => a.id)
  const linked = !!frame.comfyWorkflowName
  // Linked only means a name was reserved + a seed pushed. "Ready" means the user has
  // actually built a real workflow — so we don't imply work exists when it doesn't.
  const ready = frame.comfyWorkflowReady

  const commitName = (): void => {
    const next = draftName.trim()
    if (next && next !== frame.name) void rename(frame.id, next)
  }

  const onReorderDrop =
    (targetId: string) =>
    (e: React.DragEvent): void => {
      e.preventDefault()
      const dragged = e.dataTransfer.getData(INPUT_DND)
      if (!dragged || dragged === targetId) return
      const next = orderedIds.filter((id) => id !== dragged)
      next.splice(next.indexOf(targetId), 0, dragged)
      void reorderInputs(frame.id, next)
    }

  const onAddDrop = (e: React.DragEvent): void => {
    e.preventDefault()
    const ids = getAssetDragIds(e.dataTransfer).filter((id) => !orderedIds.includes(id))
    if (ids.length) void addInputs(frame.id, ids)
  }

  const onLinkOpen = async (): Promise<void> => {
    // ComfyUI must be reachable to link/open a workflow — otherwise send the user to the
    // Generate tab to connect first.
    if (!(await requireComfyConnected(() => setMode('generate')))) return
    const result = await linkFrame(frame.id)
    setLinkedWorkflow(result?.comfyWorkflowName ?? frame.comfyWorkflowName)
    setActiveFrame(frame.id)
    setMode('generate')
    void uploadInputs(frame.id)
  }

  const openJson = (): void => {
    if (!frame.comfyWorkflowName) return
    const url = `${comfyUrl.replace(/\/+$/, '')}/userdata/${encodeURIComponent(
      `workflows/${frame.comfyWorkflowName}.json`,
    )}`
    void studio().shell.openExternal(url)
  }

  return (
    <div className="absolute right-0 top-0 z-20 flex h-full w-80 flex-col border-l border-border bg-panel shadow-xl">
      {/* Title */}
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <input
          value={draftName}
          onChange={(e) => setDraftName(e.target.value)}
          onBlur={commitName}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
          }}
          className="min-w-0 flex-1 rounded border border-transparent bg-transparent px-1 py-0.5 text-sm font-semibold text-white outline-none hover:border-border focus:border-accent"
        />
        <button
          onClick={() => void clone(frame.id)}
          title="Clone this frame (inputs + workflow)"
          className="rounded border border-border px-2 py-1 text-[11px] text-zinc-300 hover:bg-surface"
        >
          Clone
        </button>
        <button
          onClick={() => close(null)}
          title="Close"
          className="px-1 text-zinc-500 hover:text-white"
        >
          ✕
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {/* Inputs */}
        <Section title="Inputs">
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={onAddDrop}
            className="flex flex-wrap gap-2 rounded border border-dashed border-border p-2"
          >
            {inputAssets.map((a) => (
              <div
                key={a.id}
                draggable
                onDragStart={(e) => e.dataTransfer.setData(INPUT_DND, a.id)}
                onDragOver={(e) => e.preventDefault()}
                onDrop={onReorderDrop(a.id)}
                title={`${a.name} — drag to reorder`}
                className="group relative h-16 w-16 cursor-grab overflow-hidden rounded border border-border bg-black/40"
              >
                <Media
                  url={resolveMedia(a.previewPath ?? a.filePath)}
                  kind={a.kind}
                  poster={a.kind === 'video' && a.thumbPath ? resolveMedia(a.thumbPath) : undefined}
                  waveform={
                    a.kind === 'audio' && a.thumbPath ? resolveMedia(a.thumbPath) : undefined
                  }
                />
                {inputAssets.length > 1 && (
                  <button
                    onClick={() => void removeInput(frame.id, a.id)}
                    title="Remove input"
                    className="absolute right-0.5 top-0.5 hidden h-4 w-4 items-center justify-center rounded-full bg-black/80 text-[10px] text-zinc-200 hover:text-red-400 group-hover:flex"
                  >
                    ✕
                  </button>
                )}
              </div>
            ))}
            <span className="self-center text-[10px] text-zinc-600">Drop assets here to add</span>
          </div>
        </Section>

        {/* Outputs */}
        <Section title={`Outputs${takes.length ? ` (${takes.length})` : ''}`}>
          {takes.length === 0 ? (
            <p className="text-[11px] text-zinc-600">
              No outputs yet — generate in ComfyUI and capture.
            </p>
          ) : (
            <div className="grid grid-cols-3 gap-2">
              {takes.map((t) => {
                const isHero = t.id === frame.heroTakeId
                return (
                  <div
                    key={t.id}
                    className={`group relative aspect-square overflow-hidden rounded border bg-black/40 ${
                      isHero ? 'border-accent ring-1 ring-accent' : 'border-border'
                    }`}
                  >
                    <button
                      onClick={() => void setHero(frame.id, t.id)}
                      title={isHero ? 'Hero output' : 'Set as hero output'}
                      className="h-full w-full"
                    >
                      <Media
                        url={resolveMedia(t.filePath)}
                        kind={t.kind}
                        waveform={
                          t.kind === 'audio' ? resolveMedia(takeWaveformPath(t.id)) : undefined
                        }
                      />
                    </button>
                    {isHero && (
                      <span className="absolute left-0.5 top-0.5 rounded bg-accent px-1 text-[8px] text-panel">
                        Hero
                      </span>
                    )}
                    <button
                      onClick={() => void deleteTake(t.id)}
                      title="Delete take"
                      className="absolute right-0.5 top-0.5 hidden h-4 w-4 items-center justify-center rounded-full bg-black/80 text-[10px] text-zinc-200 hover:text-red-400 group-hover:flex"
                    >
                      ✕
                    </button>
                  </div>
                )
              })}
            </div>
          )}
        </Section>

        {/* Workflow */}
        <Section title="Workflow">
          {linked ? (
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center gap-1.5 text-[11px]">
                <span
                  className={`h-1.5 w-1.5 rounded-full ${ready ? 'bg-green-500' : 'bg-amber-500'}`}
                />
                <span className={ready ? 'text-zinc-400' : 'text-amber-400'}>
                  {ready ? 'Workflow ready' : 'Not built yet — open and add nodes, edits autosave'}
                </span>
              </div>
              <button
                onClick={() => void onLinkOpen()}
                disabled={busy}
                className="rounded bg-accent px-2 py-1.5 text-xs font-medium text-panel hover:brightness-110 disabled:opacity-40"
              >
                {busy ? '…' : ready ? 'Edit in ComfyUI' : 'Build in ComfyUI'}
              </button>
              <button
                onClick={openJson}
                className="rounded border border-border px-2 py-1.5 text-xs text-zinc-200 hover:bg-surface"
              >
                Open workflow.json ↗
              </button>
              <button
                onClick={() => void unlink(frame.id)}
                className="rounded border border-border px-2 py-1.5 text-xs text-zinc-400 hover:border-red-500/50 hover:text-red-400"
              >
                Unlink workflow
              </button>
            </div>
          ) : (
            <button
              onClick={() => void onLinkOpen()}
              disabled={busy}
              className="w-full rounded bg-accent px-2 py-1.5 text-xs font-medium text-panel hover:brightness-110 disabled:opacity-40"
            >
              {busy ? '…' : 'Link workflow'}
            </button>
          )}
        </Section>
      </div>
    </div>
  )
}

function Section({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <div className="mb-4">
      <h3 className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-zinc-500">
        {title}
      </h3>
      {children}
    </div>
  )
}

function Media({
  url,
  kind,
  poster,
  waveform,
}: {
  url: string
  kind: 'image' | 'video' | 'audio'
  poster?: string
  waveform?: string
}): React.JSX.Element {
  if (kind === 'video')
    return (
      <video
        src={url}
        poster={poster}
        muted
        preload="metadata"
        className="h-full w-full object-cover"
      />
    )
  if (kind === 'audio')
    return (
      <span className="flex h-full w-full items-center justify-center p-1">
        <Waveform url={waveform ?? null} className="h-1/2 w-full text-emerald-400" />
      </span>
    )
  return <img src={url} alt="" className="h-full w-full object-cover" />
}
