import { useEffect, useState } from 'react'
import { resolveMedia } from '@/lib/media'
import { takeWaveformPath } from '@shared/media'
import type { Asset } from '@shared/types'
import { useUiStore } from '../../store/uiStore'
import { useFrameStore } from '../../store/frameStore'
import { useAssetStore } from '../../store/assetStore'
import { getAssetDragIds } from '../../lib/dnd'
import { Waveform } from '../../components/Waveform'
import { useAutoCommit } from '../../lib/useAutoCommit'

const INPUT_DND = 'application/x-inlinestudio-frame-input'

/**
 * Right-side drawer to edit a frame: rename, clone, manage inputs (add via drop /
 * remove / reorder), and pick the hero output. Driven by uiStore.inspectorFrameId.
 */
export function FrameInspector(): React.JSX.Element | null {
  const frameId = useUiStore((s) => s.inspectorFrameId)
  const close = useUiStore((s) => s.setInspectorFrame)

  const frame = useFrameStore((s) => s.frames.find((f) => f.id === frameId))
  const inputs = useFrameStore((s) => (frameId ? s.inputsByFrame[frameId] : undefined)) ?? []
  const takes = useFrameStore((s) => (frameId ? s.takesByFrame[frameId] : undefined)) ?? []
  const assets = useAssetStore((s) => s.assets)

  const rename = useFrameStore((s) => s.rename)
  const addInputs = useFrameStore((s) => s.addInputs)
  const removeInput = useFrameStore((s) => s.removeInput)
  const reorderInputs = useFrameStore((s) => s.reorderInputs)
  const setHero = useFrameStore((s) => s.setHero)
  const deleteTake = useFrameStore((s) => s.deleteTake)
  const clone = useFrameStore((s) => s.clone)

  const [draftName, setDraftName] = useState('')
  useEffect(() => {
    setDraftName(frame?.name ?? '')
  }, [frame?.name])

  // Above the early return, as hooks must be: the name is committed as it is typed, not on blur.
  const { schedule, flush } = useAutoCommit(() => {
    const next = draftName.trim()
    if (frame && next && next !== frame.name) void rename(frame.id, next)
  })

  if (!frameId || !frame) return null

  const inputAssets = inputs
    .map((i) => assets.find((a) => a.id === i.assetId))
    .filter((a): a is Asset => !!a)
  const orderedIds = inputAssets.map((a) => a.id)

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

  return (
    <div className="absolute right-0 top-0 z-20 flex h-full w-80 flex-col border-l border-border bg-panel shadow-xl">
      {/* Title */}
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <input
          value={draftName}
          onChange={(e) => {
            setDraftName(e.target.value)
            schedule()
          }}
          onBlur={flush}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
          }}
          className="min-w-0 flex-1 rounded border border-transparent bg-transparent px-1 py-0.5 text-sm font-semibold text-white outline-none hover:border-border focus:border-accent"
        />
        <button
          onClick={() => void clone(frame.id)}
          title="Clone this frame (inputs)"
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
                title={`${a.name} - drag to reorder`}
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
            <p className="text-[11px] text-zinc-600">No outputs yet.</p>
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
