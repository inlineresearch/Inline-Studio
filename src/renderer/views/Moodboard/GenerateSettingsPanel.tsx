import { useEffect, useRef, useState } from 'react'
import { getNodeDef } from '@shared/nodes/registry'
import { defaultParams } from '@shared/nodes/types'
import { useFrameStore } from '../../store/frameStore'
import { useGenerationStore } from '../../store/generationStore'
import { ParamWidget } from './ParamWidget'
import { XIcon } from './nodes/NodeBadge'

/**
 * Right-hand settings sidebar for a Generate node, opened by the node's adjust (sliders) icon.
 * Shows every param for the node's current model. The prompt lives on a connected Prompt node, so
 * it isn't here. Model selection stays on the node itself.
 */
export function GenerateSettingsPanel(): React.JSX.Element | null {
  const frameId = useGenerationStore((s) => s.settingsFrameId)
  const close = useGenerationStore((s) => s.closeSettings)
  const setParams = useGenerationStore((s) => s.setParams)
  const frame = useFrameStore((s) => s.frames.find((f) => f.id === frameId))
  const def = frame?.modelId ? getNodeDef(frame.modelId) : undefined

  const [local, setLocal] = useState<Record<string, unknown>>({})
  const rootRef = useRef<HTMLDivElement>(null)
  const defId = def?.id
  useEffect(() => {
    if (def && frame) setLocal({ ...defaultParams(def), ...frame.params })
    // Re-seed when the open node or its model changes.
  }, [frameId, defId]) // eslint-disable-line react-hooks/exhaustive-deps

  // Close when clicking anywhere outside the panel — except the node's adjust (toggle) button,
  // which manages open/close itself (so a click there doesn't close-then-reopen).
  useEffect(() => {
    if (!frameId) return
    const onDown = (e: PointerEvent): void => {
      const target = e.target as HTMLElement | null
      if (!target || rootRef.current?.contains(target)) return
      if (target.closest('[data-gen-settings-toggle]')) return
      close()
    }
    document.addEventListener('pointerdown', onDown)
    return () => document.removeEventListener('pointerdown', onDown)
  }, [frameId, close])

  if (!frameId || !frame || !def) return null

  const change = (key: string, value: string | number | boolean): void =>
    setLocal((prev) => ({ ...prev, [key]: value }))
  const commit = (key: string, value: string | number | boolean): void =>
    setLocal((prev) => {
      const next = { ...prev, [key]: value }
      void setParams(frameId, next)
      return next
    })

  return (
    <div
      ref={rootRef}
      className="absolute right-0 top-0 z-40 flex h-full w-72 flex-col border-l border-border bg-panel shadow-2xl"
    >
      <header className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex min-w-0 flex-col">
          <span className="text-xs font-semibold text-zinc-100">Settings</span>
          <span className="truncate text-[10px] text-zinc-500">{def.title}</span>
        </div>
        <button
          onClick={close}
          className="flex h-6 w-6 items-center justify-center rounded text-zinc-400 hover:bg-surface hover:text-zinc-100"
          aria-label="Close settings"
        >
          <XIcon className="h-3.5 w-3.5" />
        </button>
      </header>
      <div className="flex flex-1 flex-col gap-3 overflow-y-auto p-3">
        {def.params.length === 0 ? (
          <p className="text-[11px] text-zinc-500">This model has no adjustable settings.</p>
        ) : (
          def.params.map((field) => (
            <ParamWidget
              key={field.key}
              field={field}
              value={local[field.key]}
              onChange={(v) => change(field.key, v)}
              onCommit={(v) => commit(field.key, v)}
            />
          ))
        )}
      </div>
    </div>
  )
}
