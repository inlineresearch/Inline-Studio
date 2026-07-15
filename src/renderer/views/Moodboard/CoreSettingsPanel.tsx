import { useEffect, useMemo, useRef, useState } from 'react'
import { useGenerationStore } from '../../store/generationStore'
import { useMoodboardStore } from '../../store/moodboardStore'
import { useCoreNodesStore } from '../../store/coreNodesStore'
import { CoreParamWidget } from './CoreParamWidget'
import { XIcon } from './nodes/NodeBadge'

/**
 * Right-hand settings sidebar for an Inline Core node, opened by the node's adjust (sliders) icon —
 * the same interaction the Generate (fal) node uses, so both read identically. Shows every param the
 * node's descriptor declares (the node face itself stays clean). Edits persist to the moodboard
 * item's `data.core.params`.
 */
export function CoreSettingsPanel(): React.JSX.Element | null {
  const itemId = useGenerationStore((s) => s.settingsCoreItemId)
  const close = useGenerationStore((s) => s.closeCoreSettings)
  const item = useMoodboardStore((s) => s.items.find((i) => i.id === itemId))
  const updateItem = useMoodboardStore((s) => s.updateItem)
  const core = item?.type === 'core' ? item.data.core : undefined
  const descriptor = useCoreNodesStore((s) =>
    core ? s.descriptors.find((d) => d.type === core.type) : undefined,
  )

  const defaults = useMemo(
    () => Object.fromEntries((descriptor?.params ?? []).map((p) => [p.key, p.default])),
    [descriptor],
  )

  const [local, setLocal] = useState<Record<string, unknown>>({})
  const rootRef = useRef<HTMLDivElement>(null)
  const coreType = core?.type
  useEffect(() => {
    if (core) setLocal({ ...defaults, ...core.params })
    // Re-seed when the open node (or its type) changes.
  }, [itemId, coreType]) // eslint-disable-line react-hooks/exhaustive-deps

  // Close when clicking outside the panel — except the node's adjust (toggle) button, which manages
  // open/close itself (shared attribute with the fal settings panel; only one is open at a time).
  useEffect(() => {
    if (!itemId) return
    const onDown = (e: PointerEvent): void => {
      const target = e.target as HTMLElement | null
      if (!target || rootRef.current?.contains(target)) return
      if (target.closest('[data-gen-settings-toggle]')) return
      close()
    }
    document.addEventListener('pointerdown', onDown)
    return () => document.removeEventListener('pointerdown', onDown)
  }, [itemId, close])

  if (!itemId || !item || !core || !descriptor) return null

  const commit = (key: string, value: string | number | boolean): void =>
    setLocal((prev) => {
      const next = { ...prev, [key]: value }
      // Preserve `output` (and any other core fields); only params change.
      void updateItem(itemId, { data: { ...item.data, core: { ...core, params: next } } })
      return next
    })
  const change = (key: string, value: string | number | boolean): void =>
    setLocal((prev) => ({ ...prev, [key]: value }))

  return (
    <div
      ref={rootRef}
      className="absolute right-0 top-0 z-40 flex h-full w-72 flex-col border-l border-border bg-panel shadow-2xl"
    >
      <header className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex min-w-0 flex-col">
          <span className="text-xs font-semibold text-zinc-100">Settings</span>
          <span className="truncate text-[10px] text-zinc-500">{descriptor.title}</span>
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
        {descriptor.params.length === 0 ? (
          <p className="text-[11px] text-zinc-500">This node has no adjustable settings.</p>
        ) : (
          descriptor.params.map((field) => (
            <CoreParamWidget
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
