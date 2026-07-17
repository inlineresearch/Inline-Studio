import { useEffect, useMemo, useRef } from 'react'
import { useGenerationStore } from '../../store/generationStore'
import { useMoodboardStore } from '../../store/moodboardStore'
import { useCoreNodesStore } from '../../store/coreNodesStore'
import { useSettingsDraft } from '../../lib/useSettingsDraft'
import { CoreParamWidget } from './CoreParamWidget'
import { SettingsHeader } from './SettingsHeader'

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

  const rootRef = useRef<HTMLDivElement>(null)
  const coreType = core?.type
  const seed = useMemo(() => (core ? { ...defaults, ...core.params } : undefined), [defaults, core])
  // Preserve `output` (and any other core fields); only params change.
  const persist = (params: Record<string, unknown>): void => {
    if (!itemId || !item || !core) return
    void updateItem(itemId, { data: { ...item.data, core: { ...core, params } } })
  }
  const { local, dirty, change, apply } = useSettingsDraft(
    itemId ? `${itemId}:${coreType ?? ''}` : null,
    seed,
    persist,
  )

  // Close when clicking outside the panel — except the node's adjust (toggle) button, which manages
  // open/close itself (shared attribute with the fal settings panel; only one is open at a time).
  // Flush pending edits before closing so a click-away never drops the last change.
  useEffect(() => {
    if (!itemId) return
    const onDown = (e: PointerEvent): void => {
      const target = e.target as HTMLElement | null
      if (!target || rootRef.current?.contains(target)) return
      if (target.closest('[data-gen-settings-toggle]')) return
      apply()
      close()
    }
    document.addEventListener('pointerdown', onDown)
    return () => document.removeEventListener('pointerdown', onDown)
  }, [itemId, close, apply])

  if (!itemId || !item || !core || !descriptor) return null

  return (
    <div
      ref={rootRef}
      className="absolute right-0 top-0 z-40 flex h-full w-72 flex-col border-l border-border bg-panel shadow-2xl"
    >
      <SettingsHeader title={descriptor.title} dirty={dirty} onUpdate={apply} onClose={close} />
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
              onCommit={(v) => change(field.key, v)}
            />
          ))
        )}
      </div>
    </div>
  )
}
