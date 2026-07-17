import { useEffect, useMemo, useRef } from 'react'
import { getNodeDef } from '@shared/nodes/registry'
import { defaultParams } from '@shared/nodes/types'
import { useFrameStore } from '../../store/frameStore'
import { useGenerationStore } from '../../store/generationStore'
import { useSettingsDraft } from '../../lib/useSettingsDraft'
import { ParamWidget } from './ParamWidget'
import { SettingsHeader } from './SettingsHeader'

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

  const rootRef = useRef<HTMLDivElement>(null)
  const defId = def?.id
  const seed = useMemo(
    () => (def && frame ? { ...defaultParams(def), ...frame.params } : undefined),
    [def, frame],
  )
  const persist = (params: Record<string, unknown>): void => {
    if (frameId) void setParams(frameId, params)
  }
  const { local, dirty, change, apply } = useSettingsDraft(
    frameId ? `${frameId}:${defId ?? ''}` : null,
    seed,
    persist,
  )

  // Close when clicking anywhere outside the panel — except the node's adjust (toggle) button,
  // which manages open/close itself (so a click there doesn't close-then-reopen). Flush pending
  // edits before closing so a click-away never drops the last change.
  useEffect(() => {
    if (!frameId) return
    const onDown = (e: PointerEvent): void => {
      const target = e.target as HTMLElement | null
      if (!target || rootRef.current?.contains(target)) return
      if (target.closest('[data-gen-settings-toggle]')) return
      apply()
      close()
    }
    document.addEventListener('pointerdown', onDown)
    return () => document.removeEventListener('pointerdown', onDown)
  }, [frameId, close, apply])

  if (!frameId || !frame || !def) return null

  return (
    <div
      ref={rootRef}
      className="absolute right-0 top-0 z-40 flex h-full w-72 flex-col border-l border-border bg-panel shadow-2xl"
    >
      <SettingsHeader title={def.title} dirty={dirty} onUpdate={apply} onClose={close} />
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
              onCommit={(v) => change(field.key, v)}
            />
          ))
        )}
      </div>
    </div>
  )
}
