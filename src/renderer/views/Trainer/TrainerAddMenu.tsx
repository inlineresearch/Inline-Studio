/**
 * The Trainer's node list, opened from the toolbar's + or by double-clicking empty canvas.
 *
 * Mirrors the Studio Add menu's shape and placement rules rather than sharing it: the Studio one is
 * built around fal models and Core descriptors, while the Trainer's four node kinds are a fixed
 * list. What it does share is `useMenuPlacement`, so a list opened near the bottom edge flips and
 * caps instead of running off the canvas.
 */
import { useMemo } from 'react'

import { useMenuPlacement } from '../Moodboard/useMenuPlacement'

export interface AddEntry<K extends string> {
  kind: K
  label: string
  icon: React.JSX.Element
  category: string
}

export function TrainerAddMenu<K extends string>({
  x,
  y,
  above = false,
  entries,
  onPick,
  onClose,
}: {
  /** Container-relative anchor point (px). */
  x: number
  y: number
  /** True when the caller centres on x and grows upward (the toolbar button). */
  above?: boolean
  entries: readonly AddEntry<K>[]
  onPick: (kind: K) => void
  onClose: () => void
}): React.JSX.Element {
  const place = useMenuPlacement<HTMLDivElement>(x, y, above)
  const groups = useMemo(() => {
    const out = new Map<string, AddEntry<K>[]>()
    for (const entry of entries)
      out.set(entry.category, [...(out.get(entry.category) ?? []), entry])
    return [...out.entries()]
  }, [entries])

  return (
    <>
      <div className="absolute inset-0 z-20" onClick={onClose} />
      <div
        ref={place.ref}
        className={`absolute z-30 flex w-52 select-none flex-col overflow-hidden rounded-md border border-border bg-panel text-xs shadow-xl ${
          above ? '-translate-x-1/2 -translate-y-full' : place.flipped ? '-translate-y-full' : ''
        }`}
        style={place.style}
      >
        {/* `nowheel` keeps React Flow from swallowing the wheel and zooming instead of scrolling. */}
        <div className="nowheel overflow-y-auto p-1" style={{ maxHeight: place.maxHeight }}>
          {groups.map(([category, group]) => (
            <div key={category}>
              <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-zinc-500">
                {category}
              </div>
              {group.map((entry) => (
                <button
                  key={entry.kind}
                  onClick={() => {
                    onPick(entry.kind)
                    onClose()
                  }}
                  className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-xs text-zinc-200 hover:bg-surface"
                >
                  <span className="text-zinc-400">{entry.icon}</span>
                  {entry.label}
                </button>
              ))}
            </div>
          ))}
        </div>
      </div>
    </>
  )
}
