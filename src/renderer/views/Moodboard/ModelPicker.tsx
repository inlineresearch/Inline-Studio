import { useEffect, useRef, useState } from 'react'
import { listNodeDefs, groupByOwner } from '@shared/nodes/registry'

function InfoIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-3.5 w-3.5 shrink-0"
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4M12 8h.01" />
    </svg>
  )
}

function ChevronDownIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-2.5 w-2.5 shrink-0"
    >
      <path d="M6 9l6 6 6-6" />
    </svg>
  )
}

function CheckIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-3 w-3 shrink-0 text-accent"
    >
      <path d="M20 6L9 17l-5-5" />
    </svg>
  )
}

/** A searchable model selector for a Generate node - lists the registered fal models. */
export function ModelPicker({
  modelId,
  onSelect,
  onShowInfo,
}: {
  modelId: string | null
  onSelect: (id: string) => void
  /** Open the info sidebar (inputs/outputs/cost) for a model, from its row's hint icon. */
  onShowInfo: (id: string) => void
}): React.JSX.Element {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const rootRef = useRef<HTMLDivElement>(null)
  const defs = listNodeDefs()
  const current = defs.find((d) => d.id === modelId)
  const q = query.trim().toLowerCase()
  const filtered = q
    ? defs.filter((d) => d.title.toLowerCase().includes(q) || d.category.toLowerCase().includes(q))
    : defs

  // Close on any pointer-down outside the picker. A `fixed inset-0` backdrop doesn't work here -
  // the dropdown lives inside a React Flow node whose viewport is CSS-transformed, so `fixed` is
  // contained by that transform and canvas clicks miss it.
  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent): void => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', onDown)
    return () => document.removeEventListener('pointerdown', onDown)
  }, [open])

  return (
    // `flex-1 min-w-0` so the trigger below can measure the footer rather than a fixed cap: a node
    // wide enough to show "Seedance 2.0 · Reference → Video" should show it.
    <div className="relative min-w-0 flex-1" ref={rootRef}>
      <button
        onClick={() => setOpen((v) => !v)}
        // inline-flex keeps the pill hugging a short title; max-w-full stops it at the node edge.
        className="nodrag inline-flex max-w-full items-center gap-1 rounded-md bg-black/40 px-2 py-1 text-[11px] font-medium text-zinc-200 hover:bg-black/60"
        title={current ? `${current.title} - choose a model` : 'Choose a model'}
      >
        <span className="truncate">{current?.title ?? 'Select model'}</span>
        <ChevronDownIcon />
      </button>
      {open && (
        <>
          <div className="nowheel absolute bottom-full left-0 z-40 mb-1 w-64 overflow-hidden rounded-lg border border-border bg-panel shadow-xl">
            <input
              autoFocus
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search models…"
              className="nodrag w-full border-b border-border bg-transparent px-3 py-2 text-[11px] text-zinc-100 outline-none placeholder:text-zinc-500"
            />
            <div className="max-h-72 overflow-y-auto py-1">
              {/* Models grouped under their provider (owner). */}
              {groupByOwner(filtered).map((group) => (
                <div key={group.owner}>
                  <div className="px-3 pb-0.5 pt-1.5 text-[9px] font-semibold uppercase tracking-wide text-zinc-500">
                    {group.label}
                  </div>
                  {group.defs.map((d) => (
                    <button
                      key={d.id}
                      onClick={() => {
                        onSelect(d.id)
                        setOpen(false)
                        setQuery('')
                      }}
                      className="nodrag flex w-full items-center justify-between gap-2 px-3 py-1.5 text-left hover:bg-surface"
                    >
                      <span className="flex min-w-0 flex-col">
                        <span className="truncate text-[11px] text-zinc-100">{d.title}</span>
                        <span className="text-[10px] text-zinc-500">{d.category}</span>
                      </span>
                      <span className="flex shrink-0 items-center gap-1.5">
                        {/* Hint icon → opens the model's inputs/outputs/cost sidebar. */}
                        <button
                          type="button"
                          title="View inputs, outputs & cost"
                          className="flex items-center text-zinc-500 hover:text-zinc-200"
                          onClick={(e) => {
                            e.stopPropagation()
                            onShowInfo(d.id)
                            setOpen(false)
                            setQuery('')
                          }}
                        >
                          <InfoIcon />
                        </button>
                        {d.id === modelId && <CheckIcon />}
                      </span>
                    </button>
                  ))}
                </div>
              ))}
              {filtered.length === 0 && (
                <div className="px-3 py-2 text-[11px] text-zinc-500">No models found.</div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
