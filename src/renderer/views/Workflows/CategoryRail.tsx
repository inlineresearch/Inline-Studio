import type { WorkflowCategory, WorkflowSummary } from '@shared/types'

const ITEM =
  'flex w-full items-center justify-between gap-2 rounded-md px-2.5 py-1.5 text-left text-[12px] transition-colors'

function Item({
  label,
  count,
  active,
  onClick,
}: {
  label: string
  count: number
  active: boolean
  onClick: () => void
}): React.JSX.Element {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      className={`${ITEM} ${
        active ? 'bg-panel text-white' : 'text-zinc-400 hover:bg-panel/60 hover:text-zinc-200'
      }`}
    >
      <span className="truncate">{label}</span>
      <span className="shrink-0 text-[10px] text-zinc-600">{count}</span>
    </button>
  )
}

function Heading({ children }: { children: React.ReactNode }): React.JSX.Element {
  return (
    <h3 className="px-2.5 pb-1 pt-3 text-[10px] font-semibold uppercase tracking-wide text-zinc-600">
      {children}
    </h3>
  )
}

/**
 * The category rail: two blocks, because the catalogue's categories run on two axes - what a
 * workflow makes, and which model family it runs on. Drawn as one flat list they read as a pile.
 */
export function CategoryRail({
  categories,
  entries,
  selected,
  onSelect,
}: {
  categories: WorkflowCategory[]
  entries: WorkflowSummary[]
  selected: string | null
  onSelect: (slug: string | null) => void
}): React.JSX.Element {
  const counts = new Map<string, number>()
  for (const entry of entries) {
    for (const slug of entry.categories) counts.set(slug, (counts.get(slug) ?? 0) + 1)
  }

  // A category nothing published sits in reads as broken, so it never reaches the rail.
  const used = categories.filter((c) => (counts.get(c.slug) ?? 0) > 0)
  // A catalogue that predates the `kind` column declares none, and a heading over one undivided
  // list says less than no heading at all.
  const grouped = used.some((c) => c.kind)
  const types = grouped ? used.filter((c) => c.kind !== 'model') : used
  const models = grouped ? used.filter((c) => c.kind === 'model') : []

  return (
    <nav className="flex w-48 shrink-0 flex-col overflow-y-auto border-r border-border p-2">
      <Item
        label="All"
        count={entries.length}
        active={selected === null}
        onClick={() => onSelect(null)}
      />

      {grouped && types.length > 0 ? <Heading>Type</Heading> : null}
      {types.map((c) => (
        <Item
          key={c.slug}
          label={c.name}
          count={counts.get(c.slug) ?? 0}
          active={selected === c.slug}
          onClick={() => onSelect(c.slug)}
        />
      ))}

      {grouped && models.length > 0 ? <Heading>Model</Heading> : null}
      {models.map((c) => (
        <Item
          key={c.slug}
          label={c.name}
          count={counts.get(c.slug) ?? 0}
          active={selected === c.slug}
          onClick={() => onSelect(c.slug)}
        />
      ))}
    </nav>
  )
}
