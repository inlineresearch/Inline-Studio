/** The Workflows popup: browse the published catalogue and import one onto the canvas. */
import { useMemo } from 'react'
import type { WorkflowDetail, WorkflowSort } from '@shared/types'
import { Modal } from '../../components/Modal'
import { useWorkflowsStore } from '../../store/workflowsStore'
import { CategoryRail } from './CategoryRail'
import { WorkflowCard } from './WorkflowCard'

const SORTS: { key: WorkflowSort; label: string }[] = [
  { key: 'views', label: 'Most viewed' },
  { key: 'downloads', label: 'Most downloaded' },
  { key: 'newest', label: 'Newest' },
]

// Fixed margins rather than a fixed height: the panel grows with the window and stops at 980px.
const PANEL = 'h-[calc(100vh-160px)] w-[min(980px,calc(100vw-160px))]'

export function WorkflowsDialog({
  onImport,
}: {
  onImport: (detail: WorkflowDetail) => void
}): React.JSX.Element {
  const open = useWorkflowsStore((s) => s.open)
  const setOpen = useWorkflowsStore((s) => s.setOpen)
  const sort = useWorkflowsStore((s) => s.sort)
  const setSort = useWorkflowsStore((s) => s.setSort)
  const category = useWorkflowsStore((s) => s.category)
  const setCategory = useWorkflowsStore((s) => s.setCategory)
  const catalogue = useWorkflowsStore((s) => s.catalogue)
  const loading = useWorkflowsStore((s) => s.loading)
  const error = useWorkflowsStore((s) => s.error)
  const importing = useWorkflowsStore((s) => s.importing)
  const load = useWorkflowsStore((s) => s.load)
  const ensureDetail = useWorkflowsStore((s) => s.ensureDetail)
  const setImporting = useWorkflowsStore((s) => s.setImporting)
  const importError = useWorkflowsStore((s) => s.importError)
  const setImportError = useWorkflowsStore((s) => s.setImportError)

  const entries = useMemo(() => catalogue?.entries ?? [], [catalogue])
  const shown = useMemo(
    () => (category ? entries.filter((e) => e.categories.includes(category)) : entries),
    [entries, category],
  )

  return (
    <Modal
      open={open}
      onClose={() => setOpen(false)}
      title="Workflows"
      panelClassName={PANEL}
      bodyClassName="flex min-h-0 flex-1"
      headerAction={
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value as WorkflowSort)}
          aria-label="Sort workflows"
          className="rounded-md border border-border bg-panel px-2 py-1 text-xs text-zinc-300 focus:border-zinc-500 focus:outline-none"
        >
          {SORTS.map((s) => (
            <option key={s.key} value={s.key}>
              {s.label}
            </option>
          ))}
        </select>
      }
    >
      <CategoryRail
        categories={catalogue?.categories ?? []}
        entries={entries}
        selected={category}
        onSelect={setCategory}
      />

      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {importError ? (
          <p className="mb-3 text-[11px] text-red-300/90">{importError}</p>
        ) : catalogue?.stale && entries.length > 0 ? (
          <p className="mb-3 text-[11px] text-amber-300/80">
            Showing a saved copy. Couldn&apos;t reach inlinestudio.art.
          </p>
        ) : null}

        {error ? (
          <div className="flex h-full flex-col items-center justify-center gap-3">
            <p className="text-[12px] text-zinc-400">{error}</p>
            <button
              onClick={() => void load(true)}
              className="rounded-md border border-border px-3 py-1.5 text-xs text-zinc-200 hover:bg-panel"
            >
              Retry
            </button>
          </div>
        ) : loading && entries.length === 0 ? (
          <p className="py-10 text-center text-[12px] text-zinc-500">Loading…</p>
        ) : shown.length === 0 ? (
          <p className="py-10 text-center text-[12px] text-zinc-500">
            Nothing published in this category yet.
          </p>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(200px,1fr))] items-start gap-2.5">
            {shown.map((card) => (
              <WorkflowCard
                key={card.slug}
                card={card}
                importing={importing === card.slug}
                onImport={() => {
                  // Only the detail carries the graph, so importing is what fetches it.
                  setImporting(card.slug)
                  void ensureDetail(card.slug).then((detail) => {
                    if (detail) onImport(detail)
                    // Silence here read as a hang: the card sat on "Importing…" for the length of
                    // the timeout and then simply stopped.
                    else setImportError(`Couldn't load ${card.title}. Check your connection.`)
                  })
                }}
              />
            ))}
          </div>
        )}
      </div>
    </Modal>
  )
}
