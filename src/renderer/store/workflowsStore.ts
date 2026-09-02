/**
 * The published workflow catalogue behind the Workflows popup.
 *
 * A detail is fetched only when a workflow is imported, since the graph lives nowhere else, and is
 * kept for the popup's lifetime: the detail endpoint is uncached upstream and counts a view on
 * every call, so a re-import must not count twice.
 */
import { create } from 'zustand'
import type { WorkflowCatalogue, WorkflowDetail, WorkflowSort } from '@shared/types'
import { studio } from '@/lib/studio'

interface WorkflowsState {
  open: boolean
  sort: WorkflowSort
  /** Selected category slug, or null for All. */
  category: string | null
  catalogue: WorkflowCatalogue | null
  loading: boolean
  /** Set when the site was unreachable and nothing was cached. */
  error: string | null
  /** Details by slug, fetched on import. */
  details: Record<string, WorkflowDetail>
  importing: string | null
  /** Why the last import could not start. Cleared by the next attempt. */
  importError: string | null
  setOpen: (open: boolean) => void
  setSort: (sort: WorkflowSort) => void
  setCategory: (category: string | null) => void
  load: (refresh?: boolean) => Promise<void>
  /** The detail, fetching it once if this is the first ask. Null when the site is unreachable. */
  ensureDetail: (slug: string) => Promise<WorkflowDetail | null>
  setImporting: (slug: string | null) => void
  setImportError: (message: string | null) => void
}

export const useWorkflowsStore = create<WorkflowsState>((set, get) => ({
  open: false,
  sort: 'views',
  category: null,
  catalogue: null,
  loading: false,
  error: null,
  details: {},
  importing: null,
  importError: null,

  setOpen: (open) => {
    set({ open, importError: open ? null : get().importError })
    if (open && get().catalogue === null) void get().load()
  },

  setSort: (sort) => {
    set({ sort })
    void get().load()
  },

  setCategory: (category) => set({ category }),

  load: async (refresh = false) => {
    set({ loading: true, error: null })
    const result = await studio().workflows.list(get().sort, refresh)
    if (!result.ok) {
      set({ loading: false, error: result.error })
      return
    }
    const catalogue = result.value
    set({
      catalogue,
      loading: false,
      error:
        catalogue.stale && catalogue.entries.length === 0
          ? "Couldn't reach inlinestudio.art."
          : null,
    })
  },

  ensureDetail: async (slug) => {
    const held = get().details[slug]
    if (held) return held

    const result = await studio().workflows.detail(slug)
    if (!result.ok || !result.value) return null

    const detail = result.value
    set((s) => ({ details: { ...s.details, [slug]: detail } }))
    return detail
  },

  // Starting an import clears the last failure, so a retry does not read as still broken.
  setImporting: (importing) => set(importing ? { importing, importError: null } : { importing }),

  setImportError: (importError) => set({ importError, importing: null }),
}))
