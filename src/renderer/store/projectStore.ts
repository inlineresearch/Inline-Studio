/**
 * Feature-scoped store for the currently open project + recents. Components
 * render from this; all real work happens in main via the IPC bridge.
 */
import { create } from 'zustand'
import type { Project, RecentProject } from '@shared/types'
import { ipcErrorMessage } from '../lib/ipcError'
import { studio } from '@/lib/studio'

interface ProjectState {
  current: Project | null
  recents: RecentProject[]
  loading: boolean
  error: string | null
  /** A transient success message (e.g. "Exported to …"). */
  notice: string | null
  /** Path of the project currently being exported, if any. */
  exportingPath: string | null

  loadRecents: () => Promise<void>
  createProject: (name: string) => Promise<void>
  openFromDialog: () => Promise<void>
  openFromZip: () => Promise<void>
  openByPath: (path: string) => Promise<void>
  /** Export a project folder (by path) to a portable .zip via a save dialog. */
  exportProject: (path: string) => Promise<void>
  closeProject: () => void
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  current: null,
  recents: [],
  loading: false,
  error: null,
  notice: null,
  exportingPath: null,

  loadRecents: async () => {
    const res = await studio().project.listRecent()
    if (res.ok) set({ recents: res.value })
  },

  createProject: async (name: string) => {
    set({ loading: true, error: null })
    const dir = await studio().dialog.pickDirectory()
    if (!dir.ok) return set({ loading: false, error: dir.error })
    if (dir.value === null) return set({ loading: false })

    const res = await studio().project.create({ name, parentDir: dir.value })
    if (!res.ok) return set({ loading: false, error: res.error })
    set({ current: res.value, loading: false })
    void get().loadRecents()
  },

  openFromDialog: async () => {
    set({ loading: true, error: null })
    const res = await studio().project.openDialog()
    if (!res.ok) return set({ loading: false, error: res.error })
    if (res.value === null) return set({ loading: false })
    set({ current: res.value, loading: false })
    void get().loadRecents()
  },

  openFromZip: async () => {
    set({ loading: true, error: null })
    try {
      const res = await studio().project.openZip()
      if (!res.ok) return set({ loading: false, error: res.error })
      if (res.value === null) return set({ loading: false })
      set({ current: res.value, loading: false })
      void get().loadRecents()
    } catch (e) {
      set({ loading: false, error: ipcErrorMessage(e) })
    }
  },

  openByPath: async (path: string) => {
    set({ loading: true, error: null })
    const res = await studio().project.open(path)
    if (!res.ok) return set({ loading: false, error: res.error })
    set({ current: res.value, loading: false })
    void get().loadRecents()
  },

  exportProject: async (path: string) => {
    set({ exportingPath: path, error: null, notice: null })
    try {
      const res = await studio().project.export(path)
      if (!res.ok) return set({ exportingPath: null, error: res.error })
      // null = the user cancelled the save dialog.
      set({ exportingPath: null, notice: res.value ? `Exported to ${res.value.path}` : null })
    } catch (e) {
      set({ exportingPath: null, error: ipcErrorMessage(e) })
    }
  },

  closeProject: () => set({ current: null, error: null, notice: null }),
}))
