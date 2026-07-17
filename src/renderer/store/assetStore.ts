/**
 * Library state: the open project's folders + assets, the folder the user is
 * currently browsing, and the selected asset (shown in Preview). Work happens in
 * main via studio().assets / studio().folders.
 */
import { create } from 'zustand'
import type { Asset, AssetFolder } from '@shared/types'
import { ipcErrorMessage } from '../lib/ipcError'
import { importFilesToLibrary } from '../lib/importFiles'
import { studio } from '@/lib/studio'

interface AssetState {
  folders: AssetFolder[]
  assets: Asset[]
  /** Folder being browsed; null = library root. */
  currentFolderId: string | null
  selectedId: string | null
  loading: boolean
  error: string | null

  load: () => Promise<void>
  import: () => Promise<void>
  importPaths: (paths: string[]) => Promise<void>
  /** Import dropped/picked File objects (real paths under Electron, upload in the browser). */
  importFiles: (files: File[]) => Promise<void>
  remove: (assetId: string) => Promise<void>
  createFolder: (name: string) => Promise<void>
  deleteFolder: (id: string) => Promise<void>
  navigate: (folderId: string | null) => void
  select: (id: string | null) => void
  reset: () => void
}

export const useAssetStore = create<AssetState>((set, get) => ({
  folders: [],
  assets: [],
  currentFolderId: null,
  selectedId: null,
  loading: false,
  error: null,

  load: async () => {
    set({ loading: true, error: null })
    try {
      const [foldersRes, assetsRes] = await Promise.all([
        studio().folders.list(),
        studio().assets.list(),
      ])
      if (!foldersRes.ok) return set({ loading: false, error: foldersRes.error })
      if (!assetsRes.ok) return set({ loading: false, error: assetsRes.error })
      set({ folders: foldersRes.value, assets: assetsRes.value, loading: false })
    } catch (e) {
      set({ loading: false, error: ipcErrorMessage(e) })
    }
  },

  import: async () => {
    set({ loading: true, error: null })
    try {
      const res = await studio().assets.importDialog(get().currentFolderId)
      if (!res.ok) return set({ loading: false, error: res.error })
      const added = res.value
      set((s) => ({
        assets: [...added, ...s.assets],
        selectedId: added[0]?.id ?? s.selectedId,
        loading: false,
      }))
    } catch (e) {
      set({ loading: false, error: ipcErrorMessage(e) })
    }
  },

  importFiles: async (files: File[]) => {
    if (files.length === 0) return
    set({ loading: true, error: null })
    try {
      const added = await importFilesToLibrary(files, get().currentFolderId)
      set((s) => ({
        assets: [...added, ...s.assets],
        selectedId: added[0]?.id ?? s.selectedId,
        loading: false,
      }))
    } catch (e) {
      set({ loading: false, error: ipcErrorMessage(e) })
    }
  },

  // Import OS files (e.g. dropped from Finder/Explorer) into the current folder.
  importPaths: async (paths: string[]) => {
    if (paths.length === 0) return
    set({ loading: true, error: null })
    try {
      const res = await studio().assets.importPaths(paths, get().currentFolderId)
      if (!res.ok) return set({ loading: false, error: res.error })
      const added = res.value
      set((s) => ({
        assets: [...added, ...s.assets],
        selectedId: added[0]?.id ?? s.selectedId,
        loading: false,
      }))
    } catch (e) {
      set({ loading: false, error: ipcErrorMessage(e) })
    }
  },

  remove: async (assetId: string) => {
    set({ error: null })
    try {
      const res = await studio().assets.delete(assetId)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({
        assets: s.assets.filter((a) => a.id !== assetId),
        selectedId: s.selectedId === assetId ? null : s.selectedId,
      }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  createFolder: async (name: string) => {
    set({ error: null })
    try {
      const res = await studio().folders.create({
        name,
        parentId: get().currentFolderId,
      })
      if (!res.ok) return set({ error: res.error })
      set((s) => ({ folders: [...s.folders, res.value] }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  deleteFolder: async (id: string) => {
    set({ error: null })
    try {
      const res = await studio().folders.delete(id)
      if (!res.ok) return set({ error: res.error })
      // Reload so reparented assets/subfolders reflect the new structure.
      await get().load()
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  navigate: (folderId) => set({ currentFolderId: folderId, selectedId: null }),

  select: (id) => set({ selectedId: id }),

  reset: () =>
    set({ folders: [], assets: [], currentFolderId: null, selectedId: null, error: null }),
}))

/**
 * Subscribe to the backend's "library changed" push (a video poster or playable transcode
 * finished) and refresh the library. Called once from App's effect rather than at module load, so
 * the injected backend client is set first (the browser shell has no window.inlineStudio fallback).
 */
export function subscribeToLibraryChanges(): () => void {
  return studio().events.onLibraryChanged(() => {
    void useAssetStore.getState().load()
  })
}

/** The chain of folders from root to the current one (for breadcrumbs). */
export function folderPath(folders: AssetFolder[], currentId: string | null): AssetFolder[] {
  const byId = new Map(folders.map((f) => [f.id, f]))
  const path: AssetFolder[] = []
  let id = currentId
  while (id) {
    const folder = byId.get(id)
    if (!folder) break
    path.unshift(folder)
    id = folder.parentId
  }
  return path
}
