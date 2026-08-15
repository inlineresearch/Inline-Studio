/**
 * Saved characters: the `.char` library in `models/characters/`, which is global rather than
 * per-project. Work happens in Core via studio().characters; this only holds the list, the open
 * editor, and whether an encode is in flight (it runs two embedding passes, so it is not instant).
 */
import { create } from 'zustand'
import type { CharacterDetail, CharacterSummary } from '@shared/types'
import { ipcErrorMessage } from '../lib/ipcError'
import { studio } from '@/lib/studio'

interface CharacterState {
  characters: CharacterSummary[]
  /** The character open in the editor, with its reference URLs resolved. */
  editing: CharacterDetail | null
  loading: boolean
  /** An encode is running. Creating and editing both recompile, so both set this. */
  busy: boolean
  error: string | null

  load: () => Promise<void>
  create: (name: string, assetIds: string[], description: string) => Promise<boolean>
  createFromTake: (takeId: string, name: string) => Promise<boolean>
  open: (file: string) => Promise<void>
  closeEditor: () => void
  rename: (file: string, name: string) => Promise<void>
  setDescription: (file: string, description: string) => Promise<void>
  addRefs: (file: string, assetIds: string[]) => Promise<void>
  removeRef: (file: string, index: number) => Promise<void>
  remove: (file: string) => Promise<void>
  importFile: (file: File) => Promise<void>
  reset: () => void
}

/** Merge an updated summary into the list without reordering it under the user's cursor. */
function merged(list: CharacterSummary[], updated: CharacterSummary): CharacterSummary[] {
  const index = list.findIndex((c) => c.file === updated.file)
  if (index < 0) return [updated, ...list]
  return list.map((c, i) => (i === index ? updated : c))
}

export const useCharacterStore = create<CharacterState>((set, get) => ({
  characters: [],
  editing: null,
  loading: false,
  busy: false,
  error: null,

  load: async () => {
    set({ loading: true, error: null })
    try {
      const res = await studio().characters.list()
      if (!res.ok) return set({ loading: false, error: res.error })
      set({ characters: res.value, loading: false })
    } catch (e) {
      set({ loading: false, error: ipcErrorMessage(e) })
    }
  },

  create: async (name, assetIds, description) => {
    set({ busy: true, error: null })
    try {
      const res = await studio().characters.create({ name, assetIds, description })
      if (!res.ok) {
        set({ busy: false, error: res.error })
        return false
      }
      set((s) => ({ characters: merged(s.characters, res.value), busy: false }))
      return true
    } catch (e) {
      set({ busy: false, error: ipcErrorMessage(e) })
      return false
    }
  },

  createFromTake: async (takeId, name) => {
    set({ busy: true, error: null })
    try {
      const res = await studio().characters.createFromTake(takeId, name)
      if (!res.ok) {
        set({ busy: false, error: res.error })
        return false
      }
      set((s) => ({ characters: merged(s.characters, res.value), busy: false }))
      return true
    } catch (e) {
      set({ busy: false, error: ipcErrorMessage(e) })
      return false
    }
  },

  open: async (file) => {
    set({ error: null })
    try {
      const res = await studio().characters.get(file)
      if (!res.ok) return set({ error: res.error })
      set({ editing: res.value })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  closeEditor: () => set({ editing: null }),

  rename: async (file, name) => {
    await applyEdit(set, get, () => studio().characters.rename(file, name))
  },

  setDescription: async (file, description) => {
    await applyEdit(set, get, () => studio().characters.setDescription(file, description))
  },

  addRefs: async (file, assetIds) => {
    await applyEdit(set, get, () => studio().characters.addRefs(file, assetIds), true)
  },

  removeRef: async (file, index) => {
    await applyEdit(set, get, () => studio().characters.removeRef(file, index), true)
  },

  remove: async (file) => {
    set({ error: null })
    try {
      const res = await studio().characters.delete(file)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({
        characters: s.characters.filter((c) => c.file !== file),
        editing: s.editing?.file === file ? null : s.editing,
      }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  importFile: async (file) => {
    set({ busy: true, error: null })
    try {
      // /upload routes through the asset importer, which drops unknown extensions, so a .char
      // needs its own endpoint.
      const response = await fetch(`/upload/character?name=${encodeURIComponent(file.name)}`, {
        method: 'POST',
        body: file,
      })
      const result = (await response.json()) as { ok: boolean; error?: string }
      if (!result.ok) return set({ busy: false, error: result.error ?? 'Import failed' })
      set({ busy: false })
      await get().load()
    } catch (e) {
      set({ busy: false, error: ipcErrorMessage(e) })
    }
  },

  reset: () => set({ characters: [], editing: null, error: null, busy: false }),
}))

type Setter = (
  partial: Partial<CharacterState> | ((s: CharacterState) => Partial<CharacterState>),
) => void

/** Every edit returns the rewritten summary, so the list and the open editor stay in step. */
async function applyEdit(
  set: Setter,
  get: () => CharacterState,
  call: () => Promise<{ ok: true; value: CharacterSummary } | { ok: false; error: string }>,
  recompiles = false,
): Promise<void> {
  set({ error: null, ...(recompiles ? { busy: true } : {}) })
  try {
    const res = await call()
    if (!res.ok) return set({ busy: false, error: res.error })
    set((s) => ({ characters: merged(s.characters, res.value), busy: false }))
    // Adding or removing a reference changes refUrls, so the editor has to be refetched.
    if (get().editing?.file === res.value.file) await get().open(res.value.file)
  } catch (e) {
    set({ busy: false, error: ipcErrorMessage(e) })
  }
}

/**
 * Refresh when Core says the library moved. Called once from App's effect rather than at module
 * load, so the injected backend client is set first.
 */
export function subscribeCharacterChanges(): () => void {
  return studio().events.onCharactersChanged(() => {
    void useCharacterStore.getState().load()
  })
}
